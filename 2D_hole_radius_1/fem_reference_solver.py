"""
Standalone FEniCSx FEM Solver — 2-D Plate with a Central Hole
=============================================================
Solves the homogeneous plate with circular holes (voids) under a uniform
right-edge traction. This is the FEM reference on its own, without any GNN
involvement; the Stage-3 comparison script drives the SAME kernel
(``fem_kernel.solve_composite``) so the two can never diverge in physics.

  MATERIAL_MODEL = 'LE'  — plane-stress linear elasticity, small strain (default)
  MATERIAL_MODEL = 'NH'  — compressible Neo-Hookean, rigorous plane stress

Boundary conditions (identical to the PI-GNN training problem):
  - x = x_min        : roller, ux = 0 (uy free)
  - top-left corner  : pin, uy = 0 (removes the last rigid-body mode)
  - x = x_max        : uniform Neumann traction (T, 0) in +x
  - hole rims        : free (traction-free exterior surfaces)

Outputs (per loading case):
  - nodes_<model>_sigma_<T>.csv      : per-node coordinates, displacement, stress
  - edges_<model>_sigma_<T>.csv      : edge graph with geometry
  - simulation_<model>_sigma_<T>.vtu : VTK file for ParaView (all fields)
  - summary_<model>_sigma_<T>.txt    : applied force, Kt, peak values, timings
"""

import argparse
import meshio
import numpy as np
import pandas as pd
import os
import json
from mpi4py import MPI
from pathlib import Path
from dolfinx import mesh as dmesh, fem, plot
import ufl
import time
from scipy.spatial import cKDTree

from fem_kernel import (solve_composite, build_traction_measure,
                        applied_force, loaded_edge_length)

# ============================================================
# INPUT / OUTPUT CONFIG
# ============================================================

MESH_DIR   = Path("mesh_plate_hole")
OUTPUT_DIR = "plate_fem_reference"

# ── Material model ────────────────────────────────────────────
# 'LE' = small-strain plane-stress linear elasticity (the physical choice here)
# 'NH' = finite-strain compressible Neo-Hookean, rigorous plane stress
MATERIAL_MODEL = 'LE'

# APPLIED TRACTION(S) on the right edge, in SI Pa.
#   1e6 Pa = 1 MPa, which is the PI-GNN Config.TRACTION_MAGNITUDE.
# Against the 210 GPa steel below this is a nominal strain of T/E = 4.8e-6, so
# the far field carries ~1 MPa of sigma_xx and the hole rim ~3x that.
SIGMA_LIST = [1e6]   # Pa

# Material properties (SI: Pa) — single homogeneous matrix phase, matching the
# PI-GNN Config (210 GPa steel, nu = 0.3) so this standalone solve reproduces
# the SAME problem the pipeline trains and compares against.
E_MATRIX    = 210e9   # Pa  (= 210e3 MPa)
NU          = 0.3

# Mesh coordinates are authored in mm; dolfinx works in SI metres.
MM_TO_M = 1e-3

# Stress display unit for the summary/CSV column labels.
STRESS_UNIT = "MPa"
UNIT_TO_PA = 1e6        # MPa -> Pa
PA_TO_UNIT = 1.0 / UNIT_TO_PA


# ============================================================
# MESH LOADING
# ============================================================

def _load_plate_mesh(mesh_dir: Path):
    """
    Read the hole-plate triangulation plus the geometry the meshing stage
    recorded.

    ``mesh_summary.json`` carries the plate size and hole configuration. Those
    are read rather than re-derived so this solver reports the same geometry the
    trainer and the comparison do.
    """
    vtu_path = mesh_dir / "composite_mesh.vtu"
    if not vtu_path.exists():
        raise FileNotFoundError(f"No composite_mesh.vtu in '{mesh_dir}'")

    summary_path = mesh_dir / "mesh_summary.json"
    info = {}
    if summary_path.exists():
        with open(summary_path) as f:
            info = json.load(f)

    m = meshio.read(str(vtu_path))
    tri_blocks = [cb.data for cb in m.cells if cb.type == "triangle"]
    if not tri_blocks:
        raise ValueError(f"No triangle cells in '{vtu_path}'")
    tris = np.vstack(tri_blocks).astype(np.int32)
    points_mm = m.points[:, :2].astype(np.float64)

    mat_path = mesh_dir / "element_material_ids.npy"
    if mat_path.exists():
        mat_ids = np.load(mat_path).astype(np.int32)
    else:
        mat_ids = np.zeros(len(tris), dtype=np.int32)

    return points_mm, tris, mat_ids, info


def _build_domain(points_mm, tris, work_dir: Path):
    """Hand the mm-authored triangulation to dolfinx in SI metres."""
    work_dir.mkdir(parents=True, exist_ok=True)
    xdmf_path = work_dir / "fem_mesh.xdmf"
    meshio.write(
        str(xdmf_path),
        meshio.Mesh(points=points_mm * MM_TO_M, cells=[("triangle", tris)]))
    from dolfinx import io
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as xf:
        return xf.read_mesh(name="Grid")


# ============================================================
# PROCESS MESH (solve + save)
# ============================================================

def process_mesh(mesh_dir: Path, output_dir: str, material_model: str,
                 sigma_list) -> None:
    """Solve the plate for every requested traction and write the FEM outputs.

    Args:
        mesh_dir: Case directory holding ``composite_mesh.vtu`` and the mesh
            arrays.
        output_dir: Directory the CSV/VTU results are written into; created if
            it does not exist.
        material_model: ``'LE'`` or ``'NH'``.
        sigma_list: Iterable of applied right-edge tractions, in SI pascals.
            One solve and one output set is produced per entry.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    t_start = time.perf_counter()
    points_mm, tris, mat_ids_file, info = _load_plate_mesh(mesh_dir)
    L_mm = info.get("plate_size") or float(max(np.ptp(points_mm[:, 0]),
                                               np.ptp(points_mm[:, 1])))

    print("=" * 70)
    print("  STANDALONE FEniCSx FEM — 2-D PLATE WITH A CENTRAL HOLE")
    print("=" * 70)
    print(f"  Mesh dir       : {mesh_dir}")
    print(f"  Material model : {material_model}")
    print(f"  Plate size     : {L_mm} mm")
    print(f"  Holes          : {info.get('holes', 'n/a')}")
    print(f"  E / nu         : {E_MATRIX:.4g} Pa / {NU}")
    print(f"  Nodes / tris   : {len(points_mm)} / {len(tris)}")

    domain = _build_domain(points_mm, tris, out / "_work")
    t_mesh_end = time.perf_counter()

    coords = domain.geometry.x                  # (N, >=2) in m
    N      = coords.shape[0]
    conn   = domain.geometry.dofmap.reshape(-1, 3)

    # dolfinx re-orders cells on read, so the per-element material ids must be
    # matched by centroid rather than assumed aligned with the file order.
    fem_centroids_mm = coords[conn][:, :, :2].mean(axis=1) / MM_TO_M
    src_centroids_mm = points_mm[tris].mean(axis=1)
    d, fem_to_src = cKDTree(src_centroids_mm).query(fem_centroids_mm)
    if d.max() > 1e-6:
        print(f"  WARNING: max centroid mismatch = {d.max():.3e} mm")
    mat_ids = mat_ids_file[fem_to_src]

    # Single homogeneous matrix phase (the voids were removed at mesh time), so
    # E and nu are uniform over every element.
    V_dg = fem.functionspace(domain, ("DG", 0))
    E_func  = fem.Function(V_dg)
    nu_func = fem.Function(V_dg)
    E_func.x.array[:]  = E_MATRIX
    nu_func.x.array[:] = NU

    fdim  = domain.topology.dim - 1
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    y_max = float(coords[:, 1].max())
    x_tol = max(1e-14, 1e-8 * abs(x_max - x_min))
    y_tol = max(1e-14, 1e-8 * abs(coords[:, 1].max() - coords[:, 1].min()))

    left_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], x_min, atol=x_tol))
    right_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], x_max, atol=x_tol))
    # The pin is a VERTEX (dim-0 entity): a facet search finds nothing there and
    # would leave uy with an unconstrained rigid-body mode, which is singular
    # for the LE solve and non-convergent for the NH Newton solve.
    domain.topology.create_connectivity(0, domain.topology.dim)
    corner_verts = dmesh.locate_entities_boundary(
        domain, 0,
        lambda x: np.logical_and(np.isclose(x[0], x_min, atol=x_tol),
                                 np.isclose(x[1], y_max, atol=y_tol)))
    print(f"  Roller facets  : {len(left_facets)}")
    print(f"  Loaded facets  : {len(right_facets)}")
    print(f"  Pinned vertices: {len(corner_verts)}")

    for sigma_SI in sigma_list:
        tag = f"{sigma_SI * PA_TO_UNIT:g}{STRESS_UNIT}".replace(".", "p").replace("-", "m")
        print("\n" + "-" * 70)
        print(f"  Solving traction T = {sigma_SI:g} Pa "
              f"({sigma_SI * PA_TO_UNIT:g} {STRESS_UNIT})  | model = {material_model}")
        print("-" * 70)

        t0 = time.perf_counter()
        uh, sigma_fn = solve_composite(
            domain, E_func, nu_func, material_model,
            left_facets=left_facets, corner_verts=corner_verts,
            right_facets=right_facets,
            traction=sigma_SI, fdim=fdim,
        )
        solve_time = time.perf_counter() - t0
        print(f"  Solve time: {solve_time:.4f} s")

        _save_outputs(domain, coords, conn, uh, sigma_fn, V_dg, mat_ids,
                      L_mm, sigma_SI, tag, material_model, solve_time,
                      right_facets, fdim, out)

    t_end = time.perf_counter()
    print("\n" + "=" * 70)
    print(f"  DONE — outputs in {out}")
    print(f"  Time (mesh): {t_mesh_end - t_start:.4f}s  |  "
          f"Time (solve+save): {t_end - t_mesh_end:.4f}s  |  "
          f"Total: {t_end - t_start:.4f}s")
    print("=" * 70)


# ============================================================
# OUTPUT HELPERS
# ============================================================

def _save_outputs(domain, coords, conn, uh, sigma_fn, V_dg, mat_ids,
                  L_mm, sigma_SI, tag, material_model, solve_time,
                  right_facets, fdim, out: Path):
    """Write the displacement, stress and summary outputs for one solve.

    Args:
        domain: DOLFINx mesh of the plate.
        coords: Geometry-ordered nodal coordinates in metres, shape (N, >=2).
        conn: Triangle connectivity, shape (E, 3).
        uh: Solved displacement Function (2 components).
        sigma_fn: Callable mapping ``uh`` to the 2x2 Cauchy stress expression.
        V_dg: DG0 function space the stress is interpolated into.
        mat_ids: Per-element material tag, shape (E,).
        L_mm: Plate size in mm, for the report header.
        sigma_SI: Applied traction in pascals.
        tag: Filename tag for this loading case.
        material_model: ``'LE'`` or ``'NH'``.
        solve_time: Seconds the kernel took, excluding I/O.
        right_facets: Loaded-edge facet indices, for the assembled-force check.
        fdim: Facet dimension.
        out: Output directory.
    """
    N = len(coords)
    coords_mm = coords[:, :2] / MM_TO_M

    # ── Displacements ──
    # uh.x.array is in function-space DOF order, which dolfinx reorders
    # independently of the geometry-node order (`coords` = domain.geometry.x),
    # and for NH `uh` is a collapsed sub-space with yet another ordering.
    # geom_vtk holds the DOF coordinates that align with the raw array; match
    # against those and reorder into geometry-node order so the field lines up
    # with `coords`, with the geometry-ordered stresses and with the VTU mesh.
    # Pairing displacement with geometry coordinates instead scrambles the field
    # (ux no longer ~0 on the left roller edge, spurious strain energy).
    _, _, geom_vtk = plot.vtk_mesh(uh.function_space)
    u_dof = uh.x.array.reshape((geom_vtk.shape[0], 2))
    _, dof_for_geom = cKDTree(geom_vtk[:, :2]).query(coords[:, :2])
    u_m  = u_dof[dof_for_geom][:N]
    u_mm = u_m / MM_TO_M                                  # m -> mm

    # ── Stresses (DG0, cell centred) ──
    sigma_expr = sigma_fn(uh)

    def _interp_raw(expr):
        """Interpolate a UFL scalar into DG0 and return it unconverted."""
        f = fem.Function(V_dg)
        f.interpolate(fem.Expression(expr, V_dg.element.interpolation_points()))
        return f.x.array.copy()

    def _interp_unit(expr):
        """Interpolate a UFL stress scalar into DG0 and convert Pa -> STRESS_UNIT."""
        return _interp_raw(expr) * PA_TO_UNIT

    sxx = _interp_unit(sigma_expr[0, 0])
    syy = _interp_unit(sigma_expr[1, 1])
    sxy = _interp_unit(sigma_expr[0, 1])
    # Plane-stress von Mises closed form; must match the GNN definition.
    vm  = _interp_unit(ufl.sqrt(
        sigma_expr[0, 0]**2 - sigma_expr[0, 0] * sigma_expr[1, 1]
        + sigma_expr[1, 1]**2 + 3.0 * sigma_expr[0, 1]**2))
    # In-plane Jacobian J = det F; 1.0 would be area-preserving. Dimensionless,
    # so it goes through _interp_raw rather than the Pa -> MPa conversion.
    F_c  = ufl.Identity(2) + ufl.grad(uh)
    Jdet = _interp_raw(ufl.det(F_c))

    def cell_to_node(cell_vals):
        """Average a per-element field onto the nodes for the CSV."""
        acc = np.zeros(N, dtype=np.float64)
        cnt = np.zeros(N, dtype=np.float64)
        for col in range(conn.shape[1]):
            np.add.at(acc, conn[:, col], cell_vals)
            np.add.at(cnt, conn[:, col], 1.0)
        return acc / np.maximum(cnt, 1.0)

    # ── Assembled load check ──
    ds = build_traction_measure(domain, fdim, right_facets)
    L_edge = loaded_edge_length(domain, ds) / MM_TO_M          # m -> mm
    F_x_SI = applied_force(domain, sigma_SI, ds)               # N per unit thickness
    Kt = float(np.max(sxx)) / (sigma_SI * PA_TO_UNIT)

    # ── CSV: nodes ──
    df = pd.DataFrame({
        "node_id": np.arange(N),
        "x_mm": coords_mm[:, 0], "y_mm": coords_mm[:, 1],
        "ux_mm": u_mm[:, 0], "uy_mm": u_mm[:, 1],
        "u_mag_mm": np.linalg.norm(u_mm, axis=1),
        f"sigma_xx_{STRESS_UNIT}": cell_to_node(sxx),
        f"sigma_yy_{STRESS_UNIT}": cell_to_node(syy),
        f"sigma_xy_{STRESS_UNIT}": cell_to_node(sxy),
        f"von_mises_{STRESS_UNIT}": cell_to_node(vm),
        "J_area_ratio": cell_to_node(Jdet),
    })
    csv_path = out / f"nodes_{material_model}_sigma_{tag}.csv"
    df.to_csv(csv_path, index=False, float_format="%.8g")
    print(f"    CSV     : {csv_path}")

    # ── CSV: edges ──
    domain.topology.create_connectivity(1, 0)
    edges = domain.topology.connectivity(1, 0).array.reshape(-1, 2)
    p1 = coords_mm[edges[:, 0]]
    p2 = coords_mm[edges[:, 1]]
    d_vec = p2 - p1
    pd.DataFrame({
        "src": edges[:, 0], "dst": edges[:, 1],
        "dx_mm": d_vec[:, 0], "dy_mm": d_vec[:, 1],
        "length_mm": np.linalg.norm(d_vec, axis=1),
    }).to_csv(out / f"edges_{material_model}_sigma_{tag}.csv", index=False,
              float_format="%.8g")

    # ── VTU ──
    vtu_path = out / f"simulation_{material_model}_sigma_{tag}.vtu"
    with open(vtu_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{N}" NumberOfCells="{len(conn)}">\n')
        f.write('      <Points>\n        <DataArray type="Float64" '
                'NumberOfComponents="3" format="ascii">\n')
        for x, y in coords_mm:
            f.write(f"          {x:.10g} {y:.10g} 0.0\n")
        f.write('        </DataArray>\n      </Points>\n')
        f.write('      <Cells>\n        <DataArray type="Int64" '
                'Name="connectivity" format="ascii">\n')
        for c in conn:
            f.write(f"          {c[0]} {c[1]} {c[2]}\n")
        f.write('        </DataArray>\n        <DataArray type="Int64" '
                'Name="offsets" format="ascii">\n')
        for i in range(1, len(conn) + 1):
            f.write(f"          {3 * i}\n")
        f.write('        </DataArray>\n        <DataArray type="UInt8" '
                'Name="types" format="ascii">\n')
        for _ in conn:
            f.write("          5\n")       # VTK_TRIANGLE
        f.write('        </DataArray>\n      </Cells>\n')

        # Vectors= sets the ACTIVE point vector, which is what ParaView's
        # "Warp By Vector" defaults to. Without it the array is present but the
        # deformed shape cannot be shown in one click.
        f.write('      <PointData Scalars="disp_mag_mm" Vectors="displacement_mm">\n')
        f.write('        <DataArray type="Float64" Name="displacement_mm" '
                'NumberOfComponents="3" format="ascii">\n')
        for a, b in u_mm:
            f.write(f"          {a:.10g} {b:.10g} 0.0\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Float64" Name="disp_mag_mm" '
                'format="ascii">\n')
        for v in np.linalg.norm(u_mm, axis=1):
            f.write(f"          {v:.10g}\n")
        f.write('        </DataArray>\n      </PointData>\n')

        f.write(f'      <CellData Scalars="von_mises_{STRESS_UNIT}">\n')
        for name, arr in ((f"sigma_xx_{STRESS_UNIT}", sxx),
                          (f"sigma_yy_{STRESS_UNIT}", syy),
                          (f"sigma_xy_{STRESS_UNIT}", sxy),
                          (f"von_mises_{STRESS_UNIT}", vm),
                          ("J_area_ratio", Jdet),
                          ("Material_ID", mat_ids.astype(float))):
            f.write(f'        <DataArray type="Float64" Name="{name}" '
                    'format="ascii">\n')
            for v in arr:
                f.write(f"          {float(v):.10g}\n")
            f.write('        </DataArray>\n')
        f.write('      </CellData>\n')
        f.write('    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')
    print(f"    VTU     : {vtu_path}")

    # ── Summary ──
    T_unit = sigma_SI * PA_TO_UNIT
    lines = [
        "=" * 74,
        "  Standalone FEM — 2-D Plate with a Central Hole",
        "=" * 74,
        f"  Material model  : {material_model}",
        f"  Formulation     : plane stress"
        + ("  (rigorous sigma_33 = 0 via the F33 condensation)"
           if material_model.upper() == "NH" else "  (reduced lambda_ps)"),
        f"  Plate size      : {L_mm} mm",
        f"  E / nu          : {E_MATRIX:.6g} Pa / {NU}",
        f"  Nodes / tris    : {N} / {len(conn)}",
        f"  Solve time      : {solve_time:.4f} s",
        "",
        "-- Loading ------------------------------------------------------------",
        f"  Applied traction: {sigma_SI:.6e} Pa  ({T_unit:.6g} {STRESS_UNIT})",
        f"  BCs             : x_min roller ux=0; top-left pin uy=0; "
        f"x_max traction +x; hole rims free",
        f"  Loaded edge len : {L_edge:.6f} mm",
        f"  Resultant Fx    : {F_x_SI:.6e} N/m of thickness  "
        f"(= T x L, assembled over the discretised edge)",
        "",
        "-- Response -----------------------------------------------------------",
        f"  Max |u|         : {np.linalg.norm(u_mm, axis=1).max():.6e} mm",
        f"  Max |ux| / |uy| : {np.abs(u_mm[:, 0]).max():.6e} / "
        f"{np.abs(u_mm[:, 1]).max():.6e} mm",
        f"  Kt = max(sxx)/T : {Kt:.4f}   "
        f"(Kirsch: 3.0 for a small circular hole in a wide plate)",
        f"  J = det F range : [{Jdet.min():.6f}, {Jdet.max():.6f}]  "
        f"(1 = area preserving)",
        "",
        f"-- Peak stresses ({STRESS_UNIT}) " + "-" * 44,
        f"  sigma_xx        : [{sxx.min():.6f}, {sxx.max():.6f}]",
        f"  sigma_yy        : [{syy.min():.6f}, {syy.max():.6f}]",
        f"  sigma_xy        : [{sxy.min():.6f}, {sxy.max():.6f}]",
        f"  von Mises       : {vm.max():.6f}",
        "",
        "-- Output files -------------------------------------------------------",
        f"  CSV : {csv_path}",
        f"  VTU : {vtu_path}",
        "=" * 74,
    ]
    sum_path = out / f"summary_{material_model}_sigma_{tag}.txt"
    sum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    Summary : {sum_path}")
    print("\n".join(lines[10:24]))


# ============================================================
# CLI
# ============================================================

def main():
    """Command-line entry point for the standalone FEniCSx FEM reference solve."""
    parser = argparse.ArgumentParser(
        description="Standalone FEniCSx FEM for the 2-D plate with a central hole.")
    parser.add_argument("--mesh-dir", type=Path, default=MESH_DIR,
                        help=f"Mesh case directory (default: {MESH_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--material-model", type=str, default=MATERIAL_MODEL,
                        choices=["LE", "NH", "le", "nh"],
                        help=f"Material model (default: {MATERIAL_MODEL})")
    parser.add_argument("--traction", type=float, nargs="+", default=SIGMA_LIST,
                        metavar="PA",
                        help=f"Applied right-edge traction(s), in SI Pa "
                             f"(default: {SIGMA_LIST}; 1e6 Pa = 1 MPa, the value "
                             f"the PI-GNN trains with).")
    args = parser.parse_args()

    process_mesh(args.mesh_dir, args.output_dir,
                 args.material_model.upper(), args.traction)


if __name__ == "__main__":
    main()
