"""
Standalone FEniCSx FEM Solver — 3-D Composite Rod in Torsion
============================================================
Solves the composite rod (matrix + straight prismatic inclusion) clamped at
z = 0 and loaded on z = H by a tangential traction that applies a pure torque.
This is the FEM reference on its own, without any GNN involvement; the Stage-3
comparison script drives the SAME kernel (``fem_kernel.solve_composite``)
so the two can never diverge in physics.

  MATERIAL_MODEL = 'NH'  — compressible Neo-Hookean, finite strain (default)
  MATERIAL_MODEL = 'LE'  — small-strain linear elasticity (reference option;
                           NOT valid at the twist levels used here)

Boundary conditions (identical to the PI-GNN training problem):
  - z = 0            : clamped, ux = uy = uz = 0
  - z = H            : prescribed TORQUE, applied as the tangential dead
                       traction t = (tau/R)(-(y-cy), (x-cx), 0); tau is derived
  - lateral surface  : free (traction-free)

Outputs (per loading case):
  - nodes_torque_<T>.csv      : per-node coordinates, displacement, twist, stress
  - simulation_torque_<T>.vtu : VTK file for ParaView (all fields)
  - summary_torque_<T>.txt    : torque, twist, warping and peak values

"""

import argparse
import meshio
import numpy as np
import pandas as pd
import os
import json
from mpi4py import MPI
from pathlib import Path
from dolfinx import mesh as dmesh, fem, io, plot
import ufl
import time
from scipy.spatial import cKDTree

from fem_kernel import (solve_composite, applied_torque,
                                loaded_face_area, build_traction_measure)
from train_pignn import MeshGeometry

# ============================================================
# INPUT / OUTPUT CONFIG
# ============================================================

MESH_DIR   = Path("mesh_rod_petal")
OUTPUT_DIR = "rod_fem_reference"

# ── Material model ────────────────────────────────────────────
# 'NH' = finite-strain compressible Neo-Hookean (the physical choice here)
# 'LE' = small-strain linear elasticity (reference / debug only)
MATERIAL_MODEL = 'NH'

# APPLIED TORQUE(S) about the rod axis, in SI N*m. The rim traction that
# delivers each one is derived from the mesh at run time (see
# MeshGeometry.traction_for_torque), so the requested torque is the one actually
# applied regardless of mesh resolution.
#   2.5e-7 N*m = 0.25 uN*m = 0.25 kPa*mm^3
# is the calibrated value that twists the shipped rod by ~24 degrees (rim shear
# gamma = 0.21, well past the linear-elastic range) while keeping the dead-load
# artefact small; see train_pignn.Config.APPLIED_TORQUE for
# the calibration table and why larger torques are not usable with this load.
TORQUE_LIST = [2.5e-7]   # N*m

# Material properties (SI: Pa) — soft solid, matching the PI-GNN Config.
E_MATRIX     = 1500.0    # Pa  (1.5 kPa matrix)
E_INCLUSION  = 5000.0    # Pa  (5.0 kPa inclusion; ratio 3.3333)
NU_MATRIX    = 0.40
NU_INCLUSION = 0.35

# Mesh coordinates are authored in mm; dolfinx works in SI metres.
MM_TO_M = 1e-3

# Stress display unit for the summary/CSV column labels.
STRESS_UNIT = "kPa"
UNIT_TO_PA = 1e3        # kPa -> Pa
PA_TO_UNIT = 1.0 / UNIT_TO_PA

# Torque unit conversion. A torque in <STRESS_UNIT>*mm^3 becomes SI N*m by
# multiplying by BOTH the stress factor and mm^3 -> m^3:
#     1 kPa*mm^3 = 1e3 Pa * 1e-9 m^3 = 1e-6 N*m = 1 uN*m
TORQUE_UNIT_TO_NM = UNIT_TO_PA * (MM_TO_M ** 3)   # 1e-6 for kPa/mm


# ============================================================
# MESH LOADING
# ============================================================

def _load_rod_mesh(mesh_dir: Path):
    """
    Read the extruded tetrahedral rod mesh plus the geometry the meshing stage
    recorded.

    ``mesh_summary.json`` carries the torsion axis, radius and height. Those are
    what the traction field is defined from, so they are read rather than
    re-derived — a bounding-box guess would put the axis off-centre for a petal,
    whose bbox is not centred on the shape.
    """
    vtu_path = mesh_dir / "composite_mesh.vtu"
    if not vtu_path.exists():
        raise FileNotFoundError(f"No composite_mesh.vtu in '{mesh_dir}'")

    summary_path = mesh_dir / "mesh_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No mesh_summary.json in '{mesh_dir}' — it carries the rod "
            f"axis/radius/height the traction field is defined from.")
    with open(summary_path) as f:
        info = json.load(f)

    m = meshio.read(str(vtu_path))
    tet_blocks = [cb.data for cb in m.cells if cb.type == "tetra"]
    if not tet_blocks:
        raise ValueError(f"No tetrahedral cells in '{vtu_path}'")
    tets = np.vstack(tet_blocks).astype(np.int32)
    points_mm = m.points[:, :3].astype(np.float64)

    mat_ids = np.load(mesh_dir / "element_material_ids.npy")
    if len(mat_ids) != len(tets):
        raise ValueError(
            f"element_material_ids.npy has {len(mat_ids)} entries but the mesh "
            f"has {len(tets)} tetrahedra.")

    return points_mm, tets, mat_ids, info


def _build_domain(points_mm, tets, work_dir: Path):
    """Convert the mm mesh to an SI dolfinx domain via XDMF."""
    work_dir.mkdir(parents=True, exist_ok=True)
    xdmf_path = work_dir / "fem_mesh.xdmf"
    meshio.write(str(xdmf_path),
                 meshio.Mesh(points=points_mm * MM_TO_M,
                             cells=[("tetra", tets)]))
    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as xf:
        domain = xf.read_mesh(name="Grid")
    return domain


# ============================================================
# SOLVE + OUTPUT
# ============================================================

def _cylindrical(sigma, centroids_mm, axis_mm):
    """(s_rr, s_tt, s_zz, s_tz, s_rz) in rod coordinates. See
    train_pignn.cylindrical_components for the convention."""
    dx = centroids_mm[:, 0] - axis_mm[0]
    dy = centroids_mm[:, 1] - axis_mm[1]
    r  = np.hypot(dx, dy)
    safe = r > 1e-12
    c = np.where(safe, dx / np.maximum(r, 1e-30), 1.0)
    s = np.where(safe, dy / np.maximum(r, 1e-30), 0.0)
    zed = np.zeros_like(c)
    e_r = np.stack([c, s, zed], axis=1)
    e_t = np.stack([-s, c, zed], axis=1)
    e_z = np.stack([zed, zed, np.ones_like(c)], axis=1)

    def project_stress(a, b):
        """Contract the stress tensor onto a pair of unit directions: a.sigma.b."""
        return np.einsum('ei,eij,ej->e', a, sigma, b)

    return project_stress(e_r, e_r), project_stress(e_t, e_t), \
        project_stress(e_z, e_z), project_stress(e_t, e_z), \
        project_stress(e_r, e_z)


def process_mesh(mesh_dir: Path, output_dir: str, material_model: str,
                 torque_list) -> None:
    """Solve the rod for every requested torque and write the FEM outputs.

    Args:
        mesh_dir: Case directory holding the mesh arrays and
            ``mesh_summary.json``.
        output_dir: Directory the CSV/VTU results are written into; created if
            it does not exist.
        material_model: ``'NH'`` or ``'LE'``.
        torque_list: Iterable of applied torques about the rod axis, in SI
            newton-metres. One solve and one output set is produced per entry.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    points_mm, tets, mat_ids_file, info = _load_rod_mesh(mesh_dir)
    R_mm  = info["plate_radius"]
    H_mm  = info["height"]
    axis_mm = (info["axis_x"], info["axis_y"])

    print("=" * 70)
    print("  STANDALONE FEniCSx FEM — 3-D COMPOSITE ROD IN TORSION")
    print("=" * 70)
    print(f"  Mesh dir       : {mesh_dir}")
    print(f"  Material model : {material_model}")
    print(f"  Rod R / H      : {R_mm} mm / {H_mm} mm   (H/2R = {H_mm/(2*R_mm):.3f})")
    print(f"  Torsion axis   : ({axis_mm[0]:.6f}, {axis_mm[1]:.6f}) mm")
    print(f"  E matrix / inc : {E_MATRIX} / {E_INCLUSION} Pa")
    print(f"  nu matrix / inc: {NU_MATRIX} / {NU_INCLUSION}")
    print(f"  Nodes / tets   : {len(points_mm)} / {len(tets)}")

    domain = _build_domain(points_mm, tets, out / "_work")
    coords = domain.geometry.x                  # (N, 3) in m
    N      = coords.shape[0]
    conn   = domain.geometry.dofmap.reshape(-1, 4)

    # dolfinx re-orders cells on read, so the per-element material ids must be
    # matched by centroid rather than assumed aligned with the file order.
    fem_centroids_mm = coords[conn].mean(axis=1) / MM_TO_M
    src_centroids_mm = points_mm[tets].mean(axis=1)
    d, fem_to_src = cKDTree(src_centroids_mm).query(fem_centroids_mm)
    if d.max() > 1e-6:
        print(f"  WARNING: max centroid mismatch = {d.max():.3e} mm")
    mat_ids = mat_ids_file[fem_to_src]

    V_dg = fem.functionspace(domain, ("DG", 0))
    E_func  = fem.Function(V_dg)
    nu_func = fem.Function(V_dg)
    E_func.x.array[:]  = np.where(mat_ids == 1, E_INCLUSION, E_MATRIX)
    nu_func.x.array[:] = np.where(mat_ids == 1, NU_INCLUSION, NU_MATRIX)

    fdim  = domain.topology.dim - 1
    z_min = float(coords[:, 2].min())
    z_max = float(coords[:, 2].max())
    z_tol = max(1e-14, 1e-8 * abs(z_max - z_min))

    fixed_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_min, atol=z_tol))
    loaded_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_max, atol=z_tol))
    print(f"  Clamped facets : {len(fixed_facets)}")
    print(f"  Loaded facets  : {len(loaded_facets)}")

    axis_m = (axis_mm[0] * MM_TO_M, axis_mm[1] * MM_TO_M)
    R_m    = R_mm * MM_TO_M

    # The rim traction that applies a unit torque on THIS mesh. The traction
    # field is linear in tau, so one inversion serves every requested torque.
    # Working from the assembled torque rather than the analytic tau*pi*R^3/2
    # means the requested torque is the one actually delivered — the polygonal
    # rim of the mesh otherwise leaves it ~0.7 % low, and mesh-dependent.
    lf = np.load(mesh_dir / "loading_surface_facets.npy")
    tau_per_unit_torque_mm, _ = MeshGeometry.traction_for_torque(
        points_mm, lf, 1.0, axis_mm, R_mm, verbose=False)

    for torque_SI in torque_list:
        # N*m -> <stress unit>*mm^3, the units the mm/kPa mesh works in.
        torque_unit = torque_SI / TORQUE_UNIT_TO_NM
        tau_unit = tau_per_unit_torque_mm * torque_unit      # in STRESS_UNIT
        tau = tau_unit * UNIT_TO_PA                          # -> Pa
        tag = f"{torque_SI:g}Nm".replace(".", "p").replace("-", "m")
        print("\n" + "-" * 70)
        print(f"  Solving torque = {torque_SI:g} N*m "
              f"({torque_unit:g} {STRESS_UNIT}*mm^3)")
        print(f"  -> rim traction tau = {tau:g} Pa ({tau_unit:g} {STRESS_UNIT})")
        print("-" * 70)

        t0 = time.perf_counter()
        uh, sigma_fn = solve_composite(
            domain, E_func, nu_func, material_model,
            fixed_facets=fixed_facets, loaded_facets=loaded_facets,
            traction=tau, axis=axis_m, plate_radius=R_m, fdim=fdim,
        )
        solve_time = time.perf_counter() - t0
        print(f"  Solve time: {solve_time:.3f} s")

        _save_outputs(domain, coords, conn, uh, sigma_fn, V_dg, mat_ids,
                      axis_mm, axis_m, R_mm, R_m, H_mm, tau, torque_SI, tag,
                      material_model, solve_time, fixed_facets, loaded_facets,
                      fdim, out)

    print("\n" + "=" * 70)
    print(f"  DONE — outputs in {out}")
    print("=" * 70)


def _save_outputs(domain, coords, conn, uh, sigma_fn, V_dg, mat_ids,
                  axis_mm, axis_m, R_mm, R_m, H_mm, tau, torque_SI, tag,
                  material_model, solve_time, fixed_facets, loaded_facets,
                  fdim, out: Path):
    """Write the displacement, stress and summary outputs for one solve.

    Args:
        domain: DOLFINx mesh of the rod.
        coords: Geometry-ordered nodal coordinates in metres, shape (N, 3).
        conn: Tetrahedron connectivity, shape (E, 4).
        uh: Solved displacement Function (3 components).
        sigma_fn: Callable mapping ``uh`` to the 3x3 Cauchy stress expression.
        V_dg: DG0 function space the stress is interpolated into.
        mat_ids: Per-element material tag, shape (E,).
        axis_mm: Rod axis ``(x, y)`` in mm.
        axis_m: The same axis in metres.
        R_mm: Rod radius in mm.
        R_m: Rod radius in metres.
        H_mm: Rod height in mm.
        tau: Rim traction magnitude in Pa.
        torque_SI: Applied torque in newton-metres.
        tag: Short label used in the output filenames.
        material_model: ``'NH'`` or ``'LE'``.
        solve_time: Wall-clock seconds spent in the solve.
        fixed_facets: Facet indices of the clamped end face.
        loaded_facets: Facet indices of the loaded end face.
        fdim: Facet dimension (2 for a tetrahedral mesh).
        out: Output directory.
    """
    N = coords.shape[0]

    _, _, geom_vtk = plot.vtk_mesh(uh.function_space)
    u_m  = uh.x.array.reshape((geom_vtk.shape[0], 3))[:N]
    # The displacement lives in function-space DOF order, which dolfinx orders
    # independently of the geometry nodes, so it must be matched against its own
    # DOF coordinates before being written alongside geometry-ordered fields.
    dof_coords = geom_vtk[:N, :3]
    d, dof_to_geom = cKDTree(dof_coords).query(coords[:N, :3])
    if d.max() > 1e-12:
        print(f"    WARNING: max DOF/geometry mismatch = {d.max():.3e} m")
    u_mm = u_m[dof_to_geom] / MM_TO_M

    # ── stress (DG0, per cell) ──
    def _interp(expr_ufl):
        """Interpolate a UFL expression into DG0 and return it as a NumPy array.

        Args:
            expr_ufl: Scalar UFL expression to evaluate per cell.

        Returns:
            ndarray: One value per cell, shape (E,).
        """
        e = fem.Expression(expr_ufl, V_dg.element.interpolation_points())
        fn = fem.Function(V_dg)
        fn.interpolate(e)
        return fn.x.array.copy()

    se = sigma_fn(uh)
    sig = np.zeros((len(conn), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            v = _interp(se[i, j]) * PA_TO_UNIT
            sig[:, i, j] = v
            sig[:, j, i] = v

    sxx, syy, szz = sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2]
    sxy, syz, sxz = sig[:, 0, 1], sig[:, 1, 2], sig[:, 0, 2]
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                 + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))

    centroids_mm = coords[conn].mean(axis=1) / MM_TO_M
    s_rr, s_tt, s_zz_c, s_tz, s_rz = _cylindrical(sig, centroids_mm, axis_mm)

    F_ufl = ufl.Identity(3) + ufl.grad(uh)
    Jdet = _interp(ufl.det(F_ufl))

    # ── twist / warping ──
    coords_mm = coords[:N, :3] / MM_TO_M
    x0 = coords_mm[:, 0] - axis_mm[0]
    y0 = coords_mm[:, 1] - axis_mm[1]
    r0 = np.hypot(x0, y0)
    th0 = np.arctan2(y0, x0)
    th1 = np.arctan2(y0 + u_mm[:, 1], x0 + u_mm[:, 0])
    twist_deg = np.degrees(np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0)))

    z_mm = coords_mm[:, 2]
    top = np.abs(z_mm - z_mm.max()) < 1e-9
    # Exclude nodes near the axis: there the angle is a ratio of two small
    # numbers and is dominated by noise even though the physical twist is equal.
    sel = top & (r0 > 0.3 * R_mm)
    twist_mean = float(twist_deg[sel].mean()) if sel.any() else float('nan')
    twist_max = float(np.abs(twist_deg[sel]).max()) if sel.any() else float('nan')

    ds = build_traction_measure(domain, fdim, loaded_facets)
    M_z_SI = applied_torque(domain, tau, axis_m, R_m, ds)
    M_z = M_z_SI * PA_TO_UNIT / (MM_TO_M ** 3)     # -> STRESS_UNIT * mm^3
    A_loaded = loaded_face_area(domain, ds) / (MM_TO_M ** 2)

    # ── CSV ──
    cell_to_node = lambda cv: np.bincount(
        conn.ravel(), weights=np.repeat(cv, 4), minlength=N) / np.maximum(
        np.bincount(conn.ravel(), minlength=N), 1)

    df = pd.DataFrame({
        "node_id": np.arange(N),
        "x_mm": coords_mm[:, 0], "y_mm": coords_mm[:, 1], "z_mm": coords_mm[:, 2],
        "r_mm": r0,
        "ux_mm": u_mm[:, 0], "uy_mm": u_mm[:, 1], "uz_mm": u_mm[:, 2],
        "u_mag_mm": np.linalg.norm(u_mm, axis=1),
        "twist_deg": twist_deg,
        f"sigma_xx_{STRESS_UNIT}": cell_to_node(sxx),
        f"sigma_yy_{STRESS_UNIT}": cell_to_node(syy),
        f"sigma_zz_{STRESS_UNIT}": cell_to_node(szz),
        f"sigma_xy_{STRESS_UNIT}": cell_to_node(sxy),
        f"sigma_yz_{STRESS_UNIT}": cell_to_node(syz),
        f"sigma_xz_{STRESS_UNIT}": cell_to_node(sxz),
        f"sigma_theta_z_{STRESS_UNIT}": cell_to_node(s_tz),
        f"sigma_rz_{STRESS_UNIT}": cell_to_node(s_rz),
        f"von_mises_{STRESS_UNIT}": cell_to_node(vm),
        "J_volume_ratio": cell_to_node(Jdet),
    })
    csv_path = out / f"nodes_torque_{tag}.csv"
    df.to_csv(csv_path, index=False, float_format="%.8g")
    print(f"    CSV     : {csv_path}")

    # ── VTU ──
    vtu_path = out / f"simulation_torque_{tag}.vtu"
    with open(vtu_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{N}" NumberOfCells="{len(conn)}">\n')
        f.write('      <Points>\n        <DataArray type="Float64" '
                'NumberOfComponents="3" format="ascii">\n')
        for x, y, z in coords_mm:
            f.write(f"          {x:.10g} {y:.10g} {z:.10g}\n")
        f.write('        </DataArray>\n      </Points>\n')
        f.write('      <Cells>\n        <DataArray type="Int64" '
                'Name="connectivity" format="ascii">\n')
        for c in conn:
            f.write(f"          {c[0]} {c[1]} {c[2]} {c[3]}\n")
        f.write('        </DataArray>\n        <DataArray type="Int64" '
                'Name="offsets" format="ascii">\n')
        for i in range(1, len(conn) + 1):
            f.write(f"          {4 * i}\n")
        f.write('        </DataArray>\n        <DataArray type="UInt8" '
                'Name="types" format="ascii">\n')
        for _ in conn:
            f.write("          10\n")
        f.write('        </DataArray>\n      </Cells>\n')

        # Vectors= sets the ACTIVE point vector, which is what ParaView's
        # "Warp By Vector" defaults to. Without it the array is present but the
        # deformed shape cannot be shown in one click.
        f.write('      <PointData Scalars="disp_mag_mm" Vectors="displacement_mm">\n')
        f.write('        <DataArray type="Float64" Name="displacement_mm" '
                'NumberOfComponents="3" format="ascii">\n')
        for a, b, c in u_mm:
            f.write(f"          {a:.10g} {b:.10g} {c:.10g}\n")
        f.write('        </DataArray>\n')
        for name, arr in (("disp_mag_mm", np.linalg.norm(u_mm, axis=1)),
                          ("uz_warping_mm", u_mm[:, 2]),
                          ("twist_deg", twist_deg),
                          ("radius_mm", r0)):
            f.write(f'        <DataArray type="Float64" Name="{name}" '
                    'format="ascii">\n')
            for v in arr:
                f.write(f"          {v:.10g}\n")
            f.write('        </DataArray>\n')
        f.write('      </PointData>\n')

        f.write(f'      <CellData Scalars="von_mises_{STRESS_UNIT}">\n')
        for name, arr in ((f"sigma_xx_{STRESS_UNIT}", sxx),
                          (f"sigma_yy_{STRESS_UNIT}", syy),
                          (f"sigma_zz_{STRESS_UNIT}", szz),
                          (f"sigma_xy_{STRESS_UNIT}", sxy),
                          (f"sigma_yz_{STRESS_UNIT}", syz),
                          (f"sigma_xz_{STRESS_UNIT}", sxz),
                          (f"sigma_rr_{STRESS_UNIT}", s_rr),
                          (f"sigma_tt_{STRESS_UNIT}", s_tt),
                          (f"sigma_theta_z_{STRESS_UNIT}", s_tz),
                          (f"sigma_rz_{STRESS_UNIT}", s_rz),
                          (f"von_mises_{STRESS_UNIT}", vm),
                          ("J_volume_ratio", Jdet),
                          ("Material_ID", mat_ids.astype(float))):
            f.write(f'        <DataArray type="Float64" Name="{name}" '
                    'format="ascii">\n')
            for v in arr:
                f.write(f"          {float(v):.10g}\n")
            f.write('        </DataArray>\n')
        f.write('      </CellData>\n')
        f.write('    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')
    print(f"    VTU     : {vtu_path}")

    # ── summary ──
    le_note = ""
    if material_model.upper() != "NH":
        le_note = ("\n  *** WARNING: LE (small-strain) at a twist of tens of degrees.\n"
                   "  *** Linear kinematics cannot represent finite rotation; these\n"
                   "  *** numbers are a code check only, not physically meaningful.\n")

    lines = [
        "=" * 74,
        "  Standalone FEM — 3-D Composite Rod in Torsion",
        "=" * 74,
        f"  Material model  : {material_model}" + le_note,
        f"  Rod R / H       : {R_mm} mm / {H_mm} mm",
        f"  Torsion axis    : ({axis_mm[0]:.6f}, {axis_mm[1]:.6f}) mm",
        f"  E matrix / inc  : {E_MATRIX} / {E_INCLUSION} Pa",
        f"  nu matrix / inc : {NU_MATRIX} / {NU_INCLUSION}",
        f"  Nodes / tets    : {N} / {len(conn)}",
        f"  Solve time      : {solve_time:.4f} s",
        "",
        "-- Loading ------------------------------------------------------------",
        f"  REQUESTED torque: {torque_SI:.6e} N*m  "
        f"({torque_SI / TORQUE_UNIT_TO_NM:.6f} {STRESS_UNIT}*mm^3)",
        f"  Realised torque : {M_z_SI:.6e} N*m  ({M_z:.6f} {STRESS_UNIT}*mm^3)",
        f"  -> rim traction : {tau:g} Pa  ({tau * PA_TO_UNIT:g} {STRESS_UNIT})  [derived]",
        f"  Loaded face area: {A_loaded:.6f} mm^2  "
        f"(analytic pi*R^2 = {np.pi * R_mm**2:.6f})",
        "",
        "-- Response -----------------------------------------------------------",
        f"  Mean twist z=H  : {twist_mean:.4f} deg",
        f"  Max  twist z=H  : {twist_max:.4f} deg",
        f"  Rim shear gamma : {np.radians(twist_mean) * R_mm / H_mm:.4f}  (phi*R/H)",
        f"  Max |u|         : {np.linalg.norm(u_mm, axis=1).max():.6f} mm",
        f"  Max |uz| warping: {np.abs(u_mm[:, 2]).max():.6f} mm",
        f"  J range         : [{Jdet.min():.4f}, {Jdet.max():.4f}]  (1 = incompressible)",
        "",
        f"-- Peak stresses ({STRESS_UNIT}) " + "-" * 44,
        f"  |sigma_theta_z| : {np.abs(s_tz).max():.6f}   (torsion shear)",
        f"  |sigma_rz|      : {np.abs(s_rz).max():.6f}",
        f"  |sigma_zz|      : {np.abs(szz).max():.6f}",
        f"  |sigma_rr|      : {np.abs(s_rr).max():.6f}",
        f"  |sigma_tt|      : {np.abs(s_tt).max():.6f}",
        f"  von Mises       : {vm.max():.6f}",
        "",
        "-- Output files -------------------------------------------------------",
        f"  CSV : {csv_path}",
        f"  VTU : {vtu_path}",
        "=" * 74,
    ]
    sum_path = out / f"summary_torque_{tag}.txt"
    sum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    Summary : {sum_path}")
    print("\n".join(lines[16:36]))


def main():
    """Command-line entry point for the standalone FEniCSx FEM reference solve."""
    parser = argparse.ArgumentParser(
        description="Standalone FEniCSx FEM for the 3-D composite rod in torsion.")
    parser.add_argument("--mesh-dir", type=Path, default=MESH_DIR,
                        help=f"Mesh case directory (default: {MESH_DIR})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--material-model", type=str, default=MATERIAL_MODEL,
                        choices=["LE", "NH", "le", "nh"],
                        help=f"Material model (default: {MATERIAL_MODEL})")
    parser.add_argument("--torque", type=float, nargs="+", default=TORQUE_LIST,
                        metavar="NM",
                        help=f"Applied torque(s) about the rod axis, in SI N*m "
                             f"(default: {TORQUE_LIST}; 2.5e-7 N*m = 0.25 uN*m). "
                             f"The rim traction that delivers it is derived from "
                             f"the mesh.")
    args = parser.parse_args()

    process_mesh(args.mesh_dir, args.output_dir,
                 args.material_model.upper(), args.torque)


if __name__ == "__main__":
    main()
