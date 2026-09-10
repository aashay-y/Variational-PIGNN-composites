"""
Inference + FEM comparison for the 2-D hole-plate PI-GNN.

Supports both material models:
  MATERIAL_MODEL = 'LE'  — plane-stress linear elasticity, small strain (default)
  MATERIAL_MODEL = 'NH'  — compressible Neo-Hookean, rigorous plane stress

For each mesh case:
  1. GNN prediction (displacement & in-plane stress) from the trained model.
  2. FEniCSx FEM on the same triangular mesh, same material model and load.
  3. Combined VTU with GNN fields, FEM fields and absolute errors.
  4. CSV with per-node error data.
  5. Summary TXT with L2/R2 metrics, the potential-energy comparison, and the
     plate-specific outputs (applied force, peak stress, Kt).

The plate is a single homogeneous matrix phase: the void interiors were removed
at mesh time, so every hole rim is an exterior traction-free free surface and
there is no material interface to split the nodal stress across.

One derived quantity is reported that carries the physics of this problem:

  * ``Kt = max(sigma_xx) / T`` — the stress concentration factor at the hole
    rim. Kirsch gives exactly 3 for a small circular hole in a wide plate under
    remote uniaxial tension, so it is a dimensionless, geometry-driven check
    that the predicted field has the right SHAPE around the rim and not merely
    the right magnitude far from it.

Unit conventions (all outputs):
  Displacement : mm   (GNN native; FEM m -> x1000)
  Stress       : POSTPROC_OUTPUT_UNIT, default MPa  (FEM Pa -> x PA_TO_UNIT)
  Force        : <stress unit> * mm  (per unit thickness)

Mesh coordinates are in mm. FEniCSx receives coordinates divided by 1000 (SI
metres); its displacement output is multiplied by 1000 to return mm.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import sys
import unittest.mock as _mock
from typing import Optional


# ── CPU thread pinning (MUST run before numpy/torch/dolfinx/PETSc import) ──────
# For a fair GNN-vs-FEM timing comparison, both the GNN (PyTorch/BLAS) and the
# FEM (OpenMP/MUMPS) must use the SAME number of CPU cores. OpenMP, the BLAS
# backends and MUMPS all read their thread count from environment variables at
# library-init time, so these have to be set BEFORE those libraries are imported
# below — hence this block sits at the very top and reads the count directly
# from the --threads flag (also honoured when this module is imported by
# run_pipeline, whose argv carries the same flag) or the
# COMPOSITE_NUM_THREADS env var.
def _peek_threads_arg() -> Optional[int]:
    """Read ``--threads`` straight off argv, before any heavy import.

    OpenMP, the BLAS backends and MUMPS latch their thread count when the
    library is first initialised, which happens on importing dolfinx/PETSc.
    The budget therefore has to be known before ``parse_args`` would normally
    run.

    Returns:
        int or None: The requested core count, or None if the flag is absent
        or not an integer.
    """
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                return None
        if a.startswith("--threads="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
    env = os.environ.get("COMPOSITE_NUM_THREADS")
    if env:
        try:
            return int(env)
        except ValueError:
            return None
    return None


_NUM_THREADS = _peek_threads_arg()
if _NUM_THREADS is not None and _NUM_THREADS > 0:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = str(_NUM_THREADS)

# The trainer imports SummaryWriter at module level; mock before importing.
if "torch.utils.tensorboard" not in sys.modules:
    sys.modules["torch.utils.tensorboard"] = _mock.MagicMock()

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from mpi4py import MPI
import ufl
from dolfinx import fem, io, mesh as dmesh, plot
import meshio

from fem_kernel import (solve_composite, build_traction_measure,
                        applied_force, loaded_edge_length)
from train_pignn import (
    Config as TrainConfig,
    BoundaryConditions,
    DisplacementGNN,
    CompositeEnergyCalculator,
    MeshGeometry,
    build_graph_data,
    load_mesh_data,
    load_mesh_summary,
    stress_concentration_factor,
    effective_output_displacement_scale,
)


# ── Physical units ────────────────────────────────────────────────────────────
# TrainConfig.YOUNGS_MODULUS_MATRIX and TRACTION_MAGNITUDE are the ACTUAL
# physical values expressed in POSTPROC_OUTPUT_UNIT. There is no reduced-modulus
# training trick: the GNN and the FEM see the same material, so GNN displacement
# is already physical mm and GNN stress is already in the config unit. The only
# conversion left is config unit <-> Pa for FEniCSx, which works in SI.

_UNIT_TO_PA = {"PA": 1.0, "KPA": 1e3, "MPA": 1e6, "GPA": 1e9}
_OUTPUT_UNIT = str(getattr(TrainConfig, "POSTPROC_OUTPUT_UNIT", "MPa")).upper()
if _OUTPUT_UNIT not in _UNIT_TO_PA:
    raise ValueError(
        f"POSTPROC_OUTPUT_UNIT={TrainConfig.POSTPROC_OUTPUT_UNIT!r} not supported; "
        f"expected one of {list(_UNIT_TO_PA)} (case-insensitive)."
    )
UNIT_TO_PA = _UNIT_TO_PA[_OUTPUT_UNIT]   # config unit -> Pa
PA_TO_UNIT = 1.0 / UNIT_TO_PA            # Pa -> config unit

# Label suffix for stress in all outputs (VTU array names, CSV columns, summary
# tables), derived from POSTPROC_OUTPUT_UNIT so it matches the stored values.
STRESS_UNIT = str(getattr(TrainConfig, "POSTPROC_OUTPUT_UNIT", "MPa"))

# Mesh length unit -> SI metres. Coordinates are authored in mm.
MM_TO_M = 1e-3

FEM_E_MATRIX  = TrainConfig.YOUNGS_MODULUS_MATRIX * UNIT_TO_PA   # Pa
FEM_NU_MATRIX = TrainConfig.POISSONS_RATIO_MATRIX
# The load is a uniform right-edge traction, in <config unit>; SI is x UNIT_TO_PA.
TRACTION_MAGNITUDE = float(TrainConfig.TRACTION_MAGNITUDE)       # config unit
FEM_SIGMA          = TRACTION_MAGNITUDE * UNIT_TO_PA             # Pa

# Material model — must match what the checkpoint was trained with.
MATERIAL_MODEL = TrainConfig.MATERIAL_MODEL   # 'LE' or 'NH'

REQUIRED_FILES = [
    "nodes.npy",
    "elements.npy",
    "node_features.npy",
    "loading_surface_facets.npy",
    "node_topology.npy",
    "element_material_ids.npy",
    "mesh_summary.json",
]

# The three independent in-plane stress components, in the order used everywhere
# (VTU arrays, CSV columns, metric tables) so the outputs stay aligned.
STRESS_KEYS = ("sigma_xx", "sigma_yy", "sigma_xy")
# Fields that also get a smoothed nodal projection and enter the error metrics.
NODAL_FIELDS = STRESS_KEYS + ("von_mises",)
# Everything written as raw (unsmoothed) element data in the VTU.
CELL_FIELDS = NODAL_FIELDS

# Fields carried through the R2 / signed-error tables.
_R2_FIELDS = ("sigma_xx", "sigma_yy", "sigma_xy", "von_mises")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ===========================================================================
# Utility helpers
# ===========================================================================

def _is_case_dir(path: Path) -> bool:
    """True when ``path`` holds every array a comparison case needs."""
    return path.is_dir() and all((path / f).exists() for f in REQUIRED_FILES)


def discover_case_dirs(input_dir: Path) -> list[Path]:
    """Find every complete mesh case at or below ``input_dir``.

    Args:
        input_dir: A case directory, or a tree containing several.

    Returns:
        list[Path]: Case directories, sorted case-insensitively by path.
    """
    if _is_case_dir(input_dir):
        return [input_dir]
    case_dirs = []
    for ff in input_dir.rglob("node_features.npy"):
        c = ff.parent
        if _is_case_dir(c):
            case_dirs.append(c)
    return sorted(set(case_dirs), key=lambda p: str(p).lower())


def _extract_epoch(path: Path) -> int:
    """Parse the epoch number out of a ``model_epoch_<n>.pt`` filename.

    Args:
        path: Checkpoint path.

    Returns:
        int: The epoch number, or -1 if the name does not match the pattern.
    """
    m = re.search(r"model_epoch_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def resolve_model_path(model: Optional[Path]) -> Path:
    """Choose which checkpoint to evaluate.

    Args:
        model: Explicit checkpoint path, or None to select automatically from
            ``TrainConfig.CHECKPOINT_DIR``.

    Returns:
        Path: ``model_best.pt`` when present, otherwise the highest-epoch
        ``model_epoch_*.pt``.

    Raises:
        FileNotFoundError: If the given path, the checkpoint directory, or any
            checkpoint within it is missing.
    """
    if model is not None:
        if not model.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model}")
        return model
    ckpt_dir = Path(TrainConfig.CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Prefer the best-on-trajectory model over the last epoch. The training loss
    # is the total potential energy, and the minimum principle makes the lowest
    # Pi the best admissible approximation — whereas the final epoch is an
    # arbitrary sample from Adam's oscillation about the minimum.
    best = ckpt_dir / "model_best.pt"
    if best.exists():
        return best

    candidates = list(ckpt_dir.glob("model_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(candidates, key=lambda p: (_extract_epoch(p), p.stat().st_mtime))


def _infer_material_model(checkpoint_path: Path) -> str:
    """Read material_model tag from checkpoint if present; fall back to config."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "material_model" in ckpt:
            return str(ckpt["material_model"]).upper()
    except Exception:
        pass
    return MATERIAL_MODEL


def _cell_to_nodal(num_nodes: int, elements: np.ndarray,
                   cell_vals: np.ndarray) -> np.ndarray:
    """Average a per-element field onto the nodes for smooth plotting.

    Args:
        num_nodes: Number of mesh nodes N.
        elements: Triangle connectivity, shape (E, 3).
        cell_vals: Per-element field, shape (E,).

    Returns:
        np.ndarray: Nodal field, shape (N,), float32.
    """
    accum = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)
    for col in range(elements.shape[1]):
        np.add.at(accum, elements[:, col], cell_vals)
        np.add.at(count, elements[:, col], 1.0)
    return (accum / np.maximum(count, 1.0)).astype(np.float32)


def _cell_to_nodal_smooth(
    num_nodes: int,
    elements: np.ndarray,
    cell_vals: np.ndarray,
    element_areas: np.ndarray,
    smooth_iters: int = 0,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Area-weighted cell -> nodal projection, with optional Laplacian smoothing.

    The plate is a SINGLE phase, so there is no material interface to preserve a
    discontinuity across and the projection is a plain area-weighted nodal
    average. Weighting by area rather than element count keeps a refined patch
    of small elements from dominating a node it barely touches.

    ``smooth_iters`` is 0 by default (see ``_STRESS_SMOOTH_ITERS``): smoothing
    would only diffuse — and so under-report — the genuine stress concentration
    at the hole rim, which is exactly the feature being measured.
    """
    accum = np.zeros(num_nodes, dtype=np.float64)
    wsum  = np.zeros(num_nodes, dtype=np.float64)
    weighted = np.asarray(cell_vals, dtype=np.float64) * element_areas

    for col in range(elements.shape[1]):
        np.add.at(accum, elements[:, col], weighted)
        np.add.at(wsum,  elements[:, col], element_areas)

    nodal = accum / np.maximum(wsum, 1e-30)

    if smooth_iters == 0:
        return nodal.astype(np.float32)

    from scipy.sparse import csr_matrix
    rows, cols_list = [], []
    for tri in elements:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            rows.append(a); cols_list.append(b)
            rows.append(b); cols_list.append(a)

    rows      = np.array(rows,      dtype=np.int32)
    cols_list = np.array(cols_list, dtype=np.int32)
    data      = np.ones(len(rows),  dtype=np.float64)
    A_raw     = csr_matrix((data, (rows, cols_list)), shape=(num_nodes, num_nodes))
    degree    = np.asarray(A_raw.sum(axis=1)).ravel()
    inv_d     = np.where(degree > 0, 1.0 / np.maximum(degree, 1e-30), 0.0)
    A_norm    = csr_matrix((data * inv_d[rows], (rows, cols_list)),
                           shape=(num_nodes, num_nodes))
    for _ in range(smooth_iters):
        nodal = nodal + alpha * (A_norm.dot(nodal) - nodal)

    return nodal.astype(np.float32)


# Stress cell->nodal projection uses NO Laplacian smoothing. Smoothing would only
# diffuse (under-report) the genuine stress concentration at the hole rim of this
# single-phase plate, so the projection is a plain area-weighted nodal average.
_STRESS_SMOOTH_ITERS = 0


# ===========================================================================
# GNN model helpers
# ===========================================================================

# Populated by make_model from the loaded checkpoint; reported in summary.txt.
_CKPT_INFO: dict = {}


def make_model(device: torch.device, checkpoint_path: Path,
               material_model: str) -> DisplacementGNN:
    """Rebuild the trained GNN from a checkpoint.

    The architecture and the fixed output displacement scale are read from the
    checkpoint rather than the live ``Config``, so a fresh process reconstructs
    exactly the model that was trained. The scale is not a parameter and is
    absent from ``model_state_dict``; restoring it wrongly would silently
    rescale the whole predicted field.

    Args:
        device: Device to place the model on.
        checkpoint_path: Path to the ``.pt`` checkpoint.
        material_model: ``'LE'`` or ``'NH'``, recorded for reporting.

    Returns:
        DisplacementGNN: The model in float64, moved to ``device`` and set to
        eval mode.

    Raises:
        KeyError: If the checkpoint has no ``model_state_dict``.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")

    # The output displacement scale is a fixed multiplier, not a parameter, so it
    # is absent from model_state_dict and MUST be restored from the checkpoint —
    # rebuilding with the wrong scale silently rescales the whole field.
    out_scale = checkpoint.get("output_displacement_scale")
    if out_scale is None:
        out_scale = effective_output_displacement_scale(TrainConfig)

    # Architecture comes from the checkpoint when recorded, so a comparison run
    # in a fresh process rebuilds the trained shape instead of a live default.
    input_dim  = int(checkpoint.get("input_dim", 4))
    hidden_dim = int(checkpoint.get("hidden_dim", TrainConfig.HIDDEN_DIM))
    num_layers = int(checkpoint.get("num_layers", TrainConfig.NUM_LAYERS))
    # Keep the module-level config in sync: summary.txt reports the ML params
    # from TrainConfig, so a stale value there would mislabel the run.
    TrainConfig.HIDDEN_DIM = hidden_dim
    TrainConfig.NUM_LAYERS = num_layers

    model = DisplacementGNN(
        input_dim=input_dim,     # (X_norm, Y_norm, Material_ID, Interface)
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_scale=float(out_scale),
    ).double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _CKPT_INFO["selection"] = checkpoint.get("selection", "last (pre-best-tracking run)")
    _CKPT_INFO["epoch"] = checkpoint.get("epoch")
    _CKPT_INFO["loss"] = checkpoint.get("loss")
    print(f"Checkpoint sel.   : {_CKPT_INFO['selection']}  (epoch {_CKPT_INFO['epoch']})")
    return model


def _load_inclusions(case_dir: Path) -> list:
    """Read the hole geometry from config.json in case_dir.

    Returns a list of dicts (a single dict in the JSON is wrapped), or an empty
    list when the file is absent — in which case the near-field split degrades
    to "every node is near-field" rather than failing.
    """
    cfg_path = case_dir / "config.json"
    if not cfg_path.exists():
        return []
    import json
    with open(cfg_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return [raw]
    return list(raw)


def _pack_stress_fields(sigma: np.ndarray, vm: np.ndarray) -> dict:
    """Split a (E,2,2) Cauchy stress array into the named component fields."""
    return {
        "sigma_xx":  sigma[:, 0, 0].astype(np.float32),
        "sigma_yy":  sigma[:, 1, 1].astype(np.float32),
        "sigma_xy":  sigma[:, 0, 1].astype(np.float32),
        "von_mises": np.asarray(vm, dtype=np.float32),
    }


def compute_gnn_predictions(
    model: DisplacementGNN,
    device: torch.device,
    case_dir: Path,
    material_model: str,
) -> dict:
    """
    Run GNN inference. Displacement in mm, stress in the config unit.
    """
    nodes, elements, features, loading_facets, topology, element_material_ids = (
        load_mesh_data(str(case_dir))
    )
    mesh_info = load_mesh_summary(str(case_dir))
    L = mesh_info.get("plate_size") or float(max(np.ptp(nodes[:, 0]),
                                                 np.ptp(nodes[:, 1])))

    graph_data = build_graph_data(nodes, elements, features, topology).to(device)
    bc_handler = BoundaryConditions(nodes, features)
    ref_coords = torch.tensor(nodes,    dtype=torch.float64, device=device)
    elems_t    = torch.tensor(elements, dtype=torch.long,    device=device)

    energy_calc = CompositeEnergyCalculator(
        TrainConfig.YOUNGS_MODULUS_MATRIX,
        TrainConfig.POISSONS_RATIO_MATRIX,
        element_material_ids,
        material_model=material_model,
        output_unit=STRESS_UNIT,
    )

    # ── Time the GNN solve ONLY up to the displacement field ──────────────────
    # This is the "solve" cost for the PI-GNN: the forward pass over the mesh
    # graph plus the boundary-condition application that yields nodal
    # displacements. The clock STOPS at `disp` — the algebraic stress recovery
    # below is post-processing on an already-solved field and is NOT timed, so
    # the comparison against the FEM (also timed only to its converged
    # displacement) is like-for-like. The timer also EXCLUDES mesh loading and
    # all downstream output prep. CUDA is synchronised on both sides of the
    # timed region so GPU kernels are fully accounted for.
    import time as _time
    if device.type == "cuda":
        torch.cuda.synchronize()
    _gnn_t0 = _time.perf_counter()

    with torch.no_grad():
        disp_raw = model(graph_data.x, graph_data.edge_index,
                         getattr(graph_data, 'edge_attr', None))
        disp     = bc_handler.apply(disp_raw)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gnn_solve_time_s = _time.perf_counter() - _gnn_t0

        # ── Stress recovery (NOT timed) ───────────────────────────────────────
        cur_coords = ref_coords + disp
        F_def  = energy_calc.compute_deformation_gradient(ref_coords, cur_coords, elems_t)
        strain = energy_calc.compute_strain_tensor(F_def)

        emu, elam = energy_calc._get_cached_lame(device)

        sigma_t = energy_calc.stress_tensor(F_def, strain, emu, elam)
        _, _, _, vm_t = energy_calc.stress_components(F_def, strain, emu, elam)
        J_t = F_def[:, 0, 0] * F_def[:, 1, 1] - F_def[:, 0, 1] * F_def[:, 1, 0]

    disp_mm = disp.cpu().numpy().astype(np.float32)
    sigma   = sigma_t.cpu().numpy()
    vm      = vm_t.cpu().numpy().astype(np.float32)
    Jdet    = J_t.cpu().numpy().astype(np.float32)

    import io as _io, contextlib as _cl
    with _cl.redirect_stdout(_io.StringIO()):
        elem_areas = MeshGeometry.compute_element_areas(nodes, elements)
        nodal_load, F_x = MeshGeometry.compute_nodal_load(
            nodes, loading_facets, TRACTION_MAGNITUDE, verbose=False)
        edge_lengths = MeshGeometry.compute_facet_lengths(nodes, loading_facets)

    inclusions = _load_inclusions(case_dir)

    out = {
        "nodes":                nodes,
        "elements":             elements,
        "element_material_ids": element_material_ids,
        "element_areas":        elem_areas,
        "inclusions":           inclusions,
        "mesh_info":            mesh_info,
        "plate_size":           L,
        "applied_force":        F_x,
        "loaded_edge_length":   float(edge_lengths.sum()),
        "traction":             TRACTION_MAGNITUDE,
        # The FEM load is assembled on exactly this facet set, so it travels
        # with the prediction rather than being re-derived later.
        "loaded_node_ids":      bc_handler.loaded_nodes,
        "gnn_solve_time_s":     gnn_solve_time_s,
        "gnn_device":           str(device),
        "disp":                 disp_mm,
        "J":                    Jdet,
    }
    out.update(_pack_stress_fields(sigma, vm))
    out["Kt"] = stress_concentration_factor(out["sigma_xx"], TRACTION_MAGNITUDE)

    for key in NODAL_FIELDS:
        out[f"{key}_nodal"] = _cell_to_nodal_smooth(
            len(nodes), elements, out[key], elem_areas,
            smooth_iters=_STRESS_SMOOTH_ITERS)
    return out


# ===========================================================================
# FEniCSx FEM (2-D plane stress, LE or NH)
# ===========================================================================

def run_fenicsx_fem(
    case_dir: Path,
    work_dir: Path,
    gnn: dict,
    material_model: str,
) -> dict:
    """
    FEniCSx FEM reference for the hole plate, remapped onto the GNN ordering.

    Mesh coordinates are in mm; divided by 1000 for dolfinx (SI m).
    Displacement output (m) multiplied by 1000 to return mm.
    Stress: Pa -> the config unit.
    All returned arrays aligned to the GNN node/cell ordering.
    """
    from scipy.spatial import cKDTree

    work_dir.mkdir(parents=True, exist_ok=True)
    gnn_nodes    = gnn["nodes"]
    gnn_elements = gnn["elements"]

    input_mesh_path = case_dir / "composite_mesh.vtu"
    if not input_mesh_path.exists():
        raise FileNotFoundError(f"No composite_mesh.vtu in '{case_dir}'")

    meshio_mesh = meshio.read(str(input_mesh_path))
    tri_cells = [cb.data for cb in meshio_mesh.cells if cb.type == "triangle"]
    if not tri_cells:
        raise ValueError(f"No triangle cells in '{input_mesh_path}'")

    triangles = np.vstack(tri_cells).astype(np.int32)
    points_m  = meshio_mesh.points[:, :2].astype(np.float64) * MM_TO_M

    xdmf_path = work_dir / "fem_mesh.xdmf"
    meshio.write(str(xdmf_path),
                 meshio.Mesh(points=points_m, cells=[("triangle", triangles)]))

    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as xf:
        domain = xf.read_mesh(name="Grid")

    coords_all = domain.geometry.x          # (N_fem, >=2) [m]
    N_fem      = coords_all.shape[0]
    cells_conn = domain.geometry.dofmap.reshape(-1, 3)

    # FEM centroids (mm) for matching against GNN centroids (mm)
    fem_centroids_mm = coords_all[cells_conn][:, :, :2].mean(axis=1) / MM_TO_M
    gnn_centroids_mm = gnn_nodes[gnn_elements].mean(axis=1)

    gnn_cell_tree = cKDTree(gnn_centroids_mm)
    dists_a, fem_to_gnn_cell = gnn_cell_tree.query(fem_centroids_mm)
    if dists_a.max() > 1e-3:
        print(f"  WARNING: max centroid mismatch FEM->GNN = {dists_a.max():.3e} mm")

    # Single homogeneous matrix phase (the voids were removed at mesh time), so
    # E and nu are uniform over every element and no per-element material
    # lookup is needed — unlike the composite pipeline this code descends from.
    V_dg = fem.functionspace(domain, ("DG", 0))
    E_func  = fem.Function(V_dg)
    nu_func = fem.Function(V_dg)
    E_func.x.array[:]  = FEM_E_MATRIX
    nu_func.x.array[:] = FEM_NU_MATRIX

    fdim  = domain.topology.dim - 1
    x_min = float(coords_all[:, 0].min())
    x_max = float(coords_all[:, 0].max())
    y_max = float(coords_all[:, 1].max())
    x_tol = max(1e-14, 1e-8 * abs(x_max - x_min))
    y_tol = max(1e-14, 1e-8 * abs(coords_all[:, 1].max() - coords_all[:, 1].min()))

    left_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], x_min, atol=x_tol))
    right_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], x_max, atol=x_tol))
    # The pin is a VERTEX: a facet search finds nothing and would leave uy with
    # an unconstrained rigid-body mode.
    domain.topology.create_connectivity(0, domain.topology.dim)
    top_left_verts = dmesh.locate_entities_boundary(
        domain, 0,
        lambda x: np.isclose(x[0], x_min, atol=x_tol)
        & np.isclose(x[1], y_max, atol=y_tol))

    # ── Time the FEM solve ONLY ───────────────────────────────────────────────
    # The clock covers exactly the kernel call: assembly plus the linear (LE) or
    # Newton (NH) solve, stopping at the converged displacement. Mesh reading,
    # stress interpolation and remapping are excluded, matching what is timed on
    # the GNN side.
    import time as _time
    _fem_t0 = _time.perf_counter()
    uh, _sigma_expr = solve_composite(
        domain, E_func, nu_func, material_model,
        left_facets=left_facets, corner_verts=top_left_verts,
        right_facets=right_facets,
        traction=FEM_SIGMA, fdim=fdim,
    )
    fem_solve_time_s = _time.perf_counter() - _fem_t0

    # ── Assembled load check ──
    ds = build_traction_measure(domain, fdim, right_facets)
    fem_edge_len_mm = loaded_edge_length(domain, ds) / MM_TO_M
    # Pa*m -> config-unit*mm: the traction converts by PA_TO_UNIT and the length
    # by 1/MM_TO_M, so the assembled force lands in the same units the GNN's
    # nodal load reports.
    fem_force = applied_force(domain, FEM_SIGMA, ds) * PA_TO_UNIT / MM_TO_M

    # ── Extract displacements ────────────────────────────────
    _, _, geom_vtk = plot.vtk_mesh(uh.function_space)
    u_raw       = uh.x.array.reshape((geom_vtk.shape[0], 2))
    disp_fem_mm = (u_raw[:N_fem] / MM_TO_M).astype(np.float32)   # m -> mm

    # ── Stress interpolation ─────────────────────────────────
    def _interp_raw(expr_ufl):
        """Interpolate a UFL scalar into DG0 and return it unconverted."""
        func = fem.Function(V_dg)
        func.interpolate(fem.Expression(expr_ufl, V_dg.element.interpolation_points()))
        return func.x.array.copy()

    def _interp_unit(expr_ufl):
        """Interpolate a UFL stress scalar and convert Pa -> the config unit."""
        return (_interp_raw(expr_ufl) * PA_TO_UNIT).astype(np.float32)

    sigma_expr_val = _sigma_expr(uh)
    s_xx = _interp_unit(sigma_expr_val[0, 0])
    s_yy = _interp_unit(sigma_expr_val[1, 1])
    s_xy = _interp_unit(sigma_expr_val[0, 1])

    # Von Mises (plane-stress closed form: sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)).
    # Equivalent to the full 3-D deviator with szz = 0; a 2-D deviator
    # (Identity(2)) would drop the out-of-plane term szz_dev = -tr(sigma)/3 and
    # under-report by up to sqrt(5/6) ~= 0.913 for a uniaxial state. Must match
    # the GNN definition (CompositeEnergyCalculator.von_mises).
    vm = _interp_unit(ufl.sqrt(
        sigma_expr_val[0, 0]**2 - sigma_expr_val[0, 0] * sigma_expr_val[1, 1]
        + sigma_expr_val[1, 1]**2 + 3.0 * sigma_expr_val[0, 1]**2))

    F_c  = ufl.Identity(2) + ufl.grad(uh)
    Jdet = _interp_raw(ufl.det(F_c)).astype(np.float32)

    # ── Node-averaged stresses ───────────────────────────────
    v0_f = coords_all[cells_conn[:, 0], :2]
    v1_f = coords_all[cells_conn[:, 1], :2]
    v2_f = coords_all[cells_conn[:, 2], :2]
    fem_elem_areas = np.maximum(
        0.5 * np.abs((v1_f[:, 0] - v0_f[:, 0]) * (v2_f[:, 1] - v0_f[:, 1])
                     - (v2_f[:, 0] - v0_f[:, 0]) * (v1_f[:, 1] - v0_f[:, 1])),
        1e-30,
    ).astype(np.float64)

    cell_fields = {"sigma_xx": s_xx, "sigma_yy": s_yy,
                   "sigma_xy": s_xy, "von_mises": vm}
    nodal_fem = {
        k: _cell_to_nodal_smooth(N_fem, cells_conn, v, fem_elem_areas,
                                 smooth_iters=_STRESS_SMOOTH_ITERS)
        for k, v in cell_fields.items()
    }

    # ── Remap to GNN ordering ─────────────────────────────────
    # The nodal STRESS arrays are assembled from domain.geometry.dofmap, so they
    # live in geometry-node order — match them against the geometry coordinates.
    fem_coords_mm = coords_all[:N_fem, :2] / MM_TO_M
    node_dists, fem_to_gnn_node = cKDTree(fem_coords_mm).query(gnn_nodes)
    if node_dists.max() > 1e-4:
        print(f"  WARNING: max node mismatch FEM<->GNN = {node_dists.max():.3e} mm")

    # The DISPLACEMENT (uh.x.array) lives in function-space DOF order, which
    # dolfinx reorders independently of the geometry-node order (and for NH `uh`
    # is a collapsed sub-space with yet another ordering). It must be matched
    # against its own DOF coordinates — geom_vtk from
    # plot.vtk_mesh(uh.function_space) is exactly the coordinate array that
    # aligns with disp_fem_mm. Pairing the displacement with geometry
    # coordinates (fem_to_gnn_node) instead scrambles the field (ux no longer
    # ~0 on the left roller edge, spurious strain energy).
    dof_coords_mm = geom_vtk[:N_fem, :2] / MM_TO_M
    dof_dists, fem_to_gnn_dof = cKDTree(dof_coords_mm).query(gnn_nodes)
    if dof_dists.max() > 1e-4:
        print(f"  WARNING: max DOF mismatch FEM<->GNN = {dof_dists.max():.3e} mm")

    fem_cell_tree = cKDTree(fem_centroids_mm)
    cell_dists_b, gnn_to_fem_cell = fem_cell_tree.query(gnn_centroids_mm)
    if cell_dists_b.max() > 1e-3:
        print(f"  WARNING: max centroid mismatch GNN->FEM = {cell_dists_b.max():.3e} mm")

    out = {
        "nodes":              gnn_nodes,
        "elements":           gnn_elements,
        "disp":               disp_fem_mm[fem_to_gnn_dof].astype(np.float32),
        "J":                  Jdet[gnn_to_fem_cell].astype(np.float32),
        "applied_force":      fem_force,
        "loaded_edge_length": fem_edge_len_mm,
        "fem_solve_time_s":   fem_solve_time_s,
        # PETSc/MUMPS is CPU-only; recorded so a GNN-on-GPU comparison is
        # flagged as cross-device rather than silently reported as a speed-up.
        "fem_device":         "cpu (PETSc/MUMPS)",
    }
    for key in NODAL_FIELDS:
        out[key] = cell_fields[key][gnn_to_fem_cell].astype(np.float32)
        out[f"{key}_nodal"] = nodal_fem[key][fem_to_gnn_node].astype(np.float32)
    out["Kt"] = stress_concentration_factor(out["sigma_xx"], TRACTION_MAGNITUDE)
    return out


# ===========================================================================
# Combined VTU writer
# ===========================================================================

def write_combined_vtu(gnn: dict, fem_res: dict, out_vtu: Path) -> None:
    """Write one VTU holding the GNN field, the FEM field and their difference.

    Both a *smoothed nodal* copy (for display) and the *raw element* value (what
    the metrics are computed from) are stored for every stress component.

    Args:
        gnn: GNN result dict, supplying the mesh, displacement and stresses.
        fem_res: FEM result dict on the same node ordering.
        out_vtu: Destination ``.vtu`` path.
    """
    nodes    = gnn["nodes"]
    elements = gnn["elements"]
    N        = len(nodes)
    u        = STRESS_UNIT

    out_vtu.parent.mkdir(parents=True, exist_ok=True)
    with open(out_vtu, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                'byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{N}" NumberOfCells="{len(elements)}">\n')

        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x, y in nodes:
            f.write(f"          {x:.10g} {y:.10g} 0.0\n")
        f.write("        </DataArray>\n      </Points>\n")

        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for n0, n1, n2 in elements:
            f.write(f"          {int(n0)} {int(n1)} {int(n2)}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        for i in range(1, len(elements) + 1):
            f.write(f"          {3 * i}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in elements:
            f.write("          5\n")       # VTK_TRIANGLE
        f.write("        </DataArray>\n      </Cells>\n")

        # VTK picks the ACTIVE point vector/scalar from these attributes. Without
        # Vectors= the reader registers the displacement arrays but leaves
        # active_vectors = None, and ParaView's "Warp By Vector" then has nothing
        # to default to — the deformed shape cannot be displayed in one click.
        # Naming GNN_displacement_mm here makes warping work immediately; switch
        # to FEM_displacement_mm in the ParaView array dropdown to warp by the
        # reference solution instead.
        f.write(f'      <PointData Scalars="GNN_von_mises_{u}" '
                f'Vectors="GNN_displacement_mm">\n')

        def _pt_vec3(name, vals_2d):
            f.write(f'        <DataArray type="Float64" Name="{name}" '
                    f'NumberOfComponents="3" format="ascii">\n')
            for ux, uy in vals_2d[:N]:
                f.write(f"          {float(ux):.10g} {float(uy):.10g} 0.0\n")
            f.write("        </DataArray>\n")

        def _pt_scalar(name, vals):
            f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            for v in vals[:N]:
                f.write(f"          {float(v):.10g}\n")
            f.write("        </DataArray>\n")

        gnn_mag = np.linalg.norm(gnn["disp"][:N], axis=1)
        fem_mag = np.linalg.norm(fem_res["disp"][:N], axis=1)

        _pt_vec3("GNN_displacement_mm", gnn["disp"])
        _pt_scalar("GNN_disp_mag_mm",   gnn_mag)
        _pt_vec3("FEM_displacement_mm", fem_res["disp"])
        _pt_scalar("FEM_disp_mag_mm",   fem_mag)
        for key in NODAL_FIELDS:
            _pt_scalar(f"GNN_{key}_{u}", gnn[f"{key}_nodal"])
            _pt_scalar(f"FEM_{key}_{u}", fem_res[f"{key}_nodal"])
        _pt_scalar("AbsErr_disp_mag_mm", np.abs(gnn_mag - fem_mag))
        for key in NODAL_FIELDS:
            _pt_scalar(f"AbsErr_{key}_{u}",
                       np.abs(gnn[f"{key}_nodal"][:N] - fem_res[f"{key}_nodal"][:N]))
        _pt_scalar("Material_ID",
                   _cell_to_nodal(N, elements,
                                  gnn["element_material_ids"].astype(np.float32)))
        f.write("      </PointData>\n")

        f.write(f'      <CellData Scalars="GNN_von_mises_cell_{u}">\n')

        def _cell_scalar(name, vals):
            f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            for v in vals:
                f.write(f"          {float(v):.10g}\n")
            f.write("        </DataArray>\n")

        for key in CELL_FIELDS:
            _cell_scalar(f"GNN_{key}_cell_{u}", gnn[key])
            _cell_scalar(f"FEM_{key}_cell_{u}", fem_res[key])
            _cell_scalar(f"AbsErr_{key}_cell_{u}",
                         np.abs(gnn[key] - fem_res[key]))
        _cell_scalar("GNN_J_area_ratio", gnn["J"])
        _cell_scalar("FEM_J_area_ratio", fem_res["J"])
        _cell_scalar("Element_Material_ID",
                     gnn["element_material_ids"].astype(np.float32))
        f.write("      </CellData>\n")
        f.write("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")


# ===========================================================================
# CSV error export
# ===========================================================================

def save_error_csv(gnn: dict, fem_res: dict, out_csv: Path) -> None:
    """Write the per-node GNN/FEM/error table."""
    import pandas as pd
    N      = len(gnn["nodes"])
    coords = gnn["nodes"][:N]
    u      = STRESS_UNIT

    mag_gnn = np.linalg.norm(gnn["disp"][:N],     axis=1)
    mag_fem = np.linalg.norm(fem_res["disp"][:N], axis=1)
    abs_disp    = np.linalg.norm(gnn["disp"][:N] - fem_res["disp"][:N], axis=1)
    fem_mag_max = mag_fem.max()
    rel_disp    = abs_disp / fem_mag_max if fem_mag_max > 1e-30 else np.zeros(N)

    data = {
        "node_id":    np.arange(N),
        "x_mm":       coords[:, 0],
        "y_mm":       coords[:, 1],
        "material_id": _cell_to_nodal(N, gnn["elements"],
                                      gnn["element_material_ids"].astype(np.float32)),
        "ux_gnn_mm":  gnn["disp"][:N, 0],
        "uy_gnn_mm":  gnn["disp"][:N, 1],
        "mag_gnn_mm": mag_gnn,
        "ux_fem_mm":  fem_res["disp"][:N, 0],
        "uy_fem_mm":  fem_res["disp"][:N, 1],
        "mag_fem_mm": mag_fem,
        "abs_disp_err_mm": abs_disp,
        "rel_disp_err":    rel_disp,
    }

    for name in NODAL_FIELDS:
        gnn_s = gnn[f"{name}_nodal"][:N]
        fem_s = fem_res[f"{name}_nodal"][:N]
        abs_s = np.abs(gnn_s - fem_s)
        max_s = np.abs(fem_s).max()
        rel_s = abs_s / max_s if max_s > 1e-30 else np.zeros(N)
        data[f"{name}_gnn_{u}"]     = gnn_s
        data[f"{name}_fem_{u}"]     = fem_s
        data[f"abs_{name}_err_{u}"] = abs_s
        data[f"rel_{name}_err"]     = rel_s

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out_csv, index=False, float_format="%.8g")
    print(f"  Error CSV saved: {out_csv}  ({N} rows)")


# ===========================================================================
# Potential energy comparison
# ===========================================================================

def compute_potential_energies(
    gnn: dict,
    fem_res: dict,
    case_dir: Path,
    device: torch.device,
    material_model: str,
) -> dict:
    """Evaluate the total potential energy of both solutions on the same mesh.

    Because training minimises Pi, comparing Pi(u_GNN) against Pi(u_FEM) is a
    direct, mesh-consistent measure of how close the network came to the
    variational minimum — independent of any pointwise error norm.

    Args:
        gnn: GNN result dict, supplying the displacement and the traction.
        fem_res: FEM result dict on the same node ordering.
        case_dir: Mesh case directory the arrays are reloaded from.
        device: Torch device the energy is evaluated on.
        material_model: ``'LE'`` or ``'NH'``.

    Returns:
        dict: ``fem_tpe``, ``gnn_tpe``, the ``E_int`` and ``W_ext`` split for
        each, and ``tpe_rel_diff``, the relative difference in Pi.
    """
    nodes, elements, features, loading_facets, topology, element_material_ids = (
        load_mesh_data(str(case_dir))
    )
    import io as _io, contextlib as _cl
    with _cl.redirect_stdout(_io.StringIO()):
        element_areas = MeshGeometry.compute_element_areas(nodes, elements)
        nodal_load, _ = MeshGeometry.compute_nodal_load(
            nodes, loading_facets, gnn["traction"], verbose=False)

    energy_calc = CompositeEnergyCalculator(
        TrainConfig.YOUNGS_MODULUS_MATRIX,
        TrainConfig.POISSONS_RATIO_MATRIX,
        element_material_ids,
        material_model=material_model,
        output_unit=STRESS_UNIT,
    )

    ref_coords_t = torch.tensor(nodes,         dtype=torch.float64, device=device)
    elements_t   = torch.tensor(elements,      dtype=torch.long,    device=device)
    areas_t      = torch.tensor(element_areas, dtype=torch.float64, device=device)
    load_t       = torch.tensor(nodal_load,    dtype=torch.float64, device=device)

    def _tpe(disp_mm: np.ndarray):
        """Evaluate Pi for one displacement field.

        Args:
            disp_mm: Nodal displacements in mm, shape (N, 2).

        Returns:
            tuple: ``(Pi, E_int, W_ext)`` as Python floats.
        """
        disp_t = torch.tensor(np.asarray(disp_mm, dtype=np.float64),
                              dtype=torch.float64, device=device)
        with torch.no_grad():
            Pi, E_int, W_ext, _, _ = energy_calc.compute_total_potential_energy(
                ref_coords_t, disp_t, elements_t, areas_t, load_t)
        return float(Pi), float(E_int), float(W_ext)

    N = len(nodes)
    fem_tpe, fem_E_int, fem_W_ext = _tpe(fem_res["disp"][:N])
    gnn_tpe, gnn_E_int, gnn_W_ext = _tpe(gnn["disp"][:N])
    tpe_rel_diff = (gnn_tpe - fem_tpe) / abs(fem_tpe) if abs(fem_tpe) > 1e-30 else float("nan")

    return {
        "fem_tpe":   fem_tpe,
        "gnn_tpe":   gnn_tpe,
        "fem_E_int": fem_E_int,
        "gnn_E_int": gnn_E_int,
        "fem_W_ext": fem_W_ext,
        "gnn_W_ext": gnn_W_ext,
        "tpe_rel_diff": tpe_rel_diff,
    }


# ===========================================================================
# R2 metrics (global + near-field / far-field split)
# ===========================================================================

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination of a prediction against a reference.

    Args:
        y_true: Reference (FEM) values, shape (M,).
        y_pred: Predicted (GNN) values, shape (M,).

    Returns:
        float: R^2, or NaN when the reference is essentially constant and the
        score is undefined.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")


def _nearfield_mask(nodes: np.ndarray, inclusions: list,
                    factor: float = 2.0) -> np.ndarray:
    """
    Boolean per node: within a band of width char_length*(factor-1) outside a
    hole boundary — the shell where the stress concentrates.

    For a circular hole the signed distance is (dist - r), so sd <= r*(factor-1)
    is exactly dist <= factor*r. The hole interior contains no nodes (the void
    was carved out), so the band is one-sided by construction.
    """
    from inclusion_shapes import make_inclusion
    mask = np.zeros(len(nodes), dtype=bool)
    for inc in inclusions:
        shape = make_inclusion(inc)
        band  = shape.char_length() * (factor - 1.0)
        sd    = shape.signed_distance(nodes[:, 0], nodes[:, 1])   # vectorised
        mask |= np.asarray(sd) <= band
    return mask


def compute_r2_metrics(gnn: dict, fem_res: dict,
                       case_dir: Optional[Path] = None) -> dict:
    """Score the GNN against the FEM, globally and split by distance to the rim.

    The near-field / far-field split separates the region where the hole makes
    the field hard to represent from the smooth bulk, so a good global score
    cannot hide a poor fit at the concentration.

    Args:
        gnn: GNN result dict.
        fem_res: FEM result dict on the same node ordering.
        case_dir: Case directory used to load the hole geometry when the GNN
            dict does not already carry it. Without it every node is treated as
            near-field.

    Returns:
        dict: ``R2_<field>``, ``R2_<field>_nf`` and ``R2_<field>_ff`` for each
        displacement component and stress field, plus the node counts
        ``_n_nearfield`` and ``_n_farfield``.
    """
    N          = len(gnn["nodes"])
    nodes      = gnn["nodes"][:N]
    inclusions = gnn.get("inclusions") or (
        _load_inclusions(case_dir) if case_dir is not None else [])

    nf_mask = _nearfield_mask(nodes, inclusions) if inclusions else np.ones(N, dtype=bool)
    ff_mask = ~nf_mask if inclusions else np.ones(N, dtype=bool)

    n_near = int(nf_mask.sum())
    n_far  = int(ff_mask.sum())

    out: dict = {"_n_nearfield": n_near, "_n_farfield": n_far}

    def _add(name, true_v, pred_v):
        """Score one field globally and on the near/far-field node subsets."""
        out[f"R2_{name}"] = _r2(true_v, pred_v)
        out[f"R2_{name}_nf"] = (_r2(true_v[nf_mask], pred_v[nf_mask])
                                if n_near > 1 else float("nan"))
        out[f"R2_{name}_ff"] = (_r2(true_v[ff_mask], pred_v[ff_mask])
                                if n_far > 1 else float("nan"))

    for i, comp in enumerate(("ux", "uy")):
        _add(comp, fem_res["disp"][:N, i], gnn["disp"][:N, i])
    for key in _R2_FIELDS:
        _add(key, fem_res[f"{key}_nodal"][:N], gnn[f"{key}_nodal"][:N])

    return out


# ===========================================================================
# L2 error metrics (per case)
# ===========================================================================

def compute_l2_metrics(gnn: dict, fem_res: dict) -> dict:
    """
    Per-case L2 errors of GNN vs FEM (FEM = ground truth), for every field.

    Absolute L2 error (unweighted, summed over nodes):
        err_abs = sqrt( Sum_i ||X_i - X^_i||^2 )     [mm for u, stress unit for sigma]

    Relative L2 error (element AREA-weighted), in %:
        err_rel = sqrt( Sum_e A_e (X_e - X^_e)^2 ) / sqrt( Sum_e A_e X_e^2 ) x 100

    The weight is the element area: the graded mesh has elements differing by an
    order of magnitude in size, so an unweighted norm would over-count the
    refined band at the hole rim purely by element count.
    """
    N        = len(gnn["nodes"])
    elements = gnn["elements"]
    areas    = gnn["element_areas"].astype(np.float64)

    def _rel_area(true_cell, pred_cell):
        """Area-weighted relative L2 error of a per-element field, in percent."""
        num = np.sqrt(np.sum(areas * (pred_cell - true_cell) ** 2))
        den = np.sqrt(np.sum(areas * true_cell ** 2))
        return float(num / den * 100.0) if den > 1e-30 else float("nan")

    u_fem = fem_res["disp"][:N].astype(np.float64)
    u_gnn = gnn["disp"][:N].astype(np.float64)
    abs_disp = float(np.sqrt(np.sum((u_gnn - u_fem) ** 2)))                 # mm
    uf_e = u_fem[elements].mean(axis=1)   # nodal -> element average
    ug_e = u_gnn[elements].mean(axis=1)
    num  = np.sqrt(np.sum(areas * np.sum((ug_e - uf_e) ** 2, axis=1)))
    den  = np.sqrt(np.sum(areas * np.sum(uf_e ** 2, axis=1)))
    rel_disp = float(num / den * 100.0) if den > 1e-30 else float("nan")

    out = {"abs_disp_mm": abs_disp, "rel_disp_pct": rel_disp}

    # uy alone: the transverse (Poisson) contraction is the small, hard-to-get
    # component and its error is invisible inside the displacement-magnitude
    # norm, which the axial extension dominates.
    abs_uy = float(np.sqrt(np.sum((u_gnn[:, 1] - u_fem[:, 1]) ** 2)))
    den_uy = float(np.sqrt(np.sum(areas * uf_e[:, 1] ** 2)))
    num_uy = float(np.sqrt(np.sum(areas * (ug_e[:, 1] - uf_e[:, 1]) ** 2)))
    out["abs_uy_mm"] = abs_uy
    out["rel_uy_pct"] = (num_uy / den_uy * 100.0) if den_uy > 1e-30 else float("nan")

    # Stress fields: absolute L2 on nodal values, relative L2 on cell values
    for name in NODAL_FIELDS:
        sf = fem_res[f"{name}_nodal"][:N].astype(np.float64)
        sg = gnn[f"{name}_nodal"][:N].astype(np.float64)
        out[f"abs_{name}"] = float(np.sqrt(np.sum((sg - sf) ** 2)))
        out[f"rel_{name}_pct"] = _rel_area(
            fem_res[name].astype(np.float64), gnn[name].astype(np.float64))
    return out


def compute_l2_metrics_nearfield(gnn: dict, fem_res: dict,
                                 case_dir: Optional[Path] = None,
                                 factor: float = 1.5) -> dict:
    """
    Near-field relative L2 errors — the same area-weighted quantity as
    compute_l2_metrics, restricted to elements in the hole-rim band.

    An element is in the band when ANY of its vertices is within
    char_length*(factor-1) of a hole boundary. That is where the stress
    concentration the graph model is supposed to resolve lives, and a global
    norm dominated by the smooth far field will not show a failure there.
    """
    N        = len(gnn["nodes"])
    elements = gnn["elements"]
    areas    = gnn["element_areas"].astype(np.float64)

    inclusions = gnn.get("inclusions") or (
        _load_inclusions(case_dir) if case_dir is not None else [])
    keys = ["rel_disp_nf_pct"] + [f"rel_{n}_nf_pct" for n in NODAL_FIELDS]
    if not inclusions:
        out = {k: float("nan") for k in keys}
        out["_n_nf_elements"] = 0
        return out

    nf_nodes  = _nearfield_mask(gnn["nodes"][:N], inclusions, factor=factor)
    elem_mask = nf_nodes[elements].any(axis=1)
    n_el = int(elem_mask.sum())
    if n_el < 1:
        out = {k: float("nan") for k in keys}
        out["_n_nf_elements"] = 0
        return out

    a_m = areas[elem_mask]

    def _rel_area_m(true_cell, pred_cell):
        """Area-weighted relative L2 error over the near-field elements only."""
        t = np.asarray(true_cell, dtype=np.float64)[elem_mask]
        p = np.asarray(pred_cell, dtype=np.float64)[elem_mask]
        num = np.sqrt(np.sum(a_m * (p - t) ** 2))
        den = np.sqrt(np.sum(a_m * t ** 2))
        return float(num / den * 100.0) if den > 1e-30 else float("nan")

    u_fem = fem_res["disp"][:N].astype(np.float64)
    u_gnn = gnn["disp"][:N].astype(np.float64)
    uf_e = u_fem[elements].mean(axis=1)[elem_mask]
    ug_e = u_gnn[elements].mean(axis=1)[elem_mask]
    num  = np.sqrt(np.sum(a_m * np.sum((ug_e - uf_e) ** 2, axis=1)))
    den  = np.sqrt(np.sum(a_m * np.sum(uf_e ** 2, axis=1)))

    out = {"rel_disp_nf_pct": float(num / den * 100.0) if den > 1e-30 else float("nan"),
           "_n_nf_elements": n_el,
           "_nf_factor": float(factor)}
    for name in NODAL_FIELDS:
        out[f"rel_{name}_nf_pct"] = _rel_area_m(fem_res[name], gnn[name])
    return out


def compute_signed_rel_errors(gnn: dict, fem_res: dict,
                              floor_frac: float = 1e-2) -> dict:
    """
    Signed pointwise relative error per field, over nodes (FEM = ground truth):

        s_i = (GNN_i - FEM_i) / |FEM_i| x 100   [%]
              ( +  GNN over-predicts ,  -  GNN under-predicts )

    Reported as median (robust — primary) and mean. Nodes where
    |FEM_i| < floor_frac * max|FEM| are excluded: there the reference is ~0
    (zero-crossings, the constrained edge) and the ratio is meaningless.
    """
    N   = len(gnn["nodes"])
    out = {}
    fields = [("ux", gnn["disp"][:N, 0], fem_res["disp"][:N, 0]),
              ("uy", gnn["disp"][:N, 1], fem_res["disp"][:N, 1])]
    fields += [(k, gnn[f"{k}_nodal"][:N], fem_res[f"{k}_nodal"][:N])
               for k in _R2_FIELDS]

    for name, pred, ref in fields:
        ref   = ref.astype(np.float64)
        pred  = pred.astype(np.float64)
        denom = np.abs(ref)
        peak  = float(denom.max())
        mask  = denom > floor_frac * peak if peak > 0 else np.zeros(N, dtype=bool)
        if int(mask.sum()) > 0:
            s = (pred[mask] - ref[mask]) / denom[mask] * 100.0
            out[f"{name}_median_pct"] = float(np.median(s))
            out[f"{name}_mean_pct"]   = float(np.mean(s))
            out[f"{name}_n"]          = int(mask.sum())
        else:
            out[f"{name}_median_pct"] = float("nan")
            out[f"{name}_mean_pct"]   = float("nan")
            out[f"{name}_n"]          = 0
    return out


def write_aggregate_summary(output_dir: Path, case_names: list,
                            l2_list: list, material_model: str) -> Optional[Path]:
    """Median & mean of each per-case L2 error over the M processed cases."""
    if not l2_list:
        return None
    import datetime
    arr = {k: np.array([d[k] for d in l2_list], dtype=np.float64) for k in l2_list[0]}
    fields = [("Displacement", "abs_disp_mm", "rel_disp_pct", "mm"),
              ("uy (transverse)", "abs_uy_mm", "rel_uy_pct",  "mm")]
    fields += [(n, f"abs_{n}", f"rel_{n}_pct", STRESS_UNIT) for n in NODAL_FIELDS]
    lines = [
        "=" * 78,
        "  PI-GNN Hole Plate — Aggregate L2 errors over M test cases",
        "=" * 78,
        f"  Generated      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Material model : {material_model}",
        f"  M (cases)      : {len(l2_list)}",
        f"  Cases          : {', '.join(case_names)}",
        "",
        "  Field                Absolute L2 [unit]          Relative L2 [%]",
        "                       median       mean           median     mean",
    ]
    for label, ak, rk, unit in fields:
        a, r = arr[ak], arr[rk]
        lines.append(
            f"  {label:<18s} {np.median(a):.4e} {np.mean(a):.4e} {unit:<4s} "
            f"{np.median(r):8.4f} {np.mean(r):8.4f}"
        )
    lines.append("=" * 78)
    out_path = output_dir / "aggregate_summary.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ===========================================================================
# Summary file writer
# ===========================================================================

def save_summary_txt(
    case_name: str,
    case_out: Path,
    checkpoint_path: Path,
    gnn: dict,
    fem_res: dict,
    energy: dict,
    r2: dict,
    l2: dict,
    signed_rel: dict,
    material_model: str,
    training_time_s: Optional[float] = None,
    config_json_path: Optional[Path] = None,
) -> Path:
    """Write the human-readable per-case comparison report.

    Args:
        case_name: Short case identifier used in the report header.
        case_out: Output directory for this case.
        checkpoint_path: Checkpoint the GNN prediction came from.
        gnn: GNN result dict.
        fem_res: FEM result dict.
        energy: Potential-energy comparison from ``compute_potential_energies``.
        r2: R^2 metrics from ``compute_r2_metrics``.
        l2: L2 error metrics from ``compute_l2_metrics``.
        signed_rel: Signed relative errors from ``compute_signed_rel_errors``.
        material_model: ``'LE'`` or ``'NH'``.
        training_time_s: Optimisation compute time in seconds, if known.
        config_json_path: Mesh ``config.json`` echoed into the report, if given.

    Returns:
        Path: The written ``summary.txt``.
    """
    import datetime, json as _json
    N        = len(gnn["nodes"])
    out_path = case_out / "summary.txt"

    n_near = r2.get("_n_nearfield", 0)
    n_far  = r2.get("_n_farfield",  0)

    def _fmt(v):
        """Format a metric to 6 decimals, rendering NaN as the literal 'nan'."""
        return f"{v:.6f}" if not (v != v) else "nan"  # NaN-safe

    def _l2row(label, abs_v, abs_unit, rel_v):
        """Format one absolute/relative L2 error row of the report table."""
        return f"  {label:<16s} {abs_v:.6e} {abs_unit:<5s}   {rel_v:9.4f} %"

    def _srow(name, sr):
        """Format one signed-relative-error row (median and mean, in percent)."""
        med = sr.get(f"{name}_median_pct", float("nan"))
        mn  = sr.get(f"{name}_mean_pct", float("nan"))
        return f"  {name:<16s} {med:+10.4f} %   {mn:+10.4f} %"

    def _r2row(label, key):
        """Format one R^2 row across the reported displacement and stress fields.

        Args:
            label: Row label, e.g. the region name.
            key: Metric suffix selecting the subset — ``''`` global, ``'_nf'``
                near-field, ``'_ff'`` far-field.
        """
        return (f"  {label:<10s} ux {_fmt(r2[f'R2_ux{key}'])}  uy {_fmt(r2[f'R2_uy{key}'])}  "
                f"sxx {_fmt(r2[f'R2_sigma_xx{key}'])}  syy {_fmt(r2[f'R2_sigma_yy{key}'])}  "
                f"sxy {_fmt(r2[f'R2_sigma_xy{key}'])}  vm {_fmt(r2[f'R2_von_mises{key}'])}")

    nodes = gnn["nodes"][:N]
    info  = gnn["mesh_info"]
    unit  = STRESS_UNIT
    E_mat = float(TrainConfig.YOUNGS_MODULUS_MATRIX)
    T     = gnn["traction"]
    x0, x1 = float(nodes[:, 0].min()), float(nodes[:, 0].max())
    y0, y1 = float(nodes[:, 1].min()), float(nodes[:, 1].max())

    if training_time_s is not None:
        h = int(training_time_s) // 3600
        m = (int(training_time_s) % 3600) // 60
        s = training_time_s % 60
        train_time_str = f"{h:02d}h {m:02d}m {s:05.2f}s  ({training_time_s:.1f} s)"
    else:
        train_time_str = "n/a"

    gnn_solve_time_s = gnn.get("gnn_solve_time_s")
    fem_solve_time_s = fem_res.get("fem_solve_time_s")

    def _fmt_time(t):
        """Format a solve time in both milliseconds and seconds, or 'n/a'."""
        if t is None:
            return "n/a"
        return f"{t * 1e3:.3f} ms  ({t:.6f} s)"

    _gnn_device = gnn.get("gnn_device", "unknown")
    _fem_device = fem_res.get("fem_device", "unknown")
    _gnn_solve_str = f"{_fmt_time(gnn_solve_time_s)}   [device: {_gnn_device}]"
    _fem_solve_str = f"{_fmt_time(fem_solve_time_s)}   [device: {_fem_device}]"
    if (gnn_solve_time_s is not None and fem_solve_time_s is not None
            and gnn_solve_time_s > 0):
        # Flag cross-device comparisons: the GNN may run on GPU while the FEM
        # (PETSc/MUMPS) is CPU-only, so the ratio is not same-hardware.
        _cross = not _gnn_device.startswith("cpu")
        _note = "  (cross-device: GNN GPU vs FEM CPU)" if _cross else ""
        _speedup_str = f"{fem_solve_time_s / gnn_solve_time_s:.2f}x{_note}"
    else:
        _speedup_str = "n/a"

    nh_note = []
    if material_model.upper() == "NH":
        nh_note = [
            "",
            "  NOTE: material model is NH (finite-strain Neo-Hookean). At this",
            "  load the nominal strain is ~5e-4, so NH and LE agree to within",
            "  O(strain^2); NH is exercised here as a code path, not because the",
            "  physics demands it.",
        ]

    lines = [
        "=" * 78,
        "  PI-GNN Hole Plate — Prediction Summary",
        "=" * 78,
        f"  Generated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Case       : {case_name}",
        f"  Checkpoint : {checkpoint_path}",
        f"  Checkpoint selection : {_CKPT_INFO.get('selection', 'unknown')}"
        + (f"   @ epoch {_CKPT_INFO['epoch']},  Pi = {_CKPT_INFO['loss']:.8e}"
           if _CKPT_INFO.get("loss") is not None else ""),
        f"  Material model : {material_model}",
        f"  Problem type   : 2-D plate with circular hole(s) (single matrix phase)",
    ] + nh_note + [

        "",
        "-- Material properties ------------------------------------------------",
        f"  E_matrix                      : {E_mat:.4g} {unit}   ({FEM_E_MATRIX:.4g} Pa)",
        f"  nu_matrix                     : {FEM_NU_MATRIX}",
        f"  Constitutive model            : "
        + ("compressible Neo-Hookean, finite strain, rigorous plane stress (F33)"
           if material_model.upper() == "NH"
           else "linear elastic, small strain, plane stress (reduced lambda_ps)"),

        "",
        "-- Geometry & boundary conditions -------------------------------------",
        f"  Domain (mm)             : x=[{x0:.4f}, {x1:.4f}], y=[{y0:.4f}, {y1:.4f}]"
        f"   (W x H = {x1-x0:.4f} x {y1-y0:.4f})",
        f"  Plate size              : {gnn['plate_size']:.4f} mm",
        f"  Formulation             : plane stress (2-D)",
        f"  x = x_min               : ROLLER, ux = 0 (uy free)",
        f"  top-left corner         : PIN, uy = 0  (removes the last rigid-body mode)",
        f"  x = x_max               : uniform dead traction t = (T, 0) in +x",
        f"  hole rims / y edges     : free (traction-free)",
        f"  APPLIED TRACTION T      : {T:.6g} {unit}   ({T * UNIT_TO_PA:.6g} Pa)",
        f"  Nominal strain T/E      : {T / E_mat:.6e}",
        f"  Loaded edge length      : {gnn['loaded_edge_length']:.6f} mm "
        f"(FEM {fem_res['loaded_edge_length']:.6f} mm)",
        f"  Nodes / Triangles       : {N} / {len(gnn['elements'])}",
        f"  Holes                   : {info.get('holes')}",

        "",
        "-- Plate response (the headline outputs) ------------------------------",
        f"  Applied force Fx      : GNN {gnn['applied_force']:.6f}   "
        f"FEM {fem_res['applied_force']:.6f}   [{unit}*mm per unit thickness]",
        f"    (both are assembled over the SAME discretised loaded edge, so they",
        f"     must agree; a mismatch means the two solvers read different meshes)",
        f"  Kt = max(sxx)/T       : GNN {gnn['Kt']:8.4f}   FEM {fem_res['Kt']:8.4f}   "
        f"(err {gnn['Kt'] - fem_res['Kt']:+.4f})",
        f"    (Kirsch gives Kt = 3.0 exactly for a small circular hole in a wide",
        f"     plate under remote uniaxial tension — an independent check that the",
        f"     field has the right SHAPE at the rim, not merely the right size)",
        f"  Max |u|               : GNN {float(np.linalg.norm(gnn['disp'], axis=1).max()):.6e} mm"
        f"   FEM {float(np.linalg.norm(fem_res['disp'], axis=1).max()):.6e} mm",
        f"  Max |ux| / |uy|       : GNN {float(np.abs(gnn['disp'][:, 0]).max()):.6e} / "
        f"{float(np.abs(gnn['disp'][:, 1]).max()):.6e} mm",
        f"                          FEM {float(np.abs(fem_res['disp'][:, 0]).max()):.6e} / "
        f"{float(np.abs(fem_res['disp'][:, 1]).max()):.6e} mm",
        f"  Peak von Mises        : GNN {float(gnn['von_mises'].max()):.6f}   "
        f"FEM {float(fem_res['von_mises'].max()):.6f}   [{unit}]",
        f"  Area ratio J = det F  : GNN [{gnn['J'].min():.6f}, {gnn['J'].max():.6f}]   "
        f"FEM [{fem_res['J'].min():.6f}, {fem_res['J'].max():.6f}]   (1.0 = area preserving)",

        "",
        "-- ML parameters ------------------------------------------------------",
        f"  GNN hidden dim / layers : {TrainConfig.HIDDEN_DIM} / {TrainConfig.NUM_LAYERS}",
        f"  Training epochs         : {TrainConfig.NUM_EPOCHS}",
        f"  Learning rate           : {TrainConfig.LEARNING_RATE:g}",

        "",
        "-- Timing (same mesh; excludes mesh gen & output/file writing) --------",
        f"  GNN training time      : {train_time_str}",
        f"  GNN forward-pass solve : {_gnn_solve_str}",
        f"  FEM solve              : {_fem_solve_str}",
        f"  Speed-up (FEM / GNN)   : {_speedup_str}",

        "",
        "-- L2 errors — GNN vs FEM (FEM = ground truth) ------------------------",
        "  (single case -> median = mean = value; see aggregate_summary.txt for M>1)",
        "  Field            Absolute L2           Relative L2 (area-weighted)",
        _l2row("Displacement", l2["abs_disp_mm"], "mm", l2["rel_disp_pct"]),
        _l2row("uy (transverse)", l2["abs_uy_mm"], "mm", l2["rel_uy_pct"]),
    ] + [
        _l2row(n, l2[f"abs_{n}"], unit, l2[f"rel_{n}_pct"]) for n in NODAL_FIELDS
    ] + [

        "",
        "-- L2 errors — hole-rim band (near-field) -----------------------------",
        f"  Band = matrix shell of 0.5x char_length outside each rim "
        f"({l2.get('_n_nf_elements', 0)} elements)",
        "  Field                                Relative L2 (area-weighted)",
        f"  Displacement                         {l2.get('rel_disp_nf_pct', float('nan')):>10.4f} %",
    ] + [
        f"  {n:<36s} {l2.get(f'rel_{n}_nf_pct', float('nan')):>10.4f} %"
        for n in NODAL_FIELDS
    ] + [

        "",
        "-- R2 scores (GNN vs FEM, nodal fields) -------------------------------",
        _r2row("[global]", ""),
        f"  [near-field: <= 2x hole size, {n_near} nodes]",
        _r2row("", "_nf"),
        f"  [far-field: remainder, {n_far} nodes]",
        _r2row("", "_ff"),

        "",
        f"-- Energy & loss metrics (units {unit}*mm^2 == mJ/mm thickness; "
        f"Pi = training loss) --",
        f"  FEM  strain energy E_int / external work W_ext : "
        f"{energy['fem_E_int']:.6e} / {energy['fem_W_ext']:.6e}",
        f"  GNN  strain energy E_int / external work W_ext : "
        f"{energy['gnn_E_int']:.6e} / {energy['gnn_W_ext']:.6e}",
        f"  FEM  total potential Pi (loss) : {energy['fem_tpe']:.6e}   [ground truth]",
        f"  GNN  total potential Pi (loss) : {energy['gnn_tpe']:.6e}",
        f"  Relative Pi difference         : {energy['tpe_rel_diff']:+.4e}  ((GNN-FEM)/|FEM|)",
        f"  2*E_int/W_ext  (FEM / GNN)     : "
        f"{2 * energy['fem_E_int'] / energy['fem_W_ext'] if energy['fem_W_ext'] else float('nan'):.4f}"
        f" / {2 * energy['gnn_E_int'] / energy['gnn_W_ext'] if energy['gnn_W_ext'] else float('nan'):.4f}",
        "    (-> 1.0 exactly at the LE minimum; O(1) but not 1 for NH. A value far",
        "     from 1, a POSITIVE Pi, or a NEGATIVE W_ext means non-convergence.)",
        "    NOTE: both rows use the GNN energy functional, so the FEM row is that",
        "    functional applied to the FEM field rather than the FEM's own energy.",
        "    Because both discretise the same plane-stress functional on the same",
        "    mesh, this is a like-for-like comparison and the minimum principle",
        "    applies: Pi_GNN >= Pi_FEM, approaching it from above.",

        "",
        "-- Signed relative error per field — (GNN-FEM)/|FEM| x 100, over nodes -",
        "  (+ over-prediction, - under-prediction; |FEM| < 1% of peak excluded)",
        "  Field            median          mean",
        _srow("ux", signed_rel),
        _srow("uy", signed_rel),
    ] + [_srow(n, signed_rel) for n in _R2_FIELDS] + [

        "",
        "-- Output files -------------------------------------------------------",
        f"  VTU : {case_out / 'combined_gnn_fem.vtu'}",
        f"  CSV : {case_out / 'error_data.csv'}",
        f"  TXT : {out_path}",
    ]

    if config_json_path is not None and config_json_path.exists():
        try:
            with open(config_json_path) as f:
                cfg_text = _json.dumps(_json.load(f), indent=2)
            lines += [
                "",
                "-- Case configuration (config.json) ----------------------------",
                cfg_text,
            ]
        except Exception:
            pass

    lines.append("=" * 78)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ===========================================================================
# Main orchestration
# ===========================================================================

def run_inference(
    model_path: Optional[Path],
    input_dir: Path,
    output_dir: Path,
    device_str: str,
    material_model_override: Optional[str] = None,
) -> None:
    """Run GNN inference and the FEM comparison over every discovered case.

    For each case this predicts the displacement field with the trained GNN,
    solves the same problem with FEniCSx as the reference, and writes the
    combined VTU, the per-node error CSV and the summary report. An aggregate
    L2 summary over all cases is written at the end.

    Args:
        model_path: Checkpoint to evaluate, or None to select automatically.
        input_dir: A mesh case directory, or a tree containing several.
        output_dir: Root directory the per-case outputs are written under.
        device_str: Torch device string for the GNN, e.g. ``'cpu'`` or
            ``'cuda'``. The FEM always runs on CPU.
        material_model_override: Force ``'LE'`` or ``'NH'`` instead of the model
            recorded in the checkpoint.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist or holds no complete
            mesh case.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Pin PyTorch to the same core budget already applied to OMP/MUMPS at import
    # time, so the GNN and FEM solves are timed on equal CPU resources.
    if _NUM_THREADS is not None and _NUM_THREADS > 0:
        torch.set_num_threads(_NUM_THREADS)
        try:
            torch.set_num_interop_threads(_NUM_THREADS)
        except RuntimeError:
            pass  # inter-op pool already initialised; intra-op limit still holds
        print(f"Thread limit      : {_NUM_THREADS} core(s)  (PyTorch + OMP/MUMPS)")

    resolved_model = resolve_model_path(model_path)

    if material_model_override:
        mat_model = material_model_override.upper()
    else:
        mat_model = _infer_material_model(resolved_model)

    print(f"Material model    : {mat_model}")

    case_dirs = discover_case_dirs(input_dir)
    if not case_dirs:
        raise RuntimeError(
            f"No valid mesh case found in '{input_dir}'. "
            f"Required: {', '.join(REQUIRED_FILES)}"
        )

    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    print(f"Device            : {device}")
    print(f"Checkpoint        : {resolved_model}")
    print(f"Matrix E          : {TrainConfig.YOUNGS_MODULUS_MATRIX:.6g} {STRESS_UNIT}")
    print(f"Traction T        : {TRACTION_MAGNITUDE:.6g} {STRESS_UNIT}")

    model = make_model(device, resolved_model, mat_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    import json as _json
    _tt_path = resolved_model.parent / "training_time.json"
    _training_time_s: Optional[float] = None
    if _tt_path.exists():
        try:
            with open(_tt_path) as _f:
                _training_time_s = float(_json.load(_f).get("total_training_time_s", 0))
        except Exception:
            pass

    print(f"Found {len(case_dirs)} case(s).\n")

    l2_all: list = []
    case_names: list = []

    for idx, case_dir in enumerate(case_dirs, start=1):
        case_name = case_dir.name
        case_out  = output_dir / case_name
        case_out.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print(f"[{idx}/{len(case_dirs)}]  Case: {case_name}")
        print(f"  Input  : {case_dir}")
        print(f"  Output : {case_out}")

        print("  Running GNN inference ...")
        gnn = compute_gnn_predictions(model, device, case_dir, mat_model)
        print(f"    GNN  |u| max = {float(np.linalg.norm(gnn['disp'], axis=1).max()):.4e} mm")
        print(f"    GNN  Kt      = {gnn['Kt']:.4f}  (max sxx / T)")
        print(f"    GNN  VM  max = {gnn['von_mises_nodal'].max():.4f} {STRESS_UNIT}")
        print(f"    GNN  solve time = {gnn['gnn_solve_time_s'] * 1e3:.3f} ms")

        fem_work = case_out / "_fem_work"
        print(f"  Running FEniCSx FEM ({mat_model}) ...")
        fem_res = run_fenicsx_fem(case_dir, fem_work, gnn, material_model=mat_model)
        print(f"    FEM  |u| max = {float(np.linalg.norm(fem_res['disp'], axis=1).max()):.4e} mm")
        print(f"    FEM  Kt      = {fem_res['Kt']:.4f}  (max sxx / T)")
        print(f"    FEM  VM  max = {fem_res['von_mises_nodal'].max():.4f} {STRESS_UNIT}")
        print(f"    FEM  solve time = {fem_res['fem_solve_time_s'] * 1e3:.3f} ms")

        vtu_file = case_out / "combined_gnn_fem.vtu"
        write_combined_vtu(gnn, fem_res, vtu_file)
        print(f"  Combined VTU   : {vtu_file}")

        error_csv = case_out / "error_data.csv"
        save_error_csv(gnn, fem_res, error_csv)

        N        = len(gnn["nodes"])
        abs_disp = np.linalg.norm(gnn["disp"][:N] - fem_res["disp"][:N], axis=1)
        fem_mag_max = np.linalg.norm(fem_res["disp"][:N], axis=1).max()
        if fem_mag_max > 1e-30:
            rel_err_median = float(np.median(abs_disp / fem_mag_max))
            rel_err_mean   = float(np.mean(abs_disp / fem_mag_max))
        else:
            rel_err_median = rel_err_mean = float("nan")
        print(f"  Disp rel-error : mean={rel_err_mean:.3e}  median={rel_err_median:.3e}")

        print("  Computing R2 metrics ...")
        r2_metrics = compute_r2_metrics(gnn, fem_res, case_dir=case_dir)
        for key, val in r2_metrics.items():
            if not key.startswith("_"):
                print(f"    {key:24s}: {val:.6f}")

        print("  Computing L2 errors ...")
        l2_metrics = compute_l2_metrics(gnn, fem_res)
        l2_metrics.update(
            compute_l2_metrics_nearfield(gnn, fem_res, case_dir=case_dir))
        l2_all.append(l2_metrics)
        case_names.append(case_name)
        print(f"    abs L2 |u|         : {l2_metrics['abs_disp_mm']:.4e} mm   "
              f"(rel {l2_metrics['rel_disp_pct']:.4f} %)")
        print(f"    rel L2 uy          : {l2_metrics['rel_uy_pct']:.4f} %")
        print(f"    rel L2 sigma_xx    : {l2_metrics['rel_sigma_xx_pct']:.4f} %")
        print(f"    rel L2 von Mises   : {l2_metrics['rel_von_mises_pct']:.4f} %")
        print(f"    rel L2 vm (rim)    : {l2_metrics['rel_von_mises_nf_pct']:.4f} %   "
              f"[{l2_metrics['_n_nf_elements']} hole-rim-band elements]")

        signed_rel = compute_signed_rel_errors(gnn, fem_res)
        print(f"    signed rel vm      : median {signed_rel['von_mises_median_pct']:+.4f} %   "
              f"mean {signed_rel['von_mises_mean_pct']:+.4f} %")

        print("  Computing potential energies ...")
        energy_metrics = compute_potential_energies(
            gnn, fem_res, case_dir, device, mat_model)
        print(f"    FEM Pi : {energy_metrics['fem_tpe']:.6e}  [ground truth]")
        print(f"    GNN Pi : {energy_metrics['gnn_tpe']:.6e}")
        print(f"    Rel diff: {energy_metrics['tpe_rel_diff']:+.4e}")

        cfg_json_path = case_dir / "config.json"
        summary_file = save_summary_txt(
            case_name, case_out, resolved_model,
            gnn, fem_res, energy_metrics, r2_metrics, l2_metrics, signed_rel,
            material_model=mat_model,
            training_time_s=_training_time_s,
            config_json_path=cfg_json_path if cfg_json_path.exists() else None,
        )
        print(f"  Summary TXT    : {summary_file}")

    agg_path = write_aggregate_summary(output_dir, case_names, l2_all, mat_model)
    if agg_path is not None:
        print(f"\nAggregate L2 summary ({len(l2_all)} case(s)): {agg_path}")

    print("\nAll cases processed successfully.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse the command-line options for the inference/comparison stage.

    Returns:
        argparse.Namespace: Checkpoint, input/output paths, device, material
            model override and the CPU thread budget.
    """
    parser = argparse.ArgumentParser(
        description=(
            "GNN inference + FEniCSx FEM comparison for the 2-D hole plate. "
            "Outputs combined VTU, per-node error CSV, and summary TXT."
        )
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Path to the trained checkpoint (.pt). Defaults to model_best.pt in "
             "TrainConfig.CHECKPOINT_DIR.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path(TrainConfig.DATA_DIR),
        help="Directory containing mesh case(s) with .npy files and composite_mesh.vtu.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plate_gnn_vs_fem"),
        help="Root directory for all outputs.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Device for the GNN forward pass during comparison. Defaults to "
             "'cpu' so the GNN solve time is measured on the SAME hardware as "
             "the FEM (PETSc/MUMPS is CPU-only), giving a fair same-device "
             "speed-up. Pass 'cuda' to time accelerated GNN inference instead "
             "(flagged cross-device).",
    )
    parser.add_argument(
        "--material-model", type=str, default=None,
        choices=["LE", "NH", "le", "nh"],
        help="Material model: 'LE' (linear elastic) or 'NH' (Neo-Hookean). "
             "Auto-detected from the checkpoint if not specified.",
    )
    parser.add_argument(
        "--threads", type=int, default=None, metavar="N",
        help="Pin BOTH the GNN (PyTorch/BLAS) and the FEM (OpenMP/MUMPS) to N "
             "CPU cores for a fair same-core timing comparison, e.g. --threads 1. "
             "Read at import time, so it also works via run_pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        args.model,
        args.input_dir,
        args.output_dir,
        args.device,
        material_model_override=args.material_model,
    )
