"""
Inference + FEM comparison for the 3-D composite rod PI-GNN.

Supports both material models:
  MATERIAL_MODEL = 'NH'  — compressible Neo-Hookean, finite strain (default)
  MATERIAL_MODEL = 'LE'  — small-strain linear elasticity (reference option)

For each mesh case:
  1. GNN prediction (displacement & full 3-D stress) from the trained model.
  2. FEniCSx FEM on the same tetrahedral mesh, same material model and load.
  3. Combined VTU with GNN fields, FEM fields and absolute errors.
  4. CSV with per-node error data.
  5. Summary TXT with L2/R2 metrics, the potential-energy comparison, and the
     torsion-specific outputs (applied torque, achieved twist, warping).

Because the problem is fully three-dimensional, every field carries three
displacement components and six independent stress components. Two derived
quantities are reported that the 2-D plate version had no analogue for:

  * ``sigma_theta_z`` — the torsion shear stress in rod coordinates. For a
    homogeneous shaft this is the ONLY non-zero component, so it is the field
    where the stiff petal inclusion shows up most clearly.
  * the twist angle of the loaded face, which under a dead torque is an OUTPUT
    of the load rather than a prescribed boundary condition.

Unit conventions (all outputs):
  Displacement : mm   (GNN native; FEM m -> x1000)
  Stress       : POSTPROC_OUTPUT_UNIT, default kPa  (FEM Pa -> x PA_TO_UNIT)
  Torque       : <stress unit> * mm^3

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

# R3 imports SummaryWriter at module level; mock before importing.
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

from fem_kernel import (solve_composite, applied_torque,
                                build_traction_measure)
from train_pignn import (
    Config as TrainConfig,
    BoundaryConditions,
    DisplacementGNN,
    CompositeEnergyCalculator,
    MeshGeometry,
    build_graph_data,
    load_mesh_data,
    load_mesh_summary,
    cylindrical_components,
    twist_angle_deg,
    effective_output_displacement_scale,
)


# ── Physical units ────────────────────────────────────────────────────────────
# TrainConfig.YOUNGS_MODULUS_MATRIX and TRACTION_MAGNITUDE are the ACTUAL
# physical values expressed in POSTPROC_OUTPUT_UNIT. Unlike the 2-D plate
# version there is no reduced-modulus training trick: the GNN and the FEM see
# the same material, so GNN displacement is already physical mm and GNN stress
# is already in the config unit. The only conversion left is config unit <-> Pa
# for FEniCSx, which works in SI.

FIXED_INCLUSION_RATIO = float(
    getattr(TrainConfig, "INCLUSION_RATIO", TrainConfig.INCLUSION_RATIO_DEFAULT)
)

_UNIT_TO_PA = {"PA": 1.0, "KPA": 1e3, "MPA": 1e6, "GPA": 1e9}
_OUTPUT_UNIT = str(getattr(TrainConfig, "POSTPROC_OUTPUT_UNIT", "kPa")).upper()
if _OUTPUT_UNIT not in _UNIT_TO_PA:
    raise ValueError(
        f"POSTPROC_OUTPUT_UNIT={TrainConfig.POSTPROC_OUTPUT_UNIT!r} not supported; "
        f"expected one of {list(_UNIT_TO_PA)} (case-insensitive)."
    )
UNIT_TO_PA = _UNIT_TO_PA[_OUTPUT_UNIT]   # config unit -> Pa
PA_TO_UNIT = 1.0 / UNIT_TO_PA            # Pa -> config unit

# Label suffix for stress in all outputs (VTU array names, CSV columns, summary
# tables), derived from POSTPROC_OUTPUT_UNIT so it matches the stored values.
STRESS_UNIT = str(getattr(TrainConfig, "POSTPROC_OUTPUT_UNIT", "kPa"))

# Mesh length unit -> SI metres. Coordinates are authored in mm.
MM_TO_M = 1e-3

FEM_E_MATRIX    = TrainConfig.YOUNGS_MODULUS_MATRIX * UNIT_TO_PA               # Pa
FEM_E_INCLUSION = FEM_E_MATRIX * FIXED_INCLUSION_RATIO                         # Pa
FEM_NU_MATRIX   = TrainConfig.POISSONS_RATIO_MATRIX
FEM_NU_INCLUSION = TrainConfig.POISSONS_RATIO_INCLUSION

# The load is a TORQUE, in <config unit>*mm^3. Converting to SI N*m needs both
# the stress-unit factor and mm^3 -> m^3:
#     1 kPa*mm^3 = 1e3 Pa * 1e-9 m^3 = 1e-6 N*m = 1 uN*m
APPLIED_TORQUE  = float(TrainConfig.APPLIED_TORQUE)                            # unit*mm^3
FEM_TORQUE      = APPLIED_TORQUE * UNIT_TO_PA * (1e-3) ** 3                    # N*m

# NOTE: the rim traction tau is NOT snapshotted here. It depends on the mesh
# (it is solved for so the ASSEMBLED torque equals APPLIED_TORQUE), so it is
# derived per case inside compute_gnn_predictions / run_fenicsx_fem. Freezing a
# tau at import time would load a refined mesh with a slightly different torque
# than the one it was trained under.

# Material model — must match what the checkpoint was trained with.
MATERIAL_MODEL = TrainConfig.MATERIAL_MODEL   # 'NH' or 'LE'

REQUIRED_FILES = [
    "nodes.npy",
    "elements.npy",
    "node_features.npy",
    "loading_surface_facets.npy",
    "node_topology.npy",
    "element_material_ids.npy",
    "mesh_summary.json",
]

# The six independent Cartesian stress components, in the order used everywhere
# (VTU arrays, CSV columns, metric tables) so the outputs stay aligned.
STRESS_KEYS = ("sigma_xx", "sigma_yy", "sigma_zz",
               "sigma_xy", "sigma_yz", "sigma_xz")
# Fields that also get a smoothed nodal projection and enter the error metrics.
NODAL_FIELDS = STRESS_KEYS + ("sigma_theta_z", "von_mises")
# Everything written as raw (unsmoothed) element data in the VTU: the six
# Cartesian components, the four cylindrical ones, and the invariant.
CELL_FIELDS = STRESS_KEYS + ("sigma_rr", "sigma_tt", "sigma_theta_z",
                             "sigma_rz", "von_mises")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ===========================================================================
# Utility helpers
# ===========================================================================

def _is_case_dir(path: Path) -> bool:
    """Report whether ``path`` holds a complete mesh case.

    Args:
        path: Candidate directory.

    Returns:
        bool: True when the directory exists and contains every file in
        ``REQUIRED_FILES``.
    """
    return path.is_dir() and all((path / f).exists() for f in REQUIRED_FILES)


def discover_case_dirs(input_dir: Path) -> list[Path]:
    """Find every mesh case at or below ``input_dir``.

    Args:
        input_dir: A case directory itself, or a tree containing several.

    Returns:
        list[Path]: Case directories, de-duplicated and sorted
        case-insensitively by path.
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


def _cell_to_nodal(num_nodes: int, elements: np.ndarray, cell_vals: np.ndarray) -> np.ndarray:
    """Average a per-element field onto the nodes for smooth plotting.

    Args:
        num_nodes: Number of mesh nodes N.
        elements: Tetrahedron connectivity, shape (E, 4).
        cell_vals: Per-element values, shape (E,).

    Returns:
        ndarray: Nodal values of shape (N,), each the mean over the elements
        incident on that node.
    """
    accum = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)
    for col in range(elements.shape[1]):
        np.add.at(accum, elements[:, col], cell_vals)
        np.add.at(count, elements[:, col], 1.0)
    return (accum / np.maximum(count, 1.0)).astype(np.float32)


def _element_phase(
    elements: np.ndarray,
    element_material_ids: np.ndarray,
    nodes: Optional[np.ndarray] = None,
    inclusions: Optional[list] = None,
) -> np.ndarray:
    """Per-element phase (0 = matrix/outside, 1 = inclusion/inside).

    Primary: sign of the analytic signed distance at the element centroid
    (config.json geometry), so the partition follows the true petal. The
    inclusion is a straight prism, so only the (x, y) centroid matters.
    Fallback (no geometry): the element material id (1 = inclusion).
    """
    if nodes is not None and inclusions:
        try:
            from inclusion_shapes import make_inclusion
            cent = nodes[elements].mean(axis=1)
            inside = np.zeros(len(elements), dtype=bool)
            for inc in inclusions:
                shape = make_inclusion(inc)
                inside |= np.asarray(
                    shape.signed_distance(cent[:, 0], cent[:, 1])) < 0.0
            return inside.astype(np.intp)
        except Exception:
            pass
    return (np.asarray(element_material_ids) == 1).astype(np.intp)


def _node_inside(nodes: Optional[np.ndarray],
                 inclusions: Optional[list]) -> Optional[np.ndarray]:
    """Boolean per-node: inside any inclusion (signed distance < 0)."""
    if nodes is None or not inclusions:
        return None
    try:
        from inclusion_shapes import make_inclusion
        inside = np.zeros(len(nodes), dtype=bool)
        for inc in inclusions:
            shape = make_inclusion(inc)
            inside |= np.asarray(
                shape.signed_distance(nodes[:, 0], nodes[:, 1])) < 0.0
        return inside
    except Exception:
        return None


def _cell_to_nodal_smooth(
    num_nodes: int,
    elements: np.ndarray,
    cell_vals: np.ndarray,
    element_material_ids: np.ndarray,
    element_volumes: np.ndarray,
    smooth_iters: int = 5,
    alpha: float = 0.7,
    nodes: Optional[np.ndarray] = None,
    inclusions: Optional[list] = None,
    return_regions: bool = False,
):
    """
    Two-region cell->nodal stress recovery for a 2-phase composite.

    The rod is partitioned into the inside (inclusion) and outside (matrix) of
    the analytic interface (config.json). The cell field is projected to nodes
    and Laplacian-smoothed INDEPENDENTLY in each region, using an adjacency
    graph built only from that region's elements. Neither pass ever sees the
    other phase, so the genuine matrix<->inclusion stress jump is preserved
    exactly and each side de-noises on its own.

    Interface nodes are shared by both regions, so they carry TWO values: a
    matrix-side value (``val_out``) and an inclusion-side value (``val_in``). The
    VTU writer duplicates these nodes so the rendered boundary is a clean step
    (see _build_interface_split). A node touched by only one region has a valid
    value there and NaN in the other (never read).

    Returns
    -------
    return_regions == False : the combined single-value-per-node array, each node
        taking its own side's smoothed value — used for the error metrics.
    return_regions == True  : (combined, val_out, val_in, elem_phase).
    """
    elements = np.asarray(elements)
    element_volumes = np.asarray(element_volumes, dtype=np.float64)
    cell_vals = np.asarray(cell_vals, dtype=np.float64)

    elem_phase  = _element_phase(elements, element_material_ids, nodes, inclusions)
    node_inside = _node_inside(nodes, inclusions)
    weighted    = cell_vals * element_volumes

    from scipy.sparse import csr_matrix
    import itertools

    n_vert = elements.shape[1]
    # Edges of one cell: all vertex pairs. For a tetrahedron that is the six
    # edges; the 2-D version's fixed 3-cycle would miss half of them.
    cell_edges = list(itertools.combinations(range(n_vert), 2))

    def _project(phase_val):
        """Volume-weighted cell->node projection using only one region's cells."""
        accum = np.zeros(num_nodes, dtype=np.float64)
        wsum  = np.zeros(num_nodes, dtype=np.float64)
        emask = elem_phase == phase_val
        for col in range(n_vert):
            nid = elements[:, col]
            np.add.at(accum, nid[emask], weighted[emask])
            np.add.at(wsum,  nid[emask], element_volumes[emask])
        val = np.where(wsum > 1e-30, accum / np.maximum(wsum, 1e-30), np.nan)
        return val, wsum

    def _smooth_region(val, phase_val):
        """Laplacian relaxation over one region's element-induced subgraph.

        Edges come only from this region's cells, so an interface node averages
        solely over same-region neighbours. Out-of-region nodes have degree 0
        and are frozen (left NaN)."""
        if smooth_iters <= 0:
            return val
        emask = elem_phase == phase_val
        cells = elements[emask]
        if len(cells) == 0:
            return val
        a = np.concatenate([cells[:, i] for i, _ in cell_edges])
        b = np.concatenate([cells[:, j] for _, j in cell_edges])
        rows = np.concatenate([a, b]).astype(np.int32)
        cols = np.concatenate([b, a]).astype(np.int32)
        data = np.ones(len(rows), dtype=np.float64)
        A    = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        deg  = np.asarray(A.sum(axis=1)).ravel()
        invd = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-30), 0.0)
        Anrm = csr_matrix((data * invd[rows], (rows, cols)),
                          shape=(num_nodes, num_nodes))
        v      = np.where(np.isnan(val), 0.0, val)
        freeze = (deg > 0).astype(np.float64)   # only in-region nodes move
        for _ in range(smooth_iters):
            v = v + freeze * alpha * (Anrm.dot(v) - v)
        return np.where(deg > 0, v, np.nan)

    val_out, w_out = _project(0)
    val_in,  w_in  = _project(1)
    val_out = _smooth_region(val_out, 0)
    val_in  = _smooth_region(val_in,  1)

    if node_inside is not None:
        pick_in = node_inside
    else:
        pick_in = w_in > w_out                          # volume-dominance fallback
    combined = np.where(pick_in, val_in, val_out)
    bad = np.isnan(combined)
    if bad.any():
        combined = np.where(bad,
                            np.where(np.isnan(val_in), val_out, val_in),
                            combined)
    combined = np.nan_to_num(combined, nan=0.0).astype(np.float32)

    if return_regions:
        return combined, val_out, val_in, elem_phase
    return combined


def _build_interface_split(nodes: np.ndarray, elements: np.ndarray,
                           elem_phase: np.ndarray):
    """Split the mesh along the matrix/inclusion interface by duplicating the
    shared interface nodes, so matrix and inclusion cells reference distinct
    node copies there. This lets a nodal (point) field carry a true discontinuity
    at the interface — the inside and outside smoothed fields are overlaid with a
    clean step instead of linearly ramping across straddling cells.

    Returns (split_nodes, split_elements, orig_of, side_in):
      split_nodes   : (N+K, 3)   original nodes followed by K interface duplicates
      split_elements: (E, 4)     inclusion cells rewired to the duplicates
      orig_of       : (N+K,)     original node index each split node came from
      side_in       : (N+K,) bool  True -> take the inclusion-side value
    """
    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    N = len(nodes)

    touch_m = np.zeros(N, dtype=bool)
    touch_i = np.zeros(N, dtype=bool)
    for col in range(elements.shape[1]):
        nid = elements[:, col]
        touch_m[nid[elem_phase == 0]] = True
        touch_i[nid[elem_phase == 1]] = True
    iface = touch_m & touch_i
    iface_ids = np.where(iface)[0]

    dup_index = np.full(N, -1, dtype=np.int64)
    dup_index[iface_ids] = N + np.arange(len(iface_ids))

    split_nodes = np.vstack([nodes, nodes[iface_ids]]) if len(iface_ids) else nodes.copy()
    split_elements = elements.copy().astype(np.int64)
    incl_mask = elem_phase == 1
    for col in range(elements.shape[1]):
        v = split_elements[:, col]
        repl = incl_mask & iface[v]
        split_elements[repl, col] = dup_index[v[repl]]

    orig_of = np.concatenate([np.arange(N, dtype=np.int64), iface_ids.astype(np.int64)])
    side_in = np.zeros(len(split_nodes), dtype=bool)
    side_in[:N] = touch_i & (~touch_m)              # pure-inclusion originals
    side_in[N:] = True                              # duplicates are the inclusion side
    return split_nodes, split_elements, orig_of, side_in


# Laplacian smoothing passes for the stress cell->nodal projection. Smoothing
# suppresses the sharp artifacts the projection leaves at the matrix/inclusion
# interface, while the material-partitioned projection preserves the genuine
# stress jump across it.
_STRESS_SMOOTH_ITERS = 3


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
        material_model: ``'NH'`` or ``'LE'``, recorded for reporting.

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
    input_dim  = int(checkpoint.get("input_dim", 5))
    hidden_dim = int(checkpoint.get("hidden_dim", TrainConfig.HIDDEN_DIM))
    num_layers = int(checkpoint.get("num_layers", TrainConfig.NUM_LAYERS))
    # Keep the module-level config in sync: summary.txt reports the ML params
    # from TrainConfig, so a stale value there would mislabel the run.
    TrainConfig.HIDDEN_DIM = hidden_dim
    TrainConfig.NUM_LAYERS = num_layers

    model = DisplacementGNN(
        input_dim=input_dim,     # (X_norm, Y_norm, Z_norm, Material_ID, Interface)
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
    """Read inclusion cross-section geometry from config.json in case_dir."""
    cfg_path = case_dir / "config.json"
    if not cfg_path.exists():
        return []
    import json
    with open(cfg_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return [raw]
    return list(raw)


def _pack_stress_fields(sigma: np.ndarray, centroids: np.ndarray,
                        axis, von_mises: np.ndarray) -> dict:
    """Split a (E,3,3) Cauchy field into the named per-cell arrays used
    everywhere downstream, including the cylindrical torsion component."""
    s_rr, s_tt, s_zz_c, s_tz, s_rz, s_rt = cylindrical_components(
        sigma, centroids, axis)
    return {
        "sigma_xx":      sigma[:, 0, 0].astype(np.float32),
        "sigma_yy":      sigma[:, 1, 1].astype(np.float32),
        "sigma_zz":      sigma[:, 2, 2].astype(np.float32),
        "sigma_xy":      sigma[:, 0, 1].astype(np.float32),
        "sigma_yz":      sigma[:, 1, 2].astype(np.float32),
        "sigma_xz":      sigma[:, 0, 2].astype(np.float32),
        "sigma_rr":      s_rr.astype(np.float32),
        "sigma_tt":      s_tt.astype(np.float32),
        "sigma_theta_z": s_tz.astype(np.float32),
        "sigma_rz":      s_rz.astype(np.float32),
        "von_mises":     np.asarray(von_mises, dtype=np.float32),
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
    axis = (mesh_info["axis_x"], mesh_info["axis_y"])
    R, H = mesh_info["plate_radius"], mesh_info["height"]

    graph_data = build_graph_data(nodes, elements, features, topology).to(device)
    bc_handler = BoundaryConditions(nodes, features)
    ref_coords = torch.tensor(nodes,    dtype=torch.float64, device=device)
    elems_t    = torch.tensor(elements, dtype=torch.long,    device=device)

    energy_calc = CompositeEnergyCalculator(
        TrainConfig.YOUNGS_MODULUS_MATRIX,
        TrainConfig.POISSONS_RATIO_MATRIX,
        TrainConfig.POISSONS_RATIO_INCLUSION,
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

        emu_np, elam_np, _ = energy_calc.get_element_lame(FIXED_INCLUSION_RATIO)
        emu  = torch.tensor(emu_np,  dtype=torch.float64, device=device)
        elam = torch.tensor(elam_np, dtype=torch.float64, device=device)

        sigma_t = energy_calc.stress_tensor(F_def, strain, emu, elam)
        _, _, _, _, _, _, vm_t = energy_calc.stress_components(F_def, strain, emu, elam)
        J_t = torch.linalg.det(F_def)

    disp_mm = disp.cpu().numpy().astype(np.float32)
    sigma   = sigma_t.cpu().numpy()
    vm      = vm_t.cpu().numpy().astype(np.float32)
    Jdet    = J_t.cpu().numpy().astype(np.float32)

    import io as _io, contextlib as _cl
    with _cl.redirect_stdout(_io.StringIO()):
        elem_vols = MeshGeometry.compute_element_volumes(nodes, elements)
        tau, _ = MeshGeometry.traction_for_torque(
            nodes, loading_facets, APPLIED_TORQUE, axis, R, verbose=False)
        nodal_load, M_z = MeshGeometry.compute_nodal_load(
            nodes, loading_facets, tau, axis, R, verbose=False)

    centroids  = nodes[elements].mean(axis=1)
    inclusions = _load_inclusions(case_dir)
    elem_phase = _element_phase(elements, element_material_ids, nodes, inclusions)

    loaded_mask = np.zeros(len(nodes), dtype=bool)
    loaded_mask[bc_handler.loaded_nodes] = True
    twist_deg, twist_mean, twist_max = twist_angle_deg(
        nodes, disp_mm, axis, node_mask=loaded_mask, plate_radius=R)

    out = {
        "nodes":                nodes,
        "elements":             elements,
        "element_material_ids": element_material_ids,
        "element_volumes":      elem_vols,
        "elem_phase":           elem_phase,
        "inclusions":           inclusions,
        "mesh_info":            mesh_info,
        "axis":                 axis,
        "plate_radius":         R,
        "height":               H,
        "applied_torque":       M_z,
        "rim_traction":         tau,
        # The FEM twist is measured on exactly this node set, so it travels with
        # the prediction rather than being re-derived (or patched in) later.
        "loaded_node_ids":      bc_handler.loaded_nodes,
        "gnn_solve_time_s":     gnn_solve_time_s,
        "gnn_device":           str(device),
        "disp":                 disp_mm,
        "twist_deg":            twist_deg.astype(np.float32),
        "twist_mean_deg":       twist_mean,
        "twist_max_deg":        twist_max,
        "J":                    Jdet,
    }
    out.update(_pack_stress_fields(sigma, centroids, axis, vm))

    # Combined nodal field (for metrics) + the two region fields (out/in) that
    # the VTU writer overlays as a discontinuity across the interface.
    for key in NODAL_FIELDS:
        comb, v_out, v_in, _ = _cell_to_nodal_smooth(
            len(nodes), elements, out[key], element_material_ids, elem_vols,
            smooth_iters=_STRESS_SMOOTH_ITERS, nodes=nodes, inclusions=inclusions,
            return_regions=True)
        out[f"{key}_nodal"]     = comb
        out[f"{key}_nodal_out"] = v_out
        out[f"{key}_nodal_in"]  = v_in
    return out


# ===========================================================================
# FEniCSx FEM (heterogeneous materials, 3-D LE or NH)
# ===========================================================================

def run_fenicsx_fem(
    case_dir: Path,
    work_dir: Path,
    gnn: dict,
    material_model: str,
) -> dict:
    """
    FEniCSx composite FEM on the same tetrahedral mesh the GNN used.

    Mesh coordinates are in mm; divided by 1000 for dolfinx (SI m).
    Displacement output (m) multiplied by 1000 to return mm.
    Stress: Pa -> the config stress unit.
    All returned arrays aligned to GNN node/cell ordering.
    """
    from scipy.spatial import cKDTree

    work_dir.mkdir(parents=True, exist_ok=True)

    gnn_nodes    = gnn["nodes"]
    gnn_elements = gnn["elements"]
    gnn_mat_ids  = gnn["element_material_ids"]
    axis_mm      = gnn["axis"]
    R_mm         = gnn["plate_radius"]

    mesh_path = case_dir / "composite_mesh.vtu"
    if not mesh_path.exists():
        raise FileNotFoundError(f"No composite_mesh.vtu in '{case_dir}'")

    meshio_mesh = meshio.read(str(mesh_path))
    tet_cells = [cb.data for cb in meshio_mesh.cells if cb.type == "tetra"]
    if not tet_cells:
        raise ValueError(f"No tetrahedral cells in '{mesh_path}'")

    tets     = np.vstack(tet_cells).astype(np.int32)
    points_m = meshio_mesh.points[:, :3].astype(np.float64) * MM_TO_M  # mm -> m

    xdmf_path = work_dir / "fem_mesh.xdmf"
    meshio.write(str(xdmf_path),
                 meshio.Mesh(points=points_m, cells=[("tetra", tets)]))

    with io.XDMFFile(MPI.COMM_WORLD, str(xdmf_path), "r") as xf:
        domain = xf.read_mesh(name="Grid")

    coords_all = domain.geometry.x          # (N_fem, 3) [m]
    N_fem      = coords_all.shape[0]
    cells_conn = domain.geometry.dofmap.reshape(-1, 4)

    # FEM centroids (mm) for matching against GNN centroids (mm). dolfinx
    # reorders cells and vertices on read, so every field must be remapped
    # rather than assumed aligned.
    fem_centroids_mm = coords_all[cells_conn].mean(axis=1) / MM_TO_M
    gnn_centroids_mm = gnn_nodes[gnn_elements].mean(axis=1)

    gnn_cell_tree = cKDTree(gnn_centroids_mm)
    dists_a, fem_to_gnn_cell = gnn_cell_tree.query(fem_centroids_mm)
    if dists_a.max() > 1e-3:
        print(f"  WARNING: max centroid mismatch FEM->GNN = {dists_a.max():.3e} mm")

    fem_mat_ids = gnn_mat_ids[fem_to_gnn_cell]
    E_arr  = np.where(fem_mat_ids == 1, FEM_E_INCLUSION, FEM_E_MATRIX).astype(np.float64)
    nu_arr = np.where(fem_mat_ids == 1, FEM_NU_INCLUSION, FEM_NU_MATRIX).astype(np.float64)

    V_dg = fem.functionspace(domain, ("DG", 0))
    E_func  = fem.Function(V_dg)
    nu_func = fem.Function(V_dg)
    E_func.x.array[:]  = E_arr
    nu_func.x.array[:] = nu_arr

    fdim  = domain.topology.dim - 1
    z_min = float(coords_all[:, 2].min())
    z_max = float(coords_all[:, 2].max())
    z_tol = max(1e-14, 1e-8 * abs(z_max - z_min))

    # Clamped end at z=0, loaded end at z=H; the lateral surface is free and
    # needs no tag (a traction-free surface is the natural BC).
    fixed_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_min, atol=z_tol))
    loaded_facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[2], z_max, atol=z_tol))

    axis_m = (axis_mm[0] * MM_TO_M, axis_mm[1] * MM_TO_M)
    R_m    = R_mm * MM_TO_M

    # Rim traction that delivers the requested TORQUE on this mesh. Derived from
    # the GNN-side inversion (same mesh, same rule) and converted to Pa, so both
    # solvers are loaded by exactly the same traction field rather than each
    # inverting the torque against its own quadrature.
    FEM_TAU = gnn["rim_traction"] * UNIT_TO_PA

    # ── Time the FEM solve ONLY up to the displacement field ──────────────────
    # `_sigma_expr` is an UNEVALUATED callable, so no stress recovery happens
    # inside the timed region — matching the GNN timing, which also stops at its
    # displacement field.
    #   LE: a single direct LU (MUMPS) solve.
    #   NH: the FULL Newton iteration, including the incremental load-stepping
    #       loop, is INSIDE the timed region. Unlike stress recovery (a one-shot
    #       algebraic post-step), the iterations ARE how the displacement is
    #       computed, so their cost is the honest solve cost.
    # It EXCLUDES the mesh read/convert above, function-space/material setup,
    # and all post-processing below.
    import time as _time
    _fem_t0 = _time.perf_counter()
    uh, _sigma_expr = solve_composite(
        domain, E_func, nu_func, material_model,
        fixed_facets=fixed_facets, loaded_facets=loaded_facets,
        traction=FEM_TAU, axis=axis_m, plate_radius=R_m, fdim=fdim,
    )
    fem_solve_time_s = _time.perf_counter() - _fem_t0

    ds_loaded = build_traction_measure(domain, fdim, loaded_facets)
    M_z_fem_SI = applied_torque(domain, FEM_TAU, axis_m, R_m, ds_loaded)
    # SI N*m -> <stress unit>*mm^3, so it is directly comparable to the GNN's.
    M_z_fem = M_z_fem_SI * PA_TO_UNIT / (MM_TO_M ** 3)

    # ── Extract displacements ────────────────────────────────
    _, _, geom_vtk = plot.vtk_mesh(uh.function_space)
    u_raw       = uh.x.array.reshape((geom_vtk.shape[0], 3))
    disp_fem_mm = (u_raw[:N_fem] / MM_TO_M).astype(np.float32)   # m -> mm

    # ── Stress interpolation ─────────────────────────────────
    def _interp_raw(expr_ufl):
        """Interpolate a UFL expression onto DG0 and return the raw cell values."""
        expr = fem.Expression(expr_ufl, V_dg.element.interpolation_points())
        func = fem.Function(V_dg)
        func.interpolate(expr)
        return func.x.array.astype(np.float64).copy()

    def _interp_unit(expr_ufl):
        """Same, for a stress expression: Pa -> the config stress unit."""
        return _interp_raw(expr_ufl) * PA_TO_UNIT

    se = _sigma_expr(uh)
    sig_cells = np.zeros((len(cells_conn), 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            v = _interp_unit(se[i, j])
            sig_cells[:, i, j] = v
            sig_cells[:, j, i] = v

    # von Mises from the full 3-D deviator, identical to the GNN definition
    # (CompositeEnergyCalculator.von_mises). In torsion the out-of-plane shears
    # carry most of the load, so all three shear terms must be present.
    sxx, syy, szz = sig_cells[:, 0, 0], sig_cells[:, 1, 1], sig_cells[:, 2, 2]
    sxy, syz, sxz = sig_cells[:, 0, 1], sig_cells[:, 1, 2], sig_cells[:, 0, 2]
    vm_cells = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))

    F_ufl = ufl.Identity(3) + ufl.grad(uh)
    J_cells = _interp_raw(ufl.det(F_ufl))                 # dimensionless

    # ── Remap to GNN ordering ─────────────────────────────────
    # The nodal STRESS arrays are assembled from domain.geometry.dofmap, so they
    # live in geometry-node order — match them against the geometry coordinates.
    fem_coords_mm = coords_all[:N_fem, :3] / MM_TO_M
    node_tree     = cKDTree(fem_coords_mm)
    node_dists, fem_to_gnn_node = node_tree.query(gnn_nodes)
    if node_dists.max() > 1e-4:
        print(f"  WARNING: max node mismatch FEM<->GNN = {node_dists.max():.3e} mm")

    # The DISPLACEMENT (uh.x.array) lives in function-space DOF order, which
    # dolfinx reorders independently of the geometry-node order. It must be
    # matched against its own DOF coordinates — geom_vtk from
    # plot.vtk_mesh(uh.function_space) is exactly the coordinate array that
    # aligns with disp_fem_mm. Pairing the displacement with geometry
    # coordinates instead scrambles the field (u no longer ~0 on the clamped
    # face, spurious strain energy).
    dof_coords_mm = geom_vtk[:N_fem, :3] / MM_TO_M
    dof_tree      = cKDTree(dof_coords_mm)
    dof_dists, fem_to_gnn_dof = dof_tree.query(gnn_nodes)
    if dof_dists.max() > 1e-4:
        print(f"  WARNING: max DOF mismatch FEM<->GNN = {dof_dists.max():.3e} mm")

    fem_cell_tree = cKDTree(fem_centroids_mm)
    cell_dists_b, gnn_to_fem_cell = fem_cell_tree.query(gnn_centroids_mm)
    if cell_dists_b.max() > 1e-3:
        print(f"  WARNING: max centroid mismatch GNN->FEM = {cell_dists_b.max():.3e} mm")

    disp_gnn_order = disp_fem_mm[fem_to_gnn_dof].astype(np.float32)
    sigma_gnn_order = sig_cells[gnn_to_fem_cell]
    vm_gnn_order    = vm_cells[gnn_to_fem_cell].astype(np.float32)

    nodes    = gnn_nodes
    elements = gnn_elements
    centroids = nodes[elements].mean(axis=1)

    res = {
        "nodes":            nodes,
        "elements":         elements,
        "fem_solve_time_s": fem_solve_time_s,
        # FEniCSx/PETSc solves with a MUMPS direct LU factorization — CPU-only;
        # there is no GPU path here regardless of the GNN device.
        "fem_device":       "cpu (PETSc/MUMPS)",
        "disp":             disp_gnn_order,
        "J":                J_cells[gnn_to_fem_cell].astype(np.float32),
        "applied_torque":   M_z_fem,
    }
    res.update(_pack_stress_fields(sigma_gnn_order, centroids, axis_mm, vm_gnn_order))

    loaded_mask = np.zeros(len(nodes), dtype=bool)
    loaded_mask[gnn["loaded_node_ids"]] = True
    twist_deg, twist_mean, twist_max = twist_angle_deg(
        nodes, disp_gnn_order, axis_mm, node_mask=loaded_mask, plate_radius=R_mm)
    res["twist_deg"]      = twist_deg.astype(np.float32)
    res["twist_mean_deg"] = twist_mean
    res["twist_max_deg"]  = twist_max

    # Node-averaged stresses (two-region projection + smoothing), on the GNN
    # ordering so both sides share one node numbering.
    elem_vols  = gnn["element_volumes"]
    inclusions = gnn["inclusions"]
    for key in NODAL_FIELDS:
        comb, v_out, v_in, _ = _cell_to_nodal_smooth(
            len(nodes), elements, res[key], gnn_mat_ids, elem_vols,
            smooth_iters=_STRESS_SMOOTH_ITERS, nodes=nodes, inclusions=inclusions,
            return_regions=True)
        res[f"{key}_nodal"]     = comb
        res[f"{key}_nodal_out"] = v_out
        res[f"{key}_nodal_in"]  = v_in

    return res


# ===========================================================================
# Combined VTU writer
# ===========================================================================

def write_combined_vtu(gnn: dict, fem_res: dict, out_vtu: Path) -> None:
    """Write one VTU holding the GNN field, the FEM field and their difference.

    The mesh is split along the analytic material interface by duplicating the
    nodes that sit on it, so the nodal stress fields can carry a genuine step
    there instead of being averaged across the discontinuity.

    Args:
        gnn: GNN result dict, supplying the mesh, displacement and stresses.
        fem_res: FEM result dict on the same node ordering.
        out_vtu: Destination ``.vtu`` path.
    """
    nodes    = gnn["nodes"]
    elements = gnn["elements"]
    N        = len(nodes)

    # ── Split the mesh along the analytic interface so the nodal stress fields
    #    can carry a clean discontinuity there (inside/outside overlaid). ──
    elem_phase = gnn.get("elem_phase")
    if elem_phase is None:
        elem_phase = _element_phase(elements, gnn["element_material_ids"],
                                    nodes, gnn.get("inclusions"))
    split_nodes, split_elements, orig_of, side_in = _build_interface_split(
        nodes, elements, elem_phase)
    Ns = len(split_nodes)

    def _exp_scalar(arr):
        """Continuous scalar field -> copy to duplicated nodes."""
        return np.asarray(arr, dtype=np.float64)[:N][orig_of]

    def _exp_vec(arr2d):
        """Continuous 3-vector field -> copy to duplicated nodes."""
        return np.asarray(arr2d, dtype=np.float64)[:N][orig_of]

    def _exp_disc(v_out, v_in):
        """Discontinuous field: each split node takes its own side's value, so
        the interface becomes a true step."""
        a_out = np.asarray(v_out, dtype=np.float64)[:N]
        a_in  = np.asarray(v_in,  dtype=np.float64)[:N]
        res = np.where(side_in, a_in[orig_of], a_out[orig_of])
        return np.nan_to_num(res, nan=0.0)

    out_vtu.parent.mkdir(parents=True, exist_ok=True)
    with open(out_vtu, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{Ns}" NumberOfCells="{len(split_elements)}">\n')

        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x, y, z in split_nodes:
            f.write(f"          {x:.10g} {y:.10g} {z:.10g}\n")
        f.write("        </DataArray>\n      </Points>\n")

        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        for n0, n1, n2, n3 in split_elements:
            f.write(f"          {int(n0)} {int(n1)} {int(n2)} {int(n3)}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        for i in range(1, len(split_elements) + 1):
            f.write(f"          {4 * i}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in split_elements:
            f.write("          10\n")     # VTK_TETRA
        f.write("        </DataArray>\n      </Cells>\n")

        # VTK picks the ACTIVE point vector/scalar from these attributes. Without
        # Vectors= the reader registers the displacement arrays but leaves
        # active_vectors = None, and ParaView's "Warp By Vector" then has nothing
        # to default to — the deformed shape cannot be displayed in one click.
        # Naming GNN_displacement_mm here makes warping work immediately; switch
        # to FEM_displacement_mm in the ParaView array dropdown to warp by the
        # reference solution instead.
        f.write(f'      <PointData Scalars="GNN_von_mises_{STRESS_UNIT}" '
                f'Vectors="GNN_displacement_mm">\n')

        def _pt_vec3(name, vals_3d):
            """Write one per-node 3-vector array into the open PointData block."""
            f.write(f'        <DataArray type="Float64" Name="{name}" '
                    f'NumberOfComponents="3" format="ascii">\n')
            for u, v, w in vals_3d:
                f.write(f"          {float(u):.10g} {float(v):.10g} {float(w):.10g}\n")
            f.write("        </DataArray>\n")

        def _pt_scalar(name, vals):
            """Write one per-node scalar array into the open PointData block."""
            f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            for v in vals:
                f.write(f"          {float(v):.10g}\n")
            f.write("        </DataArray>\n")

        gnn_mag = np.linalg.norm(gnn["disp"][:N], axis=1)
        fem_mag = np.linalg.norm(fem_res["disp"][:N], axis=1)

        # Discontinuous (overlaid) GNN/FEM stress fields and their abs error.
        gnn_s = {k: _exp_disc(gnn[f"{k}_nodal_out"], gnn[f"{k}_nodal_in"])
                 for k in NODAL_FIELDS}
        fem_s = {k: _exp_disc(fem_res[f"{k}_nodal_out"], fem_res[f"{k}_nodal_in"])
                 for k in NODAL_FIELDS}

        _pt_vec3("GNN_displacement_mm", _exp_vec(gnn["disp"]))
        _pt_scalar("GNN_disp_mag_mm",   _exp_scalar(gnn_mag))
        _pt_scalar("GNN_uz_warping_mm", _exp_scalar(gnn["disp"][:, 2]))
        _pt_scalar("GNN_twist_deg",     _exp_scalar(gnn["twist_deg"]))
        for k in NODAL_FIELDS:
            _pt_scalar(f"GNN_{k}_{STRESS_UNIT}", gnn_s[k])

        _pt_vec3("FEM_displacement_mm", _exp_vec(fem_res["disp"]))
        _pt_scalar("FEM_disp_mag_mm",   _exp_scalar(fem_mag))
        _pt_scalar("FEM_uz_warping_mm", _exp_scalar(fem_res["disp"][:, 2]))
        _pt_scalar("FEM_twist_deg",     _exp_scalar(fem_res["twist_deg"]))
        for k in NODAL_FIELDS:
            _pt_scalar(f"FEM_{k}_{STRESS_UNIT}", fem_s[k])

        _pt_scalar("AbsErr_disp_mag_mm", _exp_scalar(np.abs(gnn_mag - fem_mag)))
        _pt_scalar("AbsErr_uz_mm",
                   _exp_scalar(np.abs(gnn["disp"][:, 2] - fem_res["disp"][:, 2])))
        for k in NODAL_FIELDS:
            _pt_scalar(f"AbsErr_{k}_{STRESS_UNIT}", np.abs(gnn_s[k] - fem_s[k]))
        _pt_scalar("Material_ID", side_in.astype(np.float64))
        f.write("      </PointData>\n")

        f.write(f'      <CellData Scalars="GNN_von_mises_cell_{STRESS_UNIT}">\n')

        def _cell_scalar(name, vals):
            """Write one per-element scalar array into the open CellData block."""
            f.write(f'        <DataArray type="Float64" Name="{name}" format="ascii">\n')
            for v in vals:
                f.write(f"          {float(v):.10g}\n")
            f.write("        </DataArray>\n")

        # The raw, unsmoothed element field for EVERY stress component — the six
        # Cartesian ones plus the four cylindrical ones and the invariant. The
        # PointData copies above are projected and Laplacian-smoothed for
        # display; these are what the metrics are actually computed from, so
        # both belong in the file.
        for k in CELL_FIELDS:
            _cell_scalar(f"GNN_{k}_cell_{STRESS_UNIT}", gnn[k])
            _cell_scalar(f"FEM_{k}_cell_{STRESS_UNIT}", fem_res[k])
            _cell_scalar(f"AbsErr_{k}_cell_{STRESS_UNIT}",
                         np.abs(gnn[k] - fem_res[k]))
        _cell_scalar("GNN_J_volume_ratio", gnn["J"])
        _cell_scalar("FEM_J_volume_ratio", fem_res["J"])
        _cell_scalar("Element_Material_ID",
                     gnn["element_material_ids"].astype(np.float32))
        f.write("      </CellData>\n")
        f.write("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")


# ===========================================================================
# CSV error export
# ===========================================================================

def save_error_csv(gnn: dict, fem_res: dict, out_csv: Path) -> None:
    """Export the per-node GNN/FEM comparison as a CSV table.

    Args:
        gnn: GNN result dict.
        fem_res: FEM result dict on the same node ordering.
        out_csv: Destination ``.csv`` path; parent directories are created.

    Side effects:
        Writes one row per node, holding both solutions plus absolute and
        relative errors for the displacement and every nodal stress field.
    """
    import pandas as pd
    N      = len(gnn["nodes"])
    coords = gnn["nodes"][:N]
    axis   = gnn["axis"]

    mag_gnn = np.linalg.norm(gnn["disp"][:N],     axis=1)
    mag_fem = np.linalg.norm(fem_res["disp"][:N], axis=1)
    abs_disp    = np.linalg.norm(gnn["disp"][:N] - fem_res["disp"][:N], axis=1)
    fem_mag_max = mag_fem.max()
    rel_disp    = abs_disp / fem_mag_max if fem_mag_max > 1e-30 else np.zeros(N)

    data = {
        "node_id":     np.arange(N),
        "x_mm":        coords[:, 0],
        "y_mm":        coords[:, 1],
        "z_mm":        coords[:, 2],
        "r_mm":        np.hypot(coords[:, 0] - axis[0], coords[:, 1] - axis[1]),
        "material_id": _cell_to_nodal(N, gnn["elements"],
                                      gnn["element_material_ids"].astype(np.float32)),
        "ux_gnn_mm":   gnn["disp"][:N, 0],
        "uy_gnn_mm":   gnn["disp"][:N, 1],
        "uz_gnn_mm":   gnn["disp"][:N, 2],
        "mag_gnn_mm":  mag_gnn,
        "twist_gnn_deg": gnn["twist_deg"][:N],
        "ux_fem_mm":   fem_res["disp"][:N, 0],
        "uy_fem_mm":   fem_res["disp"][:N, 1],
        "uz_fem_mm":   fem_res["disp"][:N, 2],
        "mag_fem_mm":  mag_fem,
        "twist_fem_deg": fem_res["twist_deg"][:N],
        "abs_disp_err_mm": abs_disp,
        "rel_disp_err":    rel_disp,
    }

    for key in NODAL_FIELDS:
        gnn_s = gnn[f"{key}_nodal"][:N]
        fem_s = fem_res[f"{key}_nodal"][:N]
        abs_s = np.abs(gnn_s - fem_s)
        max_s = np.abs(fem_s).max()
        rel_s = abs_s / max_s if max_s > 1e-30 else np.zeros(N)
        data[f"{key}_gnn_{STRESS_UNIT}"]     = gnn_s
        data[f"{key}_fem_{STRESS_UNIT}"]     = fem_s
        data[f"abs_{key}_err_{STRESS_UNIT}"] = abs_s
        data[f"rel_{key}_err"]               = rel_s

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
        gnn: GNN result dict, supplying the displacement and the rim traction.
        fem_res: FEM result dict on the same node ordering.
        case_dir: Mesh case directory the arrays are reloaded from.
        device: Torch device the energy is evaluated on.
        material_model: ``'NH'`` or ``'LE'``.

    Returns:
        dict: ``fem_tpe``, ``gnn_tpe``, the ``E_int`` and ``W_ext`` split for
        each, and ``tpe_rel_diff``, the relative difference in Pi.
    """
    nodes, elements, features, loading_facets, topology, element_material_ids = (
        load_mesh_data(str(case_dir))
    )
    import io as _io, contextlib as _cl
    with _cl.redirect_stdout(_io.StringIO()):
        element_volumes = MeshGeometry.compute_element_volumes(nodes, elements)
        nodal_load, _ = MeshGeometry.compute_nodal_load(
            nodes, loading_facets, gnn["rim_traction"],
            gnn["axis"], gnn["plate_radius"], verbose=False)

    energy_calc = CompositeEnergyCalculator(
        TrainConfig.YOUNGS_MODULUS_MATRIX,
        TrainConfig.POISSONS_RATIO_MATRIX,
        TrainConfig.POISSONS_RATIO_INCLUSION,
        element_material_ids,
        material_model=material_model,
        output_unit=STRESS_UNIT,
    )

    ref_coords_t = torch.tensor(nodes,           dtype=torch.float64, device=device)
    elements_t   = torch.tensor(elements,        dtype=torch.long,    device=device)
    volumes_t    = torch.tensor(element_volumes, dtype=torch.float64, device=device)
    load_t       = torch.tensor(nodal_load,      dtype=torch.float64, device=device)

    def _tpe(disp_mm: np.ndarray):
        """Evaluate Pi for one displacement field.

        Args:
            disp_mm: Nodal displacements in mm, shape (N, 3).

        Returns:
            tuple: ``(Pi, E_int, W_ext)`` as Python floats.
        """
        disp_t = torch.tensor(np.asarray(disp_mm, dtype=np.float64),
                              dtype=torch.float64, device=device)
        with torch.no_grad():
            Pi, E_int, W_ext, _, _ = energy_calc.compute_total_potential_energy(
                ref_coords_t, disp_t, elements_t, volumes_t, load_t,
                inclusion_ratio=FIXED_INCLUSION_RATIO,
            )
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


def _nearfield_mask(nodes: np.ndarray, inclusions: list, factor: float = 2.0) -> np.ndarray:
    """
    Boolean per node: inside the inclusion, or within a band of width
    char_length*(factor-1) outside its boundary. The inclusion is a straight
    prism, so the test uses only (x, y) and the band is a shell running the full
    height of the rod.
    """
    from inclusion_shapes import make_inclusion
    mask = np.zeros(len(nodes), dtype=bool)
    for inc in inclusions:
        shape = make_inclusion(inc)
        band  = shape.char_length() * (factor - 1.0)
        sd    = shape.signed_distance(nodes[:, 0], nodes[:, 1])   # vectorized
        mask |= np.asarray(sd) <= band
    return mask


# Fields carried through the R2 / signed-error tables. Kept short deliberately:
# the six Cartesian components all appear in the CSV and the L2 table, but the
# summary reads better with the components that actually carry the torsion.
_R2_FIELDS = ("sigma_theta_z", "sigma_zz", "von_mises")


def compute_r2_metrics(gnn: dict, fem_res: dict, case_dir: Optional[Path] = None) -> dict:
    """Score the GNN against the FEM, globally and split by distance to the inclusion.

    The near-field / far-field split separates the region where the material
    discontinuity makes the field hard to represent from the smooth bulk, so a
    good global score cannot hide a poor interface fit.

    Args:
        gnn: GNN result dict.
        fem_res: FEM result dict on the same node ordering.
        case_dir: Case directory used to load the inclusion geometry when the
            GNN dict does not already carry it. Without it every node is
            treated as near-field.

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

    for i, comp in enumerate(("ux", "uy", "uz")):
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
        err_abs = sqrt( Sum_i ||X_i - X^_i||^2 )        [mm for u, stress unit for sigma]

    Relative L2 error (element VOLUME-weighted), in %:
        err_rel = sqrt( Sum_e V_e (X_e - X^_e)^2 ) / sqrt( Sum_e V_e X_e^2 ) x 100

    The weight is the tetrahedron volume, not an area as in the 2-D version — the
    graded mesh has elements differing by an order of magnitude in size, so an
    unweighted norm would over-count the refined interface band.
    """
    N        = len(gnn["nodes"])
    elements = gnn["elements"]
    vols     = gnn["element_volumes"].astype(np.float64)

    def _rel_vol(true_cell, pred_cell):
        """Volume-weighted relative L2 error of a per-element field, in percent.

        Weighting by element volume makes the norm mesh-grading-independent, so
        a refined patch does not dominate the score by element count alone.
        """
        num = np.sqrt(np.sum(vols * (pred_cell - true_cell) ** 2))
        den = np.sqrt(np.sum(vols * true_cell ** 2))
        return float(num / den * 100.0) if den > 1e-30 else float("nan")

    u_fem = fem_res["disp"][:N].astype(np.float64)
    u_gnn = gnn["disp"][:N].astype(np.float64)
    abs_disp = float(np.sqrt(np.sum((u_gnn - u_fem) ** 2)))                 # mm
    uf_e = u_fem[elements].mean(axis=1)   # nodal -> element average
    ug_e = u_gnn[elements].mean(axis=1)
    num  = np.sqrt(np.sum(vols * np.sum((ug_e - uf_e) ** 2, axis=1)))
    den  = np.sqrt(np.sum(vols * np.sum(uf_e ** 2, axis=1)))
    rel_disp = float(num / den * 100.0) if den > 1e-30 else float("nan")

    out = {"abs_disp_mm": abs_disp, "rel_disp_pct": rel_disp}

    # uz alone: warping is the small, hard-to-get component and its error is
    # invisible inside the displacement-magnitude norm, which the large
    # in-plane rotation dominates.
    abs_uz = float(np.sqrt(np.sum((u_gnn[:, 2] - u_fem[:, 2]) ** 2)))
    den_uz = float(np.sqrt(np.sum(vols * uf_e[:, 2] ** 2)))
    num_uz = float(np.sqrt(np.sum(vols * (ug_e[:, 2] - uf_e[:, 2]) ** 2)))
    out["abs_uz_mm"] = abs_uz
    out["rel_uz_pct"] = (num_uz / den_uz * 100.0) if den_uz > 1e-30 else float("nan")

    # Stress fields: absolute L2 on nodal values, relative L2 on cell values
    for name in NODAL_FIELDS:
        sf = fem_res[f"{name}_nodal"][:N].astype(np.float64)
        sg = gnn[f"{name}_nodal"][:N].astype(np.float64)
        out[f"abs_{name}"] = float(np.sqrt(np.sum((sg - sf) ** 2)))
        out[f"rel_{name}_pct"] = _rel_vol(
            fem_res[name].astype(np.float64), gnn[name].astype(np.float64))
    return out


def compute_l2_metrics_nearfield(gnn: dict, fem_res: dict,
                                 case_dir: Optional[Path] = None,
                                 factor: float = 1.5) -> dict:
    """
    Near-field relative L2 errors — the same volume-weighted quantity as
    compute_l2_metrics, restricted to elements in the interface band.

    An element is in the band when ANY of its vertices is within the near-field
    mask, i.e. inside the inclusion or within char_length*(factor-1) outside its
    boundary. The band spans BOTH phases deliberately: an inclusion-only metric
    throws away the matrix side of the interface, which is exactly where the
    strain concentration the graph model is supposed to resolve lives.
    """
    N        = len(gnn["nodes"])
    elements = gnn["elements"]
    vols     = gnn["element_volumes"].astype(np.float64)

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

    v_m = vols[elem_mask]

    def _rel_vol_m(true_cell, pred_cell):
        """Volume-weighted relative L2 error over the near-field elements only."""
        t = np.asarray(true_cell, dtype=np.float64)[elem_mask]
        p = np.asarray(pred_cell, dtype=np.float64)[elem_mask]
        num = np.sqrt(np.sum(v_m * (p - t) ** 2))
        den = np.sqrt(np.sum(v_m * t ** 2))
        return float(num / den * 100.0) if den > 1e-30 else float("nan")

    u_fem = fem_res["disp"][:N].astype(np.float64)
    u_gnn = gnn["disp"][:N].astype(np.float64)
    uf_e = u_fem[elements].mean(axis=1)[elem_mask]
    ug_e = u_gnn[elements].mean(axis=1)[elem_mask]
    num  = np.sqrt(np.sum(v_m * np.sum((ug_e - uf_e) ** 2, axis=1)))
    den  = np.sqrt(np.sum(v_m * np.sum(uf_e ** 2, axis=1)))

    out = {"rel_disp_nf_pct": float(num / den * 100.0) if den > 1e-30 else float("nan"),
           "_n_nf_elements": n_el,
           "_nf_factor": float(factor)}
    for name in NODAL_FIELDS:
        out[f"rel_{name}_nf_pct"] = _rel_vol_m(fem_res[name], gnn[name])
    return out


def compute_signed_rel_errors(gnn: dict, fem_res: dict,
                              floor_frac: float = 1e-2) -> dict:
    """
    Signed pointwise relative error per field, over nodes (FEM = ground truth):

        s_i = (GNN_i - FEM_i) / |FEM_i| x 100   [%]
              ( +  GNN over-predicts ,  -  GNN under-predicts )

    Reported as median (robust — primary) and mean. Nodes where
    |FEM_i| < floor_frac * max|FEM| are excluded: there the reference is ~0
    (zero-crossings, the clamped face) and the ratio is meaningless.
    """
    N   = len(gnn["nodes"])
    out = {}
    fields = [("ux", gnn["disp"][:N, 0], fem_res["disp"][:N, 0]),
              ("uy", gnn["disp"][:N, 1], fem_res["disp"][:N, 1]),
              ("uz", gnn["disp"][:N, 2], fem_res["disp"][:N, 2])]
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
              ("uz (warping)", "abs_uz_mm",   "rel_uz_pct",   "mm")]
    fields += [(n, f"abs_{n}", f"rel_{n}_pct", STRESS_UNIT) for n in NODAL_FIELDS]
    lines = [
        "=" * 78,
        "  PI-GNN Composite Rod (torsion) — Aggregate L2 errors over M test cases",
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
        material_model: ``'NH'`` or ``'LE'``.
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
                f"uz {_fmt(r2[f'R2_uz{key}'])}  s_tz {_fmt(r2[f'R2_sigma_theta_z{key}'])}  "
                f"s_zz {_fmt(r2[f'R2_sigma_zz{key}'])}  vm {_fmt(r2[f'R2_von_mises{key}'])}")

    nodes = gnn["nodes"][:N]
    info  = gnn["mesh_info"]
    R, H  = gnn["plate_radius"], gnn["height"]
    unit  = STRESS_UNIT
    E_mat = float(TrainConfig.YOUNGS_MODULUS_MATRIX)
    E_inc = E_mat * FIXED_INCLUSION_RATIO

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

    gnn_uz = float(np.abs(gnn["disp"][:, 2]).max())
    fem_uz = float(np.abs(fem_res["disp"][:, 2]).max())

    le_warning = []
    if material_model.upper() != "NH":
        le_warning = [
            "",
            "  *** WARNING: material model is LE (small-strain linear elasticity).",
            "  *** At this load the rod twists by tens of degrees, where linear",
            "  *** kinematics cannot represent finite rotation. The numbers below",
            "  *** are a code-to-code check only, NOT physically meaningful.",
        ]

    lines = [
        "=" * 78,
        "  PI-GNN Composite Rod in Torsion — Prediction Summary",
        "=" * 78,
        f"  Generated  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Case       : {case_name}",
        f"  Checkpoint : {checkpoint_path}",
        f"  Checkpoint selection : {_CKPT_INFO.get('selection', 'unknown')}"
        + (f"   @ epoch {_CKPT_INFO['epoch']},  Pi = {_CKPT_INFO['loss']:.8e}"
           if _CKPT_INFO.get("loss") is not None else ""),
        f"  Material model : {material_model}",
        f"  Problem type   : 3-D composite rod (2-phase), clamped-twisted",
    ] + le_warning + [

        "",
        "-- Material properties ------------------------------------------------",
        f"  Inclusion ratio (E_inc/E_mat) : {FIXED_INCLUSION_RATIO:.4f}",
        f"  E_matrix                      : {E_mat:.4g} {unit}   ({FEM_E_MATRIX:.4g} Pa)",
        f"  E_inclusion                   : {E_inc:.4g} {unit}   ({FEM_E_INCLUSION:.4g} Pa)",
        f"  nu_matrix / nu_inclusion      : {FEM_NU_MATRIX}  /  {FEM_NU_INCLUSION}",
        f"  Constitutive model            : "
        + ("compressible Neo-Hookean, finite strain (full 3-D)"
           if material_model.upper() == "NH"
           else "linear elastic, small strain (full 3-D)"),

        "",
        "-- Geometry & boundary conditions -------------------------------------",
        f"  Rod radius R / height H : {R:.4f} mm / {H:.4f} mm   (H/2R = {H/(2*R):.3f})",
        f"  Torsion axis            : ({gnn['axis'][0]:.6f}, {gnn['axis'][1]:.6f}) mm",
        f"  Formulation             : full 3-D (no plane-stress/plane-strain assumption)",
        f"  z = 0                   : CLAMPED, ux = uy = uz = 0",
        f"  z = H                   : prescribed TORQUE, applied as the tangential "
        f"dead traction t = (tau/R)(-(y-cy), (x-cx), 0)",
        f"  lateral surface         : free (traction-free)",
        f"  APPLIED TORQUE (input)  : {APPLIED_TORQUE:.6g} {unit}*mm^3"
        + (f"  =  {APPLIED_TORQUE:.6g} uN*m  =  {FEM_TORQUE:.6g} N*m"
           if unit.upper() == 'KPA' else f"  =  {FEM_TORQUE:.6g} N*m"),
        f"  -> rim traction tau     : {gnn['rim_traction']:.6g} {unit}  "
        f"({gnn['rim_traction'] * UNIT_TO_PA:.6g} Pa)   [derived from the torque "
        f"on this mesh]",
        f"  Nodes / Tetrahedra      : {N} / {len(gnn['elements'])}",
        f"  Extrusion layers        : {info.get('n_layers')} "
        f"({info.get('n_section_nodes')} nodes x {info.get('n_section_elements')} tris "
        f"per cross-section)",

        "",
        "-- Torsion response (the headline outputs) ----------------------------",
        f"  Requested torque Mz   : {APPLIED_TORQUE:.6f} [{unit}*mm^3]   (the INPUT)",
        f"  Realised torque Mz    : GNN {gnn['applied_torque']:.6f}   "
        f"FEM {fem_res['applied_torque']:.6f}",
        f"    (tau was solved for so the assembled torque equals the request, so",
        f"     all three agree; a mismatch means the two solvers read different",
        f"     meshes or different axes)",
        f"  Mean twist of z=H     : GNN {gnn['twist_mean_deg']:8.3f} deg   "
        f"FEM {fem_res['twist_mean_deg']:8.3f} deg   "
        f"(err {gnn['twist_mean_deg'] - fem_res['twist_mean_deg']:+.3f} deg)",
        f"  Max  twist of z=H     : GNN {gnn['twist_max_deg']:8.3f} deg   "
        f"FEM {fem_res['twist_max_deg']:8.3f} deg",
        f"  Rim shear gamma=phi*R/H : "
        f"{np.radians(fem_res['twist_mean_deg']) * R / H:.4f}  (FEM)  "
        f"-- far beyond the ~0.02-0.05 linear-elastic range",
        f"  Max |uz| (warping)    : GNN {gnn_uz:.6f} mm   FEM {fem_uz:.6f} mm",
        f"  Max |u|               : GNN {float(np.linalg.norm(gnn['disp'], axis=1).max()):.6f} mm"
        f"   FEM {float(np.linalg.norm(fem_res['disp'], axis=1).max()):.6f} mm",
        f"  Volume ratio J        : GNN [{gnn['J'].min():.4f}, {gnn['J'].max():.4f}]   "
        f"FEM [{fem_res['J'].min():.4f}, {fem_res['J'].max():.4f}]   (1.0 = incompressible)",

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
        "  Field            Absolute L2           Relative L2 (volume-weighted)",
        _l2row("Displacement", l2["abs_disp_mm"], "mm", l2["rel_disp_pct"]),
        _l2row("uz (warping)", l2["abs_uz_mm"],   "mm", l2["rel_uz_pct"]),
    ] + [
        _l2row(n, l2[f"abs_{n}"], unit, l2[f"rel_{n}_pct"]) for n in NODAL_FIELDS
    ] + [

        "",
        "-- L2 errors — interface band (near-field, both phases) ---------------",
        f"  Band = inclusion interior + matrix shell of 0.5x char_length "
        f"({l2.get('_n_nf_elements', 0)} elements)",
        "  Field                                Relative L2 (volume-weighted)",
        f"  Displacement                         {l2.get('rel_disp_nf_pct', float('nan')):>10.4f} %",
    ] + [
        f"  {n:<36s} {l2.get(f'rel_{n}_nf_pct', float('nan')):>10.4f} %"
        for n in NODAL_FIELDS
    ] + [

        "",
        "-- R2 scores (GNN vs FEM, nodal fields) -------------------------------",
        _r2row("[global]", ""),
        f"  [near-field: <= 2x inclusion size, {n_near} nodes]",
        _r2row("", "_nf"),
        f"  [far-field: remainder, {n_far} nodes]",
        _r2row("", "_ff"),

        "",
        f"-- Energy & loss metrics (units {unit}*mm^3; Pi = training loss) -------",
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
        "    The two functionals are the SAME to machine precision (verified at",
        "    ~3e-15 relative for both LE and NH on an identical displacement",
        "    field), so this is a like-for-like comparison and the minimum",
        "    principle applies: Pi_GNN >= Pi_FEM, approaching it from above.",

        "",
        "-- Signed relative error per field — (GNN-FEM)/|FEM| x 100, over nodes -",
        "  (+ over-prediction, - under-prediction; |FEM| < 1% of peak excluded)",
        "  Field            median          mean",
        _srow("ux", signed_rel),
        _srow("uy", signed_rel),
        _srow("uz", signed_rel),
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
        material_model_override: Force ``'NH'`` or ``'LE'`` instead of the model
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
    print(f"Inclusion ratio   : {FIXED_INCLUSION_RATIO:.4f}")

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
        print(f"    GNN  twist   = {gnn['twist_mean_deg']:.3f} deg (mean, loaded face)")
        print(f"    GNN  VM  max = {gnn['von_mises_nodal'].max():.4f} {STRESS_UNIT}")
        print(f"    GNN  solve time = {gnn['gnn_solve_time_s'] * 1e3:.3f} ms")

        fem_work = case_out / "_fem_work"
        print(f"  Running FEniCSx FEM ({mat_model}) ...")
        fem_res = run_fenicsx_fem(case_dir, fem_work, gnn, material_model=mat_model)
        print(f"    FEM  |u| max = {float(np.linalg.norm(fem_res['disp'], axis=1).max()):.4e} mm")
        print(f"    FEM  twist   = {fem_res['twist_mean_deg']:.3f} deg (mean, loaded face)")
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
        print(f"    rel L2 uz warping  : {l2_metrics['rel_uz_pct']:.4f} %")
        print(f"    rel L2 sigma_theta_z: {l2_metrics['rel_sigma_theta_z_pct']:.4f} %")
        print(f"    rel L2 von Mises   : {l2_metrics['rel_von_mises_pct']:.4f} %")
        print(f"    rel L2 vm (near)   : {l2_metrics['rel_von_mises_nf_pct']:.4f} %   "
              f"[{l2_metrics['_n_nf_elements']} interface-band elements]")

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
            "GNN inference + FEniCSx FEM comparison for the 3-D composite rod in "
            "torsion. Outputs combined VTU, per-node error CSV, and summary TXT."
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
        "--output-dir", type=Path, default=Path("rod_gnn_vs_fem"),
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
        help="Material model: 'NH' (Neo-Hookean) or 'LE' (linear elastic). "
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
