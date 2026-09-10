"""
Physics-Informed Graph Neural Network for the 2-D Plate with a Central Hole
===========================================================================
Trains a GNN to predict the displacement field of a homogeneous 2-D plate
containing circular holes (voids) under a right-edge traction, by minimising
the total potential energy

        Pi(u) = E_int(u) - W_ext(u)  ->  min

over the space of kinematically admissible displacements. The plate is a single
matrix phase: the void interiors were removed at mesh time, so every hole rim is
an exterior traction-free free surface and no material interface remains.

Problem setup
-------------
  Geometry   square plate, side L, with circular hole(s) cut out of it, meshed
             with linear triangles.
  x = x_min  ROLLER, ux = 0 (uy free)         — hard-imposed by masking
  top-left   PIN, uy = 0                      — removes the last rigid-body mode
  x = x_max  loaded by a uniform dead traction t = (T, 0) in +x
  hole rims  free (traction-free, natural BC)
  y edges    free

Material models
---------------
  'LE'  small-strain plane-stress linear elasticity (the default; the load here
        is a nominal strain of ~5e-4, squarely inside the linear range):
            Psi = lambda_ps/2 (tr eps)^2 + mu tr(eps^2)
  'NH'  compressible Neo-Hookean, finite strain, with sigma_33 = 0 imposed
        RIGOROUSLY through the out-of-plane stretch F33:
            Psi = mu/2 (I1 - 3) - mu ln J + lambda_3d/2 (ln J)^2

Lame convention. LE uses the plane-stress-reduced lambda_ps; NH uses the FULL
3-D lambda, because its plane-stress condition is produced by the F33
condensation and the reduced lambda_ps would double-count that correction.
"""

import gc
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch.optim import Adam
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — prevents display server leaks
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Training configuration"""

    # ── Material model ────────────────────────────────────────
    # 'LE' = small-strain linear elasticity, plane stress (the physical choice
    #        here: the steel plate is loaded to ~0.05 % nominal strain)
    # 'NH' = compressible Neo-Hookean, finite strain, rigorous plane stress
    MATERIAL_MODEL = 'LE'

    # Paths
    DATA_DIR        = 'mesh_plate_hole'
    CHECKPOINT_DIR  = 'plate_checkpoints'
    RESULTS_DIR     = 'plate_results'

    # ── Material properties ───────────────────────────────────
    # Steel plate, in POSTPROC_OUTPUT_UNIT. The plate is a SINGLE homogeneous
    # phase (the holes are voids), so there is no inclusion contrast and no
    # second Poisson ratio.
    #
    # These are the ACTUAL physical values. There is no reduced-modulus training
    # trick: the network trains on the real material and the loss is conditioned
    # solely by OUTPUT_DISPLACEMENT_SCALE below.
    YOUNGS_MODULUS_MATRIX = 210e3    # MPa  (= 210 GPa, steel)
    POISSONS_RATIO_MATRIX = 0.3

    # Unit that YOUNGS_MODULUS_MATRIX and TRACTION_MAGNITUDE are expressed in.
    # Mesh coordinates are in mm; the FEM converts both to SI internally.
    POSTPROC_OUTPUT_UNIT = 'MPa'

    # ── Loading: the uniform right-edge traction ──────────────
    # Must scale WITH the modulus: nominal strain is T/E, so a traction tuned
    # for a soft matrix gives ~0 strain against 210 GPa steel.
    #   T = 1.0 MPa -> nominal strain T/E = 4.8e-6, far-field sigma_xx ~ 1 MPa,
    #   peak ~3x that at the hole rim (the classic Kt = 3 concentration).
    # The traction is a DEAD load: the vector is fixed in the reference frame,
    # so W_ext = Int t.u dS is a genuine potential and Pi = E_int - W_ext is a
    # real energy the GNN can minimise.
    TRACTION_MAGNITUDE = 1.0     # MPa on the right edge

    # ── Output displacement scale (conditioning of the energy loss) ───────────
    # The GNN's raw output magnitude at initialisation is O(1e-2), set by weight
    # init and independent of the physics. When that disagrees with the true
    # displacement by orders of magnitude, the two terms of Pi = E_int - W_ext
    # are wildly unbalanced at init and the only meaningful gradient is
    # "shrink u", while W_ext — the term that creates the deformation SHAPE — is
    # numerically negligible. Training then collapses to a near-trivial field
    # regardless of epoch count. Here the true displacement is O(1e-4) mm
    # against an O(1e-2) initialisation: two orders out.
    #
    # Multiplying the network output by a fixed characteristic displacement lets
    # the network learn O(1) values while the PHYSICAL strain is unchanged.
    #
    #   None  -> auto: (T / E) * L_char, a nominal strain times the plate size
    #            taken from the mesh. Here 1.0/210e3 * 20 = 9.5e-5 mm, within a
    #            factor of a few of the true peak displacement.
    #   1.0   -> disabled
    #   float -> explicit
    # Results are insensitive to the exact value; only the ORDER OF MAGNITUDE
    # matters. The scale is NOT a parameter, so it is absent from
    # model_state_dict — it is recorded in the checkpoint
    # (``output_displacement_scale``) and restored by ``make_model`` at
    # inference. Rebuilding a model without it silently rescales the whole field.
    OUTPUT_DISPLACEMENT_SCALE = None

    # ── GNN architecture ──────────────────────────────────────
    HIDDEN_DIM  = 64      # width of every hidden node-feature vector (channels)
    NUM_LAYERS  = 4       # message-passing steps; sets the receptive field in hops

    # ── Training ──────────────────────────────────────────────
    # LEARNING_RATE is matched to the SCALED output: the network now learns O(1)
    # values, so the step size that suited the old raw-magnitude output (1e-4)
    # is an order of magnitude too small here.
    LEARNING_RATE = 1e-3  # Adam step size (dimensionless, on the scaled output)
    NUM_EPOCHS    = 20000 # full-batch energy-minimisation steps
    PRINT_EVERY   = 1000  # epochs between console/TensorBoard progress reports
    SAVE_EVERY    = 5000  # epochs between checkpoint + field-plot writes

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CPU throttle. Count the cores this process may actually run on, not the
    # machine total: under `taskset` / `numactl` / a cgroup, os.cpu_count() still
    # reports every core, which would over-provision threads.
    try:
        _total_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        _total_cores = os.cpu_count() or 1
    if torch.cuda.is_available():
        CPU_INTRA_THREADS  = 2
        CPU_INTER_THREADS  = 1
        CPU_AFFINITY_CORES = 2
    else:
        CPU_INTRA_THREADS  = _total_cores
        CPU_INTER_THREADS  = max(1, _total_cores // 4)
        CPU_AFFINITY_CORES = _total_cores

    # torch.compile — fuses the GNN forward/backward for faster execution.
    # Set via env COMPOSITE_TORCH_COMPILE=0 to disable. Any compile/trace
    # failure falls back to eager automatically, so it can never break a run.
    USE_TORCH_COMPILE = os.environ.get('COMPOSITE_TORCH_COMPILE', '1') != '0'

    # Numerical stability: floor on a triangle area (mm^2). Prevents a
    # degenerate/sliver element from contributing a zero or negative weight to
    # the energy sum, which would make the loss non-differentiable there.
    MIN_AREA = 1e-14


def _normalized_material_model(config):
    """Read the material model off a config as an upper-case string.

    Args:
        config: Config-like object; a missing ``MATERIAL_MODEL`` defaults to
            linear elasticity.

    Returns:
        str: Either ``'LE'`` or ``'NH'``.
    """
    return str(getattr(config, 'MATERIAL_MODEL', 'LE')).upper()


def effective_output_displacement_scale(config, char_length=None):
    """Resolve OUTPUT_DISPLACEMENT_SCALE (see Config for the rationale).

    ``None`` -> (TRACTION_MAGNITUDE / E) * char_length, which puts the network's
    learned output at O(1) without perturbing the physical strain.
    ``char_length`` defaults to the plate size recorded on the config (set from
    the mesh by ``main``); if that is unavailable it falls back to 1.0, which
    just makes the scale a nominal strain.
    """
    scale = getattr(config, 'OUTPUT_DISPLACEMENT_SCALE', None)
    if scale is None:
        E = float(config.YOUNGS_MODULUS_MATRIX)
        if E <= 0.0:
            raise ValueError("Young's modulus must be positive.")
        if char_length is None:
            char_length = float(getattr(config, 'PLATE_SIZE', 1.0) or 1.0)
        T = float(config.TRACTION_MAGNITUDE)
        return T / E * float(char_length)
    scale = float(scale)
    if scale == 0.0:
        raise ValueError("OUTPUT_DISPLACEMENT_SCALE must be non-zero.")
    return scale


# ============================================================
# CPU THROTTLE
# ============================================================

def setup_cpu_limits(config):
    """
    Restrict CPU thread count and optionally pin process to a subset of cores.
    """
    print("\n" + "="*60)
    print("CPU LIMIT SETUP")
    print("="*60)

    torch.set_num_threads(config.CPU_INTRA_THREADS)
    try:
        torch.set_num_interop_threads(config.CPU_INTER_THREADS)
    except RuntimeError:
        pass  # inter-op pool already initialised; intra-op limit still holds
    print(f"  PyTorch intra-op threads : {config.CPU_INTRA_THREADS}")
    print(f"  PyTorch inter-op threads : {config.CPU_INTER_THREADS}")

    if hasattr(os, 'sched_setaffinity'):
        n = config.CPU_AFFINITY_CORES
        total = os.cpu_count() or 1
        try:
            current = os.sched_getaffinity(0)
        except OSError:
            current = None
        if current is not None and len(current) < total:
            # A launcher (taskset/numactl/cgroup) already pinned us to a subset.
            # Re-setting affinity here would WIDEN it back to every core.
            print(f"  CPU affinity: pre-set to {len(current)} core(s) by the "
                  f"launcher — left unchanged")
        else:
            cores = set(range(min(n, total)))
            try:
                os.sched_setaffinity(0, cores)
                print(f"  CPU affinity set to cores : {sorted(cores)}")
            except PermissionError:
                print("  CPU affinity: permission denied — skipped (not root)")
    else:
        print("  CPU affinity: not supported on this OS — skipped")

    print(f"  Device: {config.DEVICE}")


def maybe_compile_model(model, config):
    """Return (run_model, orig_model) for training.

    `run_model` is a torch.compile-wrapped model used for the hot training
    forward/backward.  `orig_model` is always the original, uncompiled
    nn.Module — it shares the same parameter tensors, so it carries the
    trained weights AND exposes clean state_dict keys (no `_orig_mod.`
    prefix).  Checkpoints are therefore saved from `orig_model` and load
    cleanly into a plain DisplacementGNN during the Stage-3 FEM comparison.
    """
    if not getattr(config, 'USE_TORCH_COMPILE', False):
        print("  torch.compile: disabled (COMPOSITE_TORCH_COMPILE=0) — eager mode")
        return model, model
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        compiled = torch.compile(model, dynamic=False)
        print("  torch.compile: ENABLED (inductor backend, dynamic=False)")
        print("    NOTE: first epoch is slow (one-time graph compilation).")
        return compiled, model
    except Exception as exc:  # pragma: no cover — defensive
        print(f"  torch.compile: unavailable ({exc!r}) — falling back to eager")
        return model, model


# ============================================================
# BOUNDARY CONDITION HANDLER
# ============================================================

class BoundaryConditions:
    """Hard-imposed Dirichlet conditions for the loaded hole plate.

        uy = 0  (pin)
          |
          v
        +-----------------------------+
        |  o                          |
        |     (  )  hole rim: free    | --> uniform traction t = (T, 0)
        |  o                          | -->
        +-----------------------------+
         ^
         ux = 0 on the whole left edge (roller; uy free)

    The roller edge kills the x translation and the rotation; the single
    top-left pin kills the remaining y translation. Together they remove all
    three rigid-body modes without over-constraining the plate — a fully fixed
    left edge would add a spurious transverse restraint and change the physics.

    The conditions are imposed *hard* by zeroing the network output at every
    constrained DOF, so they hold exactly at every epoch: the network never has
    to learn them and no penalty term competes with the energy.

    The roller and loaded node sets come from the Dirichlet/Neumann flags in
    ``node_features`` (columns 0 and 1), which the mesh processor set from the
    plate edges. Re-deriving them here from a coordinate query would risk a
    different tolerance than the mesh processor used and silently free or
    over-constrain a strip of nodes.
    """

    def __init__(self, nodes, features, tolerance=1e-6):
        """Build the constrained node sets and the per-component free masks.

        Args:
            nodes: Reference nodal coordinates, shape (N, 2), in mm.
            features: Node feature matrix, shape (N, F); column 0 flags
                left-edge (roller) nodes and column 1 flags loaded-edge nodes.
            tolerance: Coordinate tolerance used to pick the pinned corner out
                of the roller edge.

        Raises:
            ValueError: If no roller or no pinned node is found, either of which
                would leave the energy minimisation singular under rigid-body
                motion.
        """
        self.num_nodes = len(nodes)
        self.tolerance = tolerance

        x_coords = nodes[:, 0]
        y_coords = nodes[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        print("\n" + "="*60)
        print("BOUNDARY CONDITIONS SETUP")
        print("="*60)
        print(f"Domain: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}]")

        self.left_edge_nodes  = np.where(features[:, 0] > 0.5)[0]
        self.right_edge_nodes = np.where(features[:, 1] > 0.5)[0]
        # Alias matching the 3-D pipeline's vocabulary; used by the diagnostics.
        self.loaded_nodes = self.right_edge_nodes

        if len(self.left_edge_nodes) == 0:
            raise ValueError(
                "No roller nodes found (node_features column 0 is all zero). "
                "Re-run the meshing stage — without a Dirichlet set the energy "
                "minimisation is singular under rigid-body motion.")

        # The pin is the top-most node OF THE ROLLER EDGE, so it is guaranteed
        # to be a member of that set rather than a second, independently
        # tolerance-matched query that could land on a different node.
        left_y = y_coords[self.left_edge_nodes]
        self.top_left_corner = self.left_edge_nodes[
            np.abs(left_y - left_y.max()) < tolerance]
        if len(self.top_left_corner) == 0:
            raise ValueError("Failed to locate the top-left pin on the roller edge.")

        # (N,) multiplicative masks: 0 on a constrained DOF, 1 elsewhere. Two
        # separate columns because each constraint fixes ONE component, unlike
        # the 3-D rod's clamp which fixes the whole vector.
        self.u_fixed_mask = np.ones(self.num_nodes, dtype=np.float64)
        self.u_fixed_mask[self.left_edge_nodes] = 0.0

        self.v_fixed_mask = np.ones(self.num_nodes, dtype=np.float64)
        self.v_fixed_mask[self.top_left_corner] = 0.0

        print(f"\nLeft edge (roller)  : {len(self.left_edge_nodes)} nodes (ux = 0, uy free)")
        print(f"Top-left corner (pin): {len(self.top_left_corner)} node(s) (ux = uy = 0)")
        print(f"Right edge (loaded) : {len(self.right_edge_nodes)} nodes "
              f"(uniform traction -> +x)")
        print("Hole rims / y edges : free (traction-free)")
        print(f"Top-left corner node IDs: {self.top_left_corner}")
        print(f"  Coordinates: {nodes[self.top_left_corner]}")

        self._u_fixed_tensor = None
        self._v_fixed_tensor = None
        self._cached_device  = None

    def apply(self, displacements):
        """Zero the displacement at every constrained degree of freedom.

        The mask tensors are cached per device, so repeated calls in the
        training loop do not re-upload them.

        Args:
            displacements: Predicted displacements, shape (N, 2).

        Returns:
            Tensor: Displacements of shape (N, 2), with ux identically zero on
            the roller edge and uy identically zero at the pin.
        """
        dev = displacements.device
        if self._cached_device != dev:
            self._u_fixed_tensor = torch.tensor(
                self.u_fixed_mask, dtype=torch.float64, device=dev)
            self._v_fixed_tensor = torch.tensor(
                self.v_fixed_mask, dtype=torch.float64, device=dev)
            self._cached_device  = dev

        u_constrained = displacements[:, 0] * self._u_fixed_tensor
        v_constrained = displacements[:, 1] * self._v_fixed_tensor
        return torch.stack([u_constrained, v_constrained], dim=1)


# ============================================================
# GEOMETRY AND MESH UTILITIES
# ============================================================

class MeshGeometry:
    """Geometric properties of the triangular mesh."""

    @staticmethod
    def compute_element_areas(nodes, elements):
        """Triangle areas  |(v1-v0) x (v2-v0)| / 2."""
        print("\nComputing element areas...")
        v0 = nodes[elements[:, 0]]
        v1 = nodes[elements[:, 1]]
        v2 = nodes[elements[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        areas = np.maximum(0.5 * np.abs(cross), Config.MIN_AREA)

        print(f"  Element areas - Min: {areas.min():.6e}, Max: {areas.max():.6e}, "
              f"Mean: {areas.mean():.6e}")
        print(f"  Total area: {areas.sum():.6f}")
        return areas.astype(np.float64)

    @staticmethod
    def compute_facet_lengths(nodes, facets):
        """Edge lengths of ``[elem_id, n1, n2]`` facet rows."""
        facets = np.asarray(facets, dtype=np.int64).reshape(-1, 3)
        p1 = nodes[facets[:, 1]]
        p2 = nodes[facets[:, 2]]
        return np.linalg.norm(p2 - p1, axis=1).astype(np.float64)

    @staticmethod
    def compute_edge_lengths(nodes, facets):
        """Loaded-edge lengths, with the usual startup report."""
        print("\nComputing loading edge lengths...")
        lengths = MeshGeometry.compute_facet_lengths(nodes, facets)
        print(f"  Edge lengths - Min: {lengths.min():.6e}, Max: {lengths.max():.6e}")
        print(f"  Total loading edge length: {lengths.sum():.6f}")
        return lengths

    @staticmethod
    def compute_nodal_load(nodes, loading_facets, traction, verbose=True):
        """
        Assemble the consistent nodal force vector of the right-edge traction.

        The applied field is the uniform dead load  t(X) = (T, 0)  on the loaded
        edge. Over one linear segment both t and u are linear, so the exact
        external work is  Int t.u dS = Sum_i u_i . f_i  with the consistent
        nodal force

            f_i = L/6 * (2 t_i + t_j) = L/6 * (t_i + (t_i + t_j))

        For a uniform traction this collapses to the familiar f_i = T*L/2 per
        endpoint, so it reproduces the midpoint rule the earlier version used
        EXACTLY — the general form is kept because it stays exact if the
        traction is ever made position-dependent, and because it mirrors the
        assembly the 3-D pipeline uses.

        Returns
        -------
        nodal_load : (N, 2)  force per node; W_ext = sum(nodal_load * u).
        F_x        : float   assembled resultant force in +x.
        """
        if verbose:
            print("\nAssembling right-edge traction nodal load...")
        N = len(nodes)

        # Traction vector at every node (only loaded-edge rows are ever used).
        t = np.zeros((N, 2), dtype=np.float64)
        t[:, 0] = float(traction)

        facets = np.asarray(loading_facets, dtype=np.int64).reshape(-1, 3)
        lengths = MeshGeometry.compute_facet_lengths(nodes, facets)

        nodal_load = np.zeros((N, 2), dtype=np.float64)
        n1, n2 = facets[:, 1], facets[:, 2]
        s = t[n1] + t[n2]                                # (Fc, 2)
        w = (lengths / 6.0)[:, None]
        np.add.at(nodal_load, n1, w * (t[n1] + s))
        np.add.at(nodal_load, n2, w * (t[n2] + s))

        # Diagnostics. The assembled resultant must equal T x (loaded length) in
        # x and vanish in y — both are properties of the assembly, so checking
        # them here catches a bad facet set before any training time is spent.
        f_res = nodal_load.sum(axis=0)
        F_x = float(f_res[0])
        if verbose:
            F_analytic = float(traction) * float(lengths.sum())
            print(f"  Loading facets   : {len(facets)}  "
                  f"(total length {lengths.sum():.6f})")
            print(f"  Resultant force  : [{f_res[0]:.6e}, {f_res[1]:.3e}]"
                  f"   (y component must be ~0)")
            print(f"  Applied force Fx : {F_x:.6e}   "
                  f"(analytic T*L = {F_analytic:.6e}, "
                  f"rel. err {abs(F_x / F_analytic - 1) if F_analytic else float('nan'):.2e})")
        return nodal_load, F_x


# ============================================================
# ENERGY CALCULATIONS
# ============================================================

class CompositeEnergyCalculator:
    """
    Total potential energy of the 2-D hole plate.

    The plate is a single matrix phase (the holes are voids), so the Lame
    parameters are uniform across every element.

    'LE'  — plane-stress linear elasticity
                Psi = lambda_ps/2 (tr eps)^2 + mu tr(eps^2)
            evaluated on the SMALL strain eps = sym(grad u) = 1/2 (F + F^T) - I,
            which is exactly the measure the FEM's LE branch uses. The two
            solvers therefore discretise the same functional and the comparison
            is a true code-to-code check rather than a constitutive mismatch.

    'NH'  — compressible Neo-Hookean, finite strain, plane stress
                Psi = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2
            with I1 = tr(C) = tr(F^T F) and J = det F. The plane-stress
            constraint sigma_33 = 0 is enforced by solving dPsi/dF33 = 0 for the
            out-of-plane stretch F33 and substituting it back, giving the
            effective 2-D energy.

    IMPORTANT — Lame parameters. Because the NH plane-stress reduction is
    produced BY the F33 condensation, the NH model must use the FULL 3-D
    lambda_3d = E nu / ((1+nu)(1-2nu)). Using the reduced plane-stress lambda_ps
    there would double-count the sigma_33 = 0 correction. The LE model, in
    contrast, has its plane-stress reduction baked into lambda_ps and uses that.
    """

    def __init__(self, E_matrix, nu_matrix,
                 element_material_ids, material_model='LE', output_unit='MPa'):
        """Precompute the matrix Lame parameters for the active model.

        Args:
            E_matrix: Matrix Young's modulus, in ``output_unit``.
            nu_matrix: Matrix Poisson ratio (dimensionless).
            element_material_ids: Per-element material tag, shape (E,). Every
                entry is 0 here (single matrix phase); the array is retained so
                the element layout matches the composite pipeline.
            material_model: ``'LE'`` for linear elasticity or ``'NH'`` for
                Neo-Hookean.
            output_unit: Stress unit that ``E_matrix`` is expressed in, used
                for reporting only.
        """
        self.E_matrix             = E_matrix
        self.nu_matrix            = nu_matrix
        self.material_model       = material_model.upper()
        self.element_material_ids = element_material_ids
        self.output_unit          = output_unit
        self.num_elements         = len(element_material_ids)

        self.mu_matrix, self.lam_matrix = self._element_lame(E_matrix, nu_matrix)

        self._lame_cache = {}

        print("\n" + "="*60)
        print(f"PLATE MATERIAL PROPERTIES — {self.material_model} (plane stress)")
        print("="*60)
        print(f"MATRIX (single phase):")
        print(f"  Young's Modulus E: {E_matrix:.4g} {self.output_unit}")
        print(f"  Poisson's Ratio: {nu_matrix:.4f}")
        _lam_kind = ("3-D (plane stress via F33)" if self.material_model == 'NH'
                     else "plane-stress lambda_ps")
        print(f"  Shear Modulus mu: {self.mu_matrix:.4g} {self.output_unit}")
        print(f"  Lame lambda ({_lam_kind}): {self.lam_matrix:.4g} {self.output_unit}")
        print(f"\nELEMENT DISTRIBUTION:")
        print(f"  Matrix triangles: {self.num_elements} "
              f"(all elements; the void interiors were removed at mesh time)")

    # ── Lame parameter helpers ────────────────────────────────

    @staticmethod
    def _plane_stress_lame(E, nu):
        """Plane-stress Lame parameters (mu, lam_ps) — for the LE model."""
        mu     = E / (2.0 * (1.0 + nu))
        lam_3d = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        lam_ps = (2.0 * mu * lam_3d) / (lam_3d + 2.0 * mu)
        return mu, lam_ps

    @staticmethod
    def _full_3d_lame(E, nu):
        """Full 3-D Lame parameters (mu, lam_3d) — for the NH model, whose
        plane-stress condition is enforced through the out-of-plane stretch
        F33, so the reduced lam_ps must NOT be used here."""
        mu     = E / (2.0 * (1.0 + nu))
        lam_3d = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return mu, lam_3d

    def _element_lame(self, E, nu):
        """Select the Lame convention appropriate to the active material model.

        Args:
            E: Young's modulus, in the configured output unit.
            nu: Poisson ratio (dimensionless).

        Returns:
            tuple: ``(mu, lambda)`` in the same unit as ``E``.
        """
        if self.material_model == 'NH':
            return self._full_3d_lame(E, nu)
        return self._plane_stress_lame(E, nu)

    def get_element_lame(self):
        """Per-element Lame parameters (uniform matrix values everywhere).

        LE -> plane-stress-reduced lambda_ps; NH -> full 3-D lambda (plane
        stress imposed via the F33 condensation)."""
        element_mu  = np.full(self.num_elements, self.mu_matrix,  dtype=np.float64)
        element_lam = np.full(self.num_elements, self.lam_matrix, dtype=np.float64)
        return element_mu, element_lam

    # ── Deformation-gradient and strain helpers ───────────────

    def compute_deformation_gradient(self, ref_coords, cur_coords, elements):
        """
        F = Ds @ Dm^-1  for each linear triangle, where Dm and Ds hold the two
        edge vectors from the first vertex in the reference and current
        configurations. F is constant per element (linear shape functions).

        Returns (E, 2, 2).
        """
        ref_elem = ref_coords[elements]        # (E, 3, 2)
        cur_elem = cur_coords[elements]

        Dm = torch.stack([ref_elem[:, 1] - ref_elem[:, 0],
                          ref_elem[:, 2] - ref_elem[:, 0]], dim=2)   # (E,2,2)
        Ds = torch.stack([cur_elem[:, 1] - cur_elem[:, 0],
                          cur_elem[:, 2] - cur_elem[:, 0]], dim=2)
        return torch.matmul(Ds, torch.inverse(Dm))

    # Backwards-compatible alias (the name older call sites used).
    compute_deformation_gradient_2d = compute_deformation_gradient

    def compute_green_lagrange_strain(self, F):
        """Green-Lagrange strain E = 1/2 (F^T F - I), 2x2."""
        FTF = torch.matmul(F.transpose(-2, -1), F)
        I   = torch.eye(2, dtype=F.dtype, device=F.device).unsqueeze(0)
        return 0.5 * (FTF - I)

    def compute_small_strain(self, F):
        """Small (engineering) strain eps = sym(grad u) = 1/2 (F + F^T) - I."""
        I = torch.eye(2, dtype=F.dtype, device=F.device).unsqueeze(0)
        return 0.5 * (F + F.transpose(-2, -1)) - I

    def compute_strain_tensor(self, F):
        """The strain measure the ACTIVE material model is defined on.

        'LE' -> small strain sym(grad u). This must match the FEM, which solves
        sigma = lambda_ps tr(sym grad u) I + 2 mu sym(grad u); using
        Green-Lagrange here instead would have the GNN minimise the St.
        Venant-Kirchhoff energy while the reference stayed small-strain. The two
        agree only to O(|grad u|^2), so any residual difference would show up as
        a "network error" that is really a constitutive mismatch between the two
        solvers rather than a modelling error in the GNN.

        'NH' -> Green-Lagrange, the natural finite-strain measure. The
        Neo-Hookean energy and stress are written directly in terms of F, so
        this value is diagnostic only and never enters the NH energy.
        """
        if self.material_model == 'NH':
            return self.compute_green_lagrange_strain(F)
        return self.compute_small_strain(F)

    # ── Strain-energy densities ───────────────────────────────

    def compute_strain_energy_density_LE(self, strain, element_mu, element_lam):
        """Plane-stress LE strain energy density Psi = lam/2 (tr eps)^2 + mu tr(eps^2)."""
        tr_e    = strain[:, 0, 0] + strain[:, 1, 1]
        e_sq    = torch.matmul(strain, strain)
        tr_esq  = e_sq[:, 0, 0] + e_sq[:, 1, 1]
        return 0.5 * element_lam * tr_e**2 + element_mu * tr_esq

    def compute_strain_energy_density_NH(self, F, element_mu, element_lam):
        """
        Compressible Neo-Hookean strain energy density under plane stress.

        Full 3-D energy:  W = mu/2 (I1-3) - mu ln J + lambda/2 (ln J)^2
        with the FULL 3-D Lame lambda (see the class docstring).

        The plane-stress constraint sigma_33 = 0 <=> dW/dF33 = 0. With
        J = J2D * F33 this stationarity condition is

            dW/dF33 = mu F33 - mu/F33 + lambda ln(J)/F33 = 0
            =>  mu F33^2 + lambda ln(F33) - mu + lambda ln(J2D) = 0   (F33 > 0)

        which is transcendental (no closed form) and is solved by Newton
        iteration from F33 = 1 (undeformed). J2D = det(F_2D) is the 2-D Jacobian
        and J = J2D * F33 the 3-D one. Substituting F33 back gives the effective
        2-D energy with sigma_33 = 0 satisfied exactly.

        Parameters
        ----------
        F            : (E, 2, 2)   in-plane deformation gradient
        element_mu   : (E,)        shear modulus per element
        element_lam  : (E,)        FULL 3-D Lame lambda per element
        """
        # 2-D Jacobian  J2D = det(F_2D)
        J2D = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]    # (E,)
        # Guard the logarithm. A triangle that inverts mid-training (J <= 0)
        # would otherwise produce NaN and poison every subsequent gradient;
        # clamping leaves such an element with a large finite energy, which the
        # minimiser then pushes back out of inversion.
        J2D = torch.clamp(J2D, min=1e-8)

        lnJ2D = torch.log(J2D)                                         # (E,)

        x = torch.ones_like(J2D)
        for _ in range(8):
            g  = element_mu * x * x + element_lam * torch.log(x) \
                 - element_mu + element_lam * lnJ2D
            dg = 2.0 * element_mu * x + element_lam / x
            x  = x - g / dg
            x  = torch.clamp(x, min=1e-8)

        F33 = x                                    # out-of-plane stretch (E,)
        J   = J2D * F33                            # full 3-D Jacobian

        # I1 = tr(F^T F) in 3-D = tr(F_2D^T F_2D) + F33^2
        FTF_2D = torch.matmul(F.transpose(-2, -1), F)   # (E,2,2)
        I1_2D  = FTF_2D[:, 0, 0] + FTF_2D[:, 1, 1]      # (E,) — 2-D contribution
        I1     = I1_2D + F33**2

        lnJ    = torch.log(J)

        return 0.5 * element_mu * (I1 - 3.0) - element_mu * lnJ \
            + 0.5 * element_lam * lnJ**2

    # ── Internal energy ───────────────────────────────────────

    def _get_cached_lame(self, dev):
        """Return per-element Lame tensors, memoised by device.

        Args:
            dev: Torch device the tensors are needed on.

        Returns:
            tuple: ``(element_mu, element_lambda)``, each of shape (E,) and
            dtype float64.
        """
        key = str(dev)
        if key not in self._lame_cache:
            emu_np, elam_np = self.get_element_lame()
            self._lame_cache[key] = (
                torch.tensor(emu_np,  dtype=torch.float64, device=dev),
                torch.tensor(elam_np, dtype=torch.float64, device=dev),
            )
        return self._lame_cache[key]

    def compute_internal_energy(self, ref_coords, cur_coords, elements,
                                element_areas):
        """Integrate the strain-energy density over the whole plate.

        Args:
            ref_coords: Reference nodal coordinates, shape (N, 2).
            cur_coords: Deformed nodal coordinates, shape (N, 2).
            elements: Triangle connectivity, shape (E, 3).
            element_areas: Reference element areas, shape (E,).

        Returns:
            tuple: ``(E_internal, F, strain)`` where ``E_internal`` is a scalar
            tensor, ``F`` is the deformation gradient of shape (E, 2, 2) and
            ``strain`` is the active model's strain measure, shape (E, 2, 2).
            For the Neo-Hookean model ``strain`` is diagnostic only and never
            enters the energy.
        """
        F   = self.compute_deformation_gradient(ref_coords, cur_coords, elements)
        dev = ref_coords.device
        emu, elam = self._get_cached_lame(dev)

        if self.material_model == 'NH':
            psi    = self.compute_strain_energy_density_NH(F, emu, elam)
            strain = self.compute_strain_tensor(F)   # for debug / viz only
        else:
            strain = self.compute_strain_tensor(F)
            psi    = self.compute_strain_energy_density_LE(strain, emu, elam)

        E_internal = torch.sum(psi * element_areas)
        return E_internal, F, strain

    # ── External work ─────────────────────────────────────────

    def compute_external_work(self, displacements, nodal_load):
        """
        W_ext = Int t.u dS = sum_i f_i . u_i

        ``nodal_load`` is the exact consistent nodal force vector of the
        right-edge traction (see MeshGeometry.compute_nodal_load), so this is an
        exact surface integral for a linear displacement field, not a quadrature
        approximation. It is also linear in u, which is what makes Pi a genuine
        potential and the dead load conservative.
        """
        return torch.sum(nodal_load * displacements)

    # ── Total potential energy ────────────────────────────────

    def compute_total_potential_energy(self, ref_coords, displacements, elements,
                                       element_areas, nodal_load):
        """Evaluate Pi(u) = E_int(u) - W_ext(u), the quantity training minimises.

        Args:
            ref_coords: Reference nodal coordinates, shape (N, 2).
            displacements: Predicted displacements, shape (N, 2), already
                masked by the boundary conditions.
            elements: Triangle connectivity, shape (E, 3).
            element_areas: Reference element areas, shape (E,).
            nodal_load: Consistent nodal force vector, shape (N, 2).

        Returns:
            tuple: ``(Pi, E_int, W_ext, F, strain)``. The first three are
            scalar tensors; ``F`` and ``strain`` have shape (E, 2, 2).
        """
        cur_coords = ref_coords + displacements
        E_int, F, strain = self.compute_internal_energy(
            ref_coords, cur_coords, elements, element_areas)
        W_ext = self.compute_external_work(displacements, nodal_load)
        Pi    = E_int - W_ext
        return Pi, E_int, W_ext, F, strain

    # ── Cauchy stress ─────────────────────────────────────────

    def cauchy_stress_LE(self, strain, emu, elam):
        """sigma = lambda_ps tr(eps) I + 2 mu eps  (E,2,2)."""
        tr_e = strain[:, 0, 0] + strain[:, 1, 1]
        I    = torch.eye(2, dtype=strain.dtype, device=strain.device
                         ).unsqueeze(0).expand(len(strain), -1, -1)
        return elam[:, None, None] * tr_e[:, None, None] * I \
            + 2.0 * emu[:, None, None] * strain

    def cauchy_stress_NH(self, F, emu, elam):
        """
        Compressible Neo-Hookean Cauchy stress under plane stress:

            sigma = (1/J) ( mu (B - I) + lambda ln J I ),   B = F F^T

        evaluated on the in-plane 2x2 block. sigma_33 = 0 holds by construction
        because F33 is recovered from the same Newton root as in the energy.
        """
        J2D = F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
        J2D = torch.clamp(J2D, min=1e-8)
        lnJ2D = torch.log(J2D)

        x = torch.ones_like(J2D)
        for _ in range(8):
            g  = emu * x * x + elam * torch.log(x) - emu + elam * lnJ2D
            dg = 2.0 * emu * x + elam / x
            x  = x - g / dg
            x  = torch.clamp(x, min=1e-8)
        F33 = x
        J   = J2D * F33

        B  = torch.matmul(F, F.transpose(-2, -1))    # (E,2,2)
        I2 = torch.eye(2, dtype=F.dtype, device=F.device
                       ).unsqueeze(0).expand(len(F), -1, -1)
        lnJ = torch.log(J)
        return (1.0 / J[:, None, None]) * (
            emu[:, None, None] * (B - I2) + elam[:, None, None] * lnJ[:, None, None] * I2)

    @staticmethod
    def von_mises(sxx, syy, sxy):
        """
        Plane-stress von Mises invariant:

            sigma_vm = sqrt( sxx^2 - sxx syy + syy^2 + 3 sxy^2 )

        This is the closed form of the full 3-D deviator with szz = 0. A 2-D
        deviator built on Identity(2) would drop the out-of-plane term
        szz_dev = -tr(sigma)/3 and under-report by up to sqrt(5/6) ~= 0.913 for
        a uniaxial state, so both the FEM and the GNN use THIS form.
        """
        return torch.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)

    def stress_tensor(self, F, strain, emu, elam):
        """Full 2x2 in-plane Cauchy stress for the active material model."""
        if self.material_model == 'NH':
            return self.cauchy_stress_NH(F, emu, elam)
        return self.cauchy_stress_LE(strain, emu, elam)

    def stress_components(self, F, strain, emu, elam):
        """(sxx, syy, sxy, von_mises) for the active model."""
        sigma = self.stress_tensor(F, strain, emu, elam)
        sxx = sigma[:, 0, 0]
        syy = sigma[:, 1, 1]
        sxy = sigma[:, 0, 1]
        return sxx, syy, sxy, self.von_mises(sxx, syy, sxy)


# ============================================================
# DERIVED PLATE DIAGNOSTICS
# ============================================================

def stress_concentration_factor(sigma_xx, traction):
    """
    Kt = max(sigma_xx) / T — the headline output of the plate-with-hole problem.

    For a small circular hole in a wide plate under remote uniaxial tension the
    Kirsch solution gives Kt = 3 exactly, approached from below as the hole
    shrinks relative to the plate. It is dimensionless and geometry-driven, so
    it is the single best check that the predicted field has the right SHAPE
    around the rim rather than merely the right magnitude far from it.

    Args:
        sigma_xx: Per-node or per-element sigma_xx, any array-like.
        traction: The applied remote traction T (same stress unit).

    Returns:
        float: The concentration factor, or NaN if the traction is zero.
    """
    T = float(traction)
    if abs(T) < 1e-30:
        return float('nan')
    return float(np.max(np.asarray(sigma_xx, dtype=np.float64)) / T)


# ============================================================
# GNN ARCHITECTURE
# ============================================================

# Width of the geometric edge feature vector: [dx, dy, |d|].
EDGE_ATTR_DIM = 3


class GNNLayer(MessagePassing):
    """
    Single GNN layer with message passing.

    The message is built from the receiver state x_i, the *difference*
    x_j - x_i, and the geometric edge attribute [dx, dy, |d|] (the
    reference-configuration offset of the sender from the receiver, and its
    length).

    Passing x_j - x_i together with the edge vector — rather than the raw x_j —
    is what lets a layer represent a finite-difference stencil: a directional
    derivative needs both the state difference and the physical separation it is
    taken over. With only [x_i, x_j] the layer can do little more than
    mean-smooth its neighbourhood, so stacking depth blurs the field instead of
    widening the effective stencil — which is exactly the wrong behaviour at a
    hole rim, where the whole point is to resolve a steep gradient.
    """

    def __init__(self, in_dim, out_dim, edge_dim=EDGE_ATTR_DIM):
        """Build the message and update multilayer perceptrons.

        Args:
            in_dim: Width of the incoming node features.
            out_dim: Width of the produced node features.
            edge_dim: Width of the geometric edge attribute, 3 for
                ``[dx, dy, |d|]``.
        """
        super(GNNLayer, self).__init__(aggr='mean')
        self.edge_dim    = int(edge_dim)
        self.lin_message = nn.Linear(2 * in_dim + self.edge_dim, out_dim)
        self.lin_update  = nn.Linear(in_dim + out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """Run one round of message passing.

        Args:
            x: Node features, shape (N, in_dim).
            edge_index: Graph connectivity, shape (2, E), rows ``(dst, src)``.
            edge_attr: Geometric edge features, shape (E, edge_dim). Zeros are
                substituted when omitted.

        Returns:
            Tensor: Updated node features, shape (N, out_dim).
        """
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.shape[1], self.edge_dim))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Form the message sent along each edge.

        Args:
            x_i: Receiver node features, shape (E, in_dim).
            x_j: Sender node features, shape (E, in_dim).
            edge_attr: Geometric edge features, shape (E, edge_dim).

        Returns:
            Tensor: Per-edge messages, shape (E, out_dim).
        """
        return F.relu(self.lin_message(
            torch.cat([x_i, x_j - x_i, edge_attr], dim=-1)))

    def update(self, aggr_out, x):
        """Combine the aggregated messages with the previous node state.

        Args:
            aggr_out: Mean-aggregated messages, shape (N, out_dim).
            x: Previous node features, shape (N, in_dim).

        Returns:
            Tensor: Updated node features, shape (N, out_dim).
        """
        return F.relu(self.lin_update(torch.cat([x, aggr_out], dim=-1)))


class DisplacementGNN(nn.Module):
    """Graph Neural Network predicting the 2-component displacement field."""

    def __init__(self, input_dim, hidden_dim, num_layers, output_scale=1.0):
        """Assemble the input embedding, message-passing stack and output head.

        Args:
            input_dim: Width of the per-node input features (normalised
                coordinates plus material one-hot).
            hidden_dim: Width of every hidden node-feature vector.
            num_layers: Number of message-passing layers.
            output_scale: Fixed, non-trainable multiplier applied to the
                predicted displacement (see ``Config.OUTPUT_DISPLACEMENT_SCALE``).
                It is not part of ``state_dict`` and must be restored from the
                checkpoint at inference.
        """
        super(DisplacementGNN, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Fixed (non-trainable) multiplier on the predicted displacement — see
        # Config.OUTPUT_DISPLACEMENT_SCALE. Not a parameter and not part of
        # state_dict, so it must be restored from the checkpoint at inference.
        self.output_scale = float(output_scale)

        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.gnn_layers  = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, 2)

        print("\n" + "="*60)
        print("GNN ARCHITECTURE")
        print("="*60)
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Number of GNN layers: {num_layers}")
        print(f"Output dimension: 2 (ux, uy displacements)")
        print(f"Output displacement scale: {self.output_scale:.6g}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, x, edge_index, edge_attr=None):
        """Predict the nodal displacement field.

        Args:
            x: Node features, shape (N, input_dim).
            edge_index: Graph connectivity, shape (2, E).
            edge_attr: Geometric edge features, shape (E, 3).

        Returns:
            Tensor: Displacements ``(ux, uy)``, shape (N, 2), already multiplied
            by ``output_scale``. Constrained nodes are NOT zeroed here — that is
            applied separately by :meth:`BoundaryConditions.apply`.
        """
        h = F.relu(self.input_embed(x))
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)
        return self.output_layer(h) * self.output_scale


# ============================================================
# DATA LOADING
# ============================================================

def load_mesh_summary(data_dir):
    """Read the plate geometry the meshing stage recorded.

    The plate size sets the characteristic length that
    ``OUTPUT_DISPLACEMENT_SCALE`` is derived from, so it must come from the mesh
    that was actually built rather than a default.
    """
    path = Path(data_dir) / 'mesh_summary.json'
    if not path.exists():
        raise FileNotFoundError(
            f"mesh_summary.json not found in {data_dir}. It carries the plate "
            f"size the output displacement scale is derived from — re-run the "
            f"meshing stage.")
    with open(path) as f:
        return json.load(f)


def load_mesh_data(data_dir):
    """Load every mesh array the training stage needs.

    Args:
        data_dir: Directory written by the meshing stage.

    Returns:
        tuple: ``(nodes, elements, features, loading_facets, topology,
        element_material_ids)`` with shapes (N, 2), (E, 3), (N, F), (L, 3),
        (N, max_degree) and (E,) respectively. ``topology`` is padded with -1
        where a node has fewer neighbours than the maximum degree, and
        ``loading_facets`` rows are ``[elem_id, n1, n2]``.
    """
    print("\n" + "="*60)
    print("LOADING HOLE-PLATE MESH DATA")
    print("="*60)

    data_dir = Path(data_dir)
    nodes                = np.load(data_dir / 'nodes.npy')
    elements             = np.load(data_dir / 'elements.npy')
    features             = np.load(data_dir / 'node_features.npy')
    loading_facets       = np.load(data_dir / 'loading_surface_facets.npy')
    topology             = np.load(data_dir / 'node_topology.npy')
    element_material_ids = np.load(data_dir / 'element_material_ids.npy')

    print(f"Loaded hole-plate mesh with:")
    print(f"  Nodes: {len(nodes)}  (2-D)")
    print(f"  Triangles: {len(elements)}  (all matrix; voids removed at mesh time)")
    print(f"  Loading facets (right-edge segments): {len(loading_facets)}")

    return nodes, elements, features, loading_facets, topology, element_material_ids


def build_graph_data(nodes, elements, features, topology):
    """Assemble the PyTorch Geometric graph the GNN consumes.

    Node coordinates are min-max normalised to the unit square so the input
    features are O(1); the geometric edge attributes are built from the same
    normalised coordinates, which keeps them mesh-size-consistent.

    Args:
        nodes: Reference nodal coordinates, shape (N, 2), in mm.
        elements: Triangle connectivity, shape (E, 3). Accepted for signature
            symmetry with the other loaders; the graph is built from
            ``topology``.
        features: Node feature matrix, shape (N, F); columns 2:4 are the
            material one-hot and columns 4:6 the raw coordinates.
        topology: Neighbour lists padded with -1, shape (N, max_degree).

    Returns:
        Data: Graph with ``x`` of shape (N, 4), ``edge_index`` of shape (2, E)
        and ``edge_attr`` of shape (E, 3) holding ``[dx, dy, |d|]``.
    """
    print("\nBuilding graph structure...")

    xy     = features[:, 4:6].astype(np.float64)
    mat_id = features[:, 2:4].astype(np.float64)
    xy_min = xy.min(axis=0)
    xy_range = xy.max(axis=0) - xy_min
    xy_range[xy_range < 1e-10] = 1.0
    xy_norm = (xy - xy_min) / xy_range

    node_features = torch.tensor(np.concatenate([xy_norm, mat_id], axis=1),
                                 dtype=torch.float64)
    print(f"  Coords normalised to [0,1]^2  -> input dim {node_features.shape[1]}")

    # Vectorised edge list from the padded topology array (-1 = no neighbour).
    valid = topology >= 0
    src_nodes = np.repeat(np.arange(len(topology)), valid.sum(axis=1))
    dst_nodes = topology[valid]
    edge_index = torch.tensor(np.stack([src_nodes, dst_nodes]), dtype=torch.long)

    print(f"  Graph edges: {edge_index.shape[1]}")
    print(f"  Average node degree: {edge_index.shape[1] / len(nodes):.2f}")

    # Geometric edge attributes [dx, dy, |d|] in the *reference* configuration,
    # from the same normalised coordinates the network sees as input, so the
    # feature is O(1) and mesh-size-consistent. Row k corresponds to
    # edge_index[:, k] = (dst, src): the offset of the sender from the receiver.
    dst, src = edge_index[0], edge_index[1]
    xy_t = torch.tensor(xy_norm, dtype=torch.float64)
    d_vec = xy_t[src] - xy_t[dst]                        # (E, 2)
    d_len = d_vec.norm(dim=1, keepdim=True)              # (E, 1)
    edge_attr = torch.cat([d_vec, d_len], dim=1)         # (E, 3)
    print(f"  Edge attributes: {tuple(edge_attr.shape)} [dx, dy, |d|] "
          f"(mean |d| = {d_len.mean().item():.5f} in normalised units)")

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================
# TRAINING
# ============================================================

class Trainer:
    """Training loop for the physics-informed GNN on the hole plate."""

    def __init__(self, model, energy_calculator, bc_handler, graph_data,
                 ref_coords, elements, element_areas, nodal_load,
                 nodes_np, elements_np, element_material_ids, config,
                 mesh_info=None):
        """Wire up the optimizer, logging and the plotting triangulation.

        Args:
            model: The ``DisplacementGNN`` to train.
            energy_calculator: ``CompositeEnergyCalculator`` supplying Pi(u).
            bc_handler: ``BoundaryConditions`` imposing the roller and pin.
            graph_data: PyTorch Geometric graph holding ``x``, ``edge_index``
                and ``edge_attr``.
            ref_coords: Reference nodal coordinates as a tensor, shape (N, 2).
            elements: Triangle connectivity as a tensor, shape (E, 3).
            element_areas: Reference element areas as a tensor, shape (E,).
            nodal_load: Consistent nodal force vector as a tensor, shape (N, 2).
            nodes_np: Reference nodal coordinates as a NumPy array, shape
                (N, 2), used by the plotting and Kt diagnostics.
            elements_np: Triangle connectivity as a NumPy array, shape (E, 3).
            element_material_ids: Per-element material tag, shape (E,).
            config: Config-like object holding the hyperparameters and paths.
            mesh_info: Parsed ``mesh_summary.json``; supplies the plate size and
                hole geometry for the report headers.
        """
        # `self.model` is the (possibly torch.compile-wrapped) model used for
        # the hot training forward/backward. `self._orig_model` is the original
        # eager module — same parameters, clean state_dict — used for
        # checkpointing and eval/plot passes.
        self.model, self._orig_model = maybe_compile_model(model, config)
        self.energy_calc  = energy_calculator
        self.bc           = bc_handler
        self.graph_data   = graph_data
        self.ref_coords   = ref_coords
        self.elements     = elements
        self.element_areas = element_areas
        self.nodal_load   = nodal_load
        self.nodes_np     = nodes_np
        self.elements_np  = elements_np
        self.element_material_ids = element_material_ids
        self.config       = config
        self.mesh_info    = mesh_info or {}

        self.optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, eps=1e-12)

        _HISTORY_CAP = 5000
        self._HISTORY_CAP = _HISTORY_CAP
        self.history = {'epoch': [], 'loss': [], 'E_int': [], 'W_ext': [],
                        'max_disp': [], 'kt': []}

        # ── Best-model tracking ────────────────────────────────────────────
        # The loss is the total potential energy Pi. The minimum principle says
        # Pi(u) >= Pi(u*) for every admissible u, so the LOWEST Pi seen is the
        # best approximation on the trajectory — argmin is the principled
        # selection rule, not a heuristic. Adam does not descend monotonically,
        # so taking the final epoch samples an arbitrary point in its
        # oscillation band, which is avoidable noise.
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.best_state = None

        tb_dir = Path(config.CHECKPOINT_DIR) / 'runs'
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        # Single matrix phase — one triangulation over all elements.
        self._triang_ref = mtri.Triangulation(nodes_np[:, 0], nodes_np[:, 1], elements_np)

        print("\n" + "="*60)
        print("TRAINER INITIALIZED")
        print("="*60)
        print(f"Optimizer: Adam (lr={config.LEARNING_RATE})")
        print(f"Material model: {config.MATERIAL_MODEL}")
        print(f"Traction: {config.TRACTION_MAGNITUDE} {config.POSTPROC_OUTPUT_UNIT}")
        print(f"TensorBoard logs: {tb_dir}")

    def _make_input(self):
        """Return the node-feature matrix fed to the model, shape (N, input_dim)."""
        return self.graph_data.x

    def train_step(self):
        """Take one full-batch gradient step on the total potential energy.

        Returns:
            tuple: ``(loss, E_int, W_ext, max_disp)`` as Python floats, where
            ``loss`` is Pi and ``max_disp`` is the largest absolute
            displacement component in mm.
        """
        self.model.train()
        self.optimizer.zero_grad()

        x_input           = self._make_input()
        displacements_raw = self.model(x_input, self.graph_data.edge_index,
                                       getattr(self.graph_data, 'edge_attr', None))
        displacements     = self.bc.apply(displacements_raw)

        Pi, E_int, W_ext, _F, _strain = self.energy_calc.compute_total_potential_energy(
            self.ref_coords, displacements, self.elements, self.element_areas,
            self.nodal_load
        )

        Pi.backward()
        self.optimizer.step()

        loss_val  = Pi.item()
        E_int_val = E_int.item()
        W_ext_val = W_ext.item()
        max_disp  = torch.max(torch.abs(displacements)).item()

        del _F, _strain, displacements, displacements_raw, x_input, Pi, E_int, W_ext
        return loss_val, E_int_val, W_ext_val, max_disp

    def _current_kt(self):
        """Peak sigma_xx over the applied traction — the headline output."""
        res = self._eval()
        return stress_concentration_factor(res['sigma_xx'],
                                           self.config.TRACTION_MAGNITUDE)

    def train(self):
        """Minimise the total potential energy for ``config.NUM_EPOCHS`` epochs.

        Logs to TensorBoard, writes periodic checkpoints and field plots, and
        tracks the lowest-Pi weights seen. Because the minimum principle gives
        Pi(u) >= Pi(u*) for every admissible u, the argmin over the trajectory
        is the principled model-selection rule, and it is written to
        ``model_best.pt`` for the downstream FEM comparison.

        Side effects:
            Sets ``self.total_training_time`` (seconds of optimisation compute,
            excluding checkpoint and plot I/O) and writes ``training_time.json``
            into the checkpoint directory.
        """
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)

        start_time = time.time()
        # Accumulates wall-clock time spent WRITING output (checkpoints + field
        # plots) inside the loop, so it can be subtracted from the reported
        # training time — that number must reflect optimisation compute only.
        self._checkpoint_io_time = 0.0

        for epoch in range(self.config.NUM_EPOCHS):
            loss, E_int, W_ext, max_disp = self.train_step()

            cap = self._HISTORY_CAP
            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss)

            # NaN never compares less-than, so a diverged step cannot become
            # "best" and poison the selection.
            if loss < self.best_loss:
                self.best_loss = float(loss)
                self.best_epoch = epoch + 1
                self.best_state = {k: v.detach().clone()
                                   for k, v in self._orig_model.state_dict().items()}
            self.history['E_int'].append(E_int)
            self.history['W_ext'].append(W_ext)
            self.history['max_disp'].append(max_disp)
            if len(self.history['epoch']) > cap:
                for v in self.history.values():
                    if len(v) > cap:
                        del v[0]

            self.writer.add_scalar('Loss/potential_energy', loss, epoch)
            self.writer.add_scalar('Energy/internal',       E_int, epoch)
            self.writer.add_scalar('Energy/external_work',  W_ext, epoch)
            self.writer.add_scalar('Displacement/max',      max_disp, epoch)

            if (epoch + 1) % self.config.PRINT_EVERY == 0:
                self.writer.flush()
                elapsed = time.time() - start_time - self._checkpoint_io_time
                kt = self._current_kt()
                self.history['kt'].append(kt)
                self.writer.add_scalar('Stress/Kt', kt, epoch)
                # For LE at the exact minimum of Pi, 2*E_int/W_ext -> 1 exactly.
                # It is dimensionless and config-independent, so it is the single
                # best health check. For NH it is not identically 1 but stays
                # O(1); a value far from 1, a POSITIVE Pi, or a NEGATIVE W_ext
                # all mean the optimisation has not converged.
                ratio = (2.0 * E_int / W_ext) if abs(W_ext) > 1e-30 else float('nan')
                print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} ({elapsed:.1f}s compute)")
                print(f"  Loss (Pi): {loss:.6e}")
                print(f"  E_internal: {E_int:.6e}   W_external: {W_ext:.6e}")
                print(f"  2*E_int/W_ext: {ratio:.4f}   (-> 1.0 at the LE minimum)")
                print(f"  Max |displacement|: {max_disp:.6e}")
                print(f"  Stress concentration Kt = max(sxx)/T: {kt:.3f}   "
                      f"(-> ~3.0 for a small circular hole)")

            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                _io_t0 = time.time()
                self.save_checkpoint(epoch + 1)
                self._checkpoint_io_time += time.time() - _io_t0

        # Always save a final checkpoint so runs whose epoch count is not a
        # multiple of SAVE_EVERY (e.g. short test runs) still produce a model.
        if self.config.NUM_EPOCHS % self.config.SAVE_EVERY != 0:
            _io_t0 = time.time()
            self.save_checkpoint(self.config.NUM_EPOCHS)
            self._checkpoint_io_time += time.time() - _io_t0

        # Unconditional final flush: SAVE_EVERY may have written model_best.pt
        # part-way through, but the best epoch can be anywhere after that.
        _io_t0 = time.time()
        _best_path = self.save_best_checkpoint()
        self._checkpoint_io_time += time.time() - _io_t0
        if _best_path is not None:
            _final = self.history['loss'][-1] if self.history['loss'] else float('nan')
            _gap = ((_final - self.best_loss) / abs(self.best_loss) * 100.0
                    if self.best_loss not in (0.0,) and self.best_loss == self.best_loss
                    else float('nan'))
            print(f"\n  Best model : epoch {self.best_epoch}  "
                  f"(Pi = {self.best_loss:.8e})")
            print(f"  Final epoch: {self.config.NUM_EPOCHS}  (Pi = {_final:.8e})  "
                  f"-> final is {_gap:+.4f}% vs best")
            print(f"  Saved      : {_best_path}   [Stage 3 uses this]")

        wall_time = time.time() - start_time
        self.total_training_time = wall_time - self._checkpoint_io_time
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED in {self.total_training_time:.2f}s compute "
              f"(wall {wall_time:.2f}s - {self._checkpoint_io_time:.2f}s checkpoint/plot I/O)")
        print(f"{'='*60}")
        self.writer.close()

        tt_path = Path(self.config.CHECKPOINT_DIR) / 'training_time.json'
        try:
            with open(tt_path, 'w') as _f:
                json.dump({'total_training_time_s': self.total_training_time,
                           'wall_time_s': wall_time,
                           'checkpoint_io_time_s': self._checkpoint_io_time,
                           'epochs': self.config.NUM_EPOCHS}, _f)
        except Exception:
            pass

    def _checkpoint_payload(self, epoch, state_dict, selection, loss=None):
        """
        Build the checkpoint dict. Shared by the periodic and best-model saves so
        the two can never drift apart in what metadata they record — Stage 3
        rebuilds the model from these keys.
        """
        return {
            'epoch':               epoch,
            # Save from the uncompiled module so keys have no `_orig_mod.`
            # prefix and load cleanly into a plain DisplacementGNN (Stage 3).
            'model_state_dict':    state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'material_model':      self.config.MATERIAL_MODEL,
            'youngs_modulus_matrix': float(self.config.YOUNGS_MODULUS_MATRIX),
            'poissons_ratio_matrix': float(self.config.POISSONS_RATIO_MATRIX),
            # Needed to rebuild an identical model at inference time; the scale
            # is not a parameter, so it does NOT live in model_state_dict, and
            # rebuilding without it would silently rescale the entire field.
            'output_displacement_scale': self._orig_model.output_scale,
            # Architecture, so Stage 3 can rebuild the right shape in a fresh
            # process rather than falling back to the live Config defaults.
            'input_dim':   int(self._orig_model.input_dim),
            'hidden_dim':  int(self._orig_model.hidden_dim),
            'num_layers':  int(self._orig_model.num_layers),
            'traction_magnitude':  float(self.config.TRACTION_MAGNITUDE),
            'output_unit': str(self.config.POSTPROC_OUTPUT_UNIT),
            # How this checkpoint was chosen: 'best' (lowest Pi over the whole
            # trajectory) or 'last' (the weights at that epoch).
            'selection':           selection,
            'loss':                (float(loss) if loss is not None else None),
        }

    def save_checkpoint(self, epoch):
        """Write the epoch checkpoint, refresh ``model_best.pt`` and plot fields.

        Args:
            epoch: 1-based epoch number, used in the checkpoint filename.
        """
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        path = checkpoint_dir / f'model_epoch_{epoch}.pt'
        loss = self.history['loss'][-1] if self.history['loss'] else None
        torch.save(self._checkpoint_payload(
            epoch, self._orig_model.state_dict(), 'last', loss), path)
        print(f"  Checkpoint saved: {path}")
        self.save_best_checkpoint()
        self._save_field_plots(epoch, checkpoint_dir)

    def save_best_checkpoint(self):
        """
        Write the lowest-Pi weights seen so far to `model_best.pt`.

        Stage 3 prefers this file over the numbered epoch checkpoints, so the
        FEM comparison evaluates the best model on the trajectory rather than
        whichever epoch training happened to stop on.
        """
        if self.best_state is None:
            return None
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        path = checkpoint_dir / 'model_best.pt'
        torch.save(self._checkpoint_payload(
            self.best_epoch, self.best_state, 'best', self.best_loss), path)
        return path

    def _eval(self):
        """Inference: nodal displacement + element-centred stress arrays."""
        with torch.no_grad():
            x_input           = self._make_input()
            # Eager module (shared weights) for inference — no extra compile.
            displacements_raw = self._orig_model(
                x_input, self.graph_data.edge_index,
                getattr(self.graph_data, 'edge_attr', None))
            displacements     = self.bc.apply(displacements_raw)

            cur_coords = self.ref_coords + displacements
            F_def      = self.energy_calc.compute_deformation_gradient(
                             self.ref_coords, cur_coords, self.elements)
            strain     = self.energy_calc.compute_strain_tensor(F_def)

            dev       = F_def.device
            emu, elam = self.energy_calc._get_cached_lame(dev)

            sigma = self.energy_calc.stress_tensor(F_def, strain, emu, elam)
            sxx, syy, sxy, vm = self.energy_calc.stress_components(
                F_def, strain, emu, elam)
            J = F_def[:, 0, 0] * F_def[:, 1, 1] - F_def[:, 0, 1] * F_def[:, 1, 0]

            out = dict(
                disp=displacements.cpu().numpy(),
                sigma=sigma.cpu().numpy(),
                sigma_xx=sxx.cpu().numpy(),
                sigma_yy=syy.cpu().numpy(),
                sigma_xy=sxy.cpu().numpy(),
                von_mises=vm.cpu().numpy(),
                J=J.cpu().numpy(),
            )
            del x_input, displacements_raw, displacements, cur_coords
            del F_def, strain, sigma, sxx, syy, sxy, vm, J
        return out

    def _save_field_plots(self, epoch, save_dir):
        """Displacement and stress fields on the reference mesh."""
        save_dir = Path(save_dir)
        self._orig_model.eval()

        res = self._eval()
        disp = res['disp']
        unit = self.config.POSTPROC_OUTPUT_UNIT
        nodes, elements = self.nodes_np, self.elements_np
        triang_ref = self._triang_ref

        kt = stress_concentration_factor(res['sigma_xx'],
                                         self.config.TRACTION_MAGNITUDE)

        # ── Displacement field ──
        deformed_nodes = nodes + disp * 100
        triang_def = mtri.Triangulation(deformed_nodes[:, 0], deformed_nodes[:, 1],
                                        elements)

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            f'Displacement Field — Epoch {epoch}  [{self.config.MATERIAL_MODEL}]  '
            f'E_mat={self.energy_calc.E_matrix:.4g} {unit},  '
            f'T={self.config.TRACTION_MAGNITUDE:.4g} {unit}',
            fontsize=13, fontweight='bold')

        ax = axes[0, 0]
        ax.triplot(triang_ref, 'b-', linewidth=0.3, alpha=0.3)
        ax.set_title('Original Mesh (matrix, voids removed)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.triplot(triang_def, 'b-', linewidth=0.3, alpha=0.5)
        ax.set_title('Deformed Mesh (100x)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        for ax, comp, label in [(axes[1, 0], disp[:, 0], 'ux (mm)'),
                                (axes[1, 1], disp[:, 1], 'uy (mm)')]:
            sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=comp, cmap='jet', s=20)
            plt.colorbar(sc, ax=ax, label=label)
            ax.set_title(label); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'displacement_field_epoch_{epoch}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Stress field ──
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            f'Stress Field — Epoch {epoch}  [{self.config.MATERIAL_MODEL}]  '
            f'Kt = max(sxx)/T = {kt:.3f}',
            fontsize=13, fontweight='bold')
        for ax, values, title in zip(axes.flat,
                                     [res['sigma_xx'], res['sigma_yy'],
                                      res['sigma_xy'], res['von_mises']],
                                     ['sigma_xx', 'sigma_yy', 'sigma_xy',
                                      'von Mises']):
            tc = ax.tripcolor(triang_ref, facecolors=values, cmap='jet',
                              shading='flat')
            plt.colorbar(tc, ax=ax, label=unit)
            ax.set_title(title); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'stress_field_epoch_{epoch}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        del res, disp, deformed_nodes, triang_def

        plt.close('all')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Field plots saved for epoch {epoch}")


# ============================================================
# VISUALIZATION
# ============================================================

def _make_model_input(graph_data):
    """Extract the node-feature matrix the model takes as input.

    Args:
        graph_data: The PyTorch Geometric graph.

    Returns:
        Tensor: Node features, shape (N, input_dim).
    """
    return graph_data.x


def visualize_results(model, graph_data, bc_handler, nodes, elements,
                      element_material_ids, history, config, mesh_info=None):
    """Deformed-plate view + training history."""
    print("\n" + "="*60)
    print(f"GENERATING VISUALIZATIONS")
    print("="*60)

    model.eval()
    with torch.no_grad():
        disp = bc_handler.apply(model(
            _make_model_input(graph_data), graph_data.edge_index,
            getattr(graph_data, 'edge_attr', None))).cpu().numpy()

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)

    deformed_nodes = nodes + disp * 100
    triang_mat = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    triang_def = mtri.Triangulation(deformed_nodes[:, 0], deformed_nodes[:, 1],
                                    elements)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax = axes[0, 0]
    ax.triplot(triang_mat, 'b-', linewidth=0.3, alpha=0.3, label='Matrix')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Original Mesh [{config.MATERIAL_MODEL}]', fontsize=14,
                 fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.triplot(triang_def, 'b-', linewidth=0.3, alpha=0.5)
    ax.set_title('Deformed Mesh (100x)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    for ax, comp, label in [(axes[1, 0], disp[:, 0], 'ux (mm)'),
                            (axes[1, 1], disp[:, 1], 'uy (mm)')]:
        sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=comp, cmap='jet', s=20)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'displacement_field.png', dpi=300, bbox_inches='tight')
    print("  Saved: displacement_field.png")
    plt.close(fig)
    del triang_mat, triang_def, deformed_nodes

    # ── Training history ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    _L = (mesh_info or {}).get('plate_size')
    fig.suptitle(
        f"Training history [{config.MATERIAL_MODEL}]"
        + (f"  —  plate {_L:g} mm, {len((mesh_info or {}).get('holes') or [])} hole(s)"
           if _L else ""),
        fontsize=13, fontweight='bold')
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (Potential Energy)')
    ax.set_title('Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_yscale('symlog')

    ax = axes[0, 1]
    ax.plot(history['epoch'], history['E_int'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Internal Energy')
    ax.set_title('Strain Energy', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_yscale('log')

    ax = axes[1, 0]
    ax.plot(history['epoch'], history['W_ext'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('External Work')
    ax.set_title('External Work (right-edge traction)', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_yscale('log')

    ax = axes[1, 1]
    if history['kt']:
        xs = np.linspace(0, history['epoch'][-1] if history['epoch'] else 0,
                         len(history['kt']))
        ax.plot(xs, history['kt'], 'm-o', markersize=3, linewidth=2)
        ax.axhline(3.0, color='k', linestyle='--', linewidth=1,
                   label='Kirsch Kt = 3')
        ax.legend(fontsize=8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Kt = max(sigma_xx) / T')
    ax.set_title('Stress concentration factor', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print("  Saved: training_history.png")
    plt.close(fig)
    gc.collect()

    print(f"  Max |ux| : {np.abs(disp[:, 0]).max():.6e} mm")
    print(f"  Max |uy| : {np.abs(disp[:, 1]).max():.6e} mm")


def visualize_stress_field(model, graph_data, bc_handler, nodes, elements,
                           element_material_ids, energy_calc, ref_coords,
                           elements_tensor, element_areas_tensor, config):
    """Element-centred stress fields on the reference mesh."""
    print("\n" + "="*60)
    print(f"GENERATING STRESS FIELD VISUALIZATIONS")
    print("="*60)

    model.eval()
    with torch.no_grad():
        displacements = bc_handler.apply(model(
            _make_model_input(graph_data), graph_data.edge_index,
            getattr(graph_data, 'edge_attr', None)))

        cur_coords = ref_coords + displacements
        F_def      = energy_calc.compute_deformation_gradient(
                         ref_coords, cur_coords, elements_tensor)
        strain     = energy_calc.compute_strain_tensor(F_def)
        emu, elam  = energy_calc._get_cached_lame(F_def.device)
        sxx, syy, sxy, vm = energy_calc.stress_components(F_def, strain, emu, elam)

    sigma_xx_np  = sxx.cpu().numpy()
    sigma_yy_np  = syy.cpu().numpy()
    sigma_xy_np  = sxy.cpu().numpy()
    von_mises_np = vm.cpu().numpy()

    unit  = config.POSTPROC_OUTPUT_UNIT
    kt    = stress_concentration_factor(sigma_xx_np, config.TRACTION_MAGNITUDE)
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f'Stress Field [{config.MATERIAL_MODEL}]  '
        f'(E_mat={energy_calc.E_matrix:.4g} {unit}, '
        f'T={config.TRACTION_MAGNITUDE:.4g} {unit})   '
        f'Kt = max(sxx)/T = {kt:.3f}',
        fontsize=14, fontweight='bold')
    for ax, values, title in [
        (axes[0, 0], sigma_xx_np,  'sigma_xx (normal X-stress)'),
        (axes[0, 1], sigma_yy_np,  'sigma_yy (normal Y-stress)'),
        (axes[1, 0], sigma_xy_np,  'sigma_xy (shear stress)'),
        (axes[1, 1], von_mises_np, 'von Mises'),
    ]:
        tc = ax.tripcolor(triang, facecolors=values, cmap='jet', shading='flat')
        plt.colorbar(tc, ax=ax, label=unit)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / 'stress_field.png', dpi=300, bbox_inches='tight')
    print("  Saved: stress_field.png")
    plt.close(fig)
    gc.collect()

    print(f"  Stress concentration Kt : {kt:.4f}   (-> ~3.0 for a small circular hole)")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the full training stage end to end.

    Loads the mesh, assembles the consistent nodal load of the right-edge
    traction, builds the graph, energy calculator and model, trains to
    convergence and writes checkpoints plus summary figures. Paths and
    hyperparameters are read from the module-level ``Config``.
    """
    print("="*60)
    print("PHYSICS-INFORMED GNN — 2-D PLATE WITH A CENTRAL HOLE")
    print("="*60)
    print(f"Device:         {Config.DEVICE}")
    print(f"Material model: {Config.MATERIAL_MODEL}")

    setup_cpu_limits(Config)

    mesh_info = load_mesh_summary(Config.DATA_DIR)
    nodes, elements, features, loading_facets, topology, element_material_ids = \
        load_mesh_data(Config.DATA_DIR)

    L = mesh_info.get('plate_size')
    if not L:
        L = float(max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1])))
    # Recorded on Config so effective_output_displacement_scale can size the
    # characteristic displacement from the actual plate, not a guess.
    Config.PLATE_SIZE = L
    print(f"\nPlate geometry: L={L}, "
          f"holes={len(mesh_info.get('holes', []) or [])}")

    element_areas = MeshGeometry.compute_element_areas(nodes, elements)
    _ = MeshGeometry.compute_edge_lengths(nodes, loading_facets)
    nodal_load, F_x = MeshGeometry.compute_nodal_load(
        nodes, loading_facets, Config.TRACTION_MAGNITUDE)
    bc_handler = BoundaryConditions(nodes, features)

    ref_coords           = torch.tensor(nodes,         dtype=torch.float64, device=Config.DEVICE)
    elements_tensor      = torch.tensor(elements,      dtype=torch.long,    device=Config.DEVICE)
    element_areas_tensor = torch.tensor(element_areas, dtype=torch.float64, device=Config.DEVICE)
    nodal_load_t         = torch.tensor(nodal_load,    dtype=torch.float64, device=Config.DEVICE)

    graph_data = build_graph_data(nodes, elements, features, topology).to(Config.DEVICE)

    print("\n" + "="*60)
    print("TRAINING SETUP")
    print("="*60)
    E_mat = float(Config.YOUNGS_MODULUS_MATRIX)
    unit  = Config.POSTPROC_OUTPUT_UNIT
    print(f"  E_matrix         : {E_mat:.4g} {unit}")
    print(f"  nu_matrix        : {Config.POISSONS_RATIO_MATRIX}")
    print(f"  Applied traction : {Config.TRACTION_MAGNITUDE:.6g} {unit}")
    print(f"  Nominal strain   : {Config.TRACTION_MAGNITUDE / E_mat:.6e}  (T/E)")
    print(f"  Resultant force  : {F_x:.6g} {unit}*mm")

    energy_calc = CompositeEnergyCalculator(
        E_mat,
        Config.POISSONS_RATIO_MATRIX,
        element_material_ids,
        material_model=Config.MATERIAL_MODEL,
        output_unit=unit,
    )

    input_dim = graph_data.x.shape[1]
    model = DisplacementGNN(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        output_scale=effective_output_displacement_scale(Config, char_length=L),
    ).double().to(Config.DEVICE)

    trainer = Trainer(
        model, energy_calc, bc_handler, graph_data,
        ref_coords, elements_tensor, element_areas_tensor, nodal_load_t,
        nodes, elements, element_material_ids, Config, mesh_info=mesh_info
    )

    trainer.train()

    visualize_results(model, graph_data, bc_handler, nodes, elements,
                      element_material_ids, trainer.history, Config, mesh_info)
    visualize_stress_field(
        model, graph_data, bc_handler, nodes, elements, element_material_ids,
        energy_calc, ref_coords, elements_tensor, element_areas_tensor, Config
    )

    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
