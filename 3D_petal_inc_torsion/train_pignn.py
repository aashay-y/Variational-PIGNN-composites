"""
Physics-Informed Graph Neural Network for the 3-D Composite Rod in Torsion
==========================================================================
Trains a GNN to predict the displacement field of a circular composite rod
(matrix + straight prismatic inclusion) under an applied torque, by minimising
the total potential energy

        Pi(u) = E_int(u) - W_ext(u)  ->  min

over the space of kinematically admissible displacements.  This is the full
3-D problem: no plane-stress or plane-strain assumption is made anywhere, so
every node carries three displacement components and every element carries the
full 3x3 stress tensor.

Problem setup
-------------
  Geometry   circular rod, radius R, length H, meshed with linear tetrahedra;
             the inclusion is the petal cross-section swept through the depth.
  z = 0      CLAMPED, u = 0 (all three components) — hard-imposed by masking
  z = H      loaded by a prescribed TORQUE M_z about the rod axis, applied as
             the tangential dead traction
                 t(X) = (tau/R) * ( -(y-cy), (x-cx), 0 )
             whose resultant force is zero and whose resultant moment is M_z.
             The user specifies M_z; tau is solved for so the torque assembled
             over the discretised loaded face matches it exactly.
  lateral    free (traction-free cylindrical surface, natural BC)

The twist angle is an OUTPUT of the applied torque, not an input.

Material models
---------------
  'NH'  compressible Neo-Hookean, finite strain (default — torsion to ~50 %
        shear is far outside the linear range):
            W = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2
  'LE'  small-strain linear elasticity, retained as a reference option. It is
        NOT physically valid at this load level; the summary says so.
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
    # 'NH' = compressible Neo-Hookean, finite strain  (the physical choice here)
    # 'LE' = small-strain linear elasticity           (reference / debug only)
    MATERIAL_MODEL = 'NH'

    # Paths
    DATA_DIR        = 'mesh_rod_petal'
    CHECKPOINT_DIR  = 'rod_checkpoints'
    RESULTS_DIR     = 'rod_results'

    # ── Material properties ───────────────────────────────────
    # Soft-solid (hydrogel / tissue-like) properties, in POSTPROC_OUTPUT_UNIT.
    # Matrix:    E = 1.5 kPa, nu = 0.40
    # Inclusion: E = 5.0 kPa, nu = 0.35   ->  INCLUSION_RATIO = 5.0/1.5 = 3.3333
    # These are the ACTUAL physical values: unlike the 2-D plate version, there
    # is no reduced-modulus training trick — the network trains on the real
    # material and the loss is conditioned solely by OUTPUT_DISPLACEMENT_SCALE.
    YOUNGS_MODULUS_MATRIX    = 1.5
    POISSONS_RATIO_MATRIX    = 0.40
    POISSONS_RATIO_INCLUSION = 0.35

    # Material contrast: E_inclusion / E_matrix = 5.0 / 1.5.
    INCLUSION_RATIO         = 5.0 / 1.5
    INCLUSION_RATIO_DEFAULT = INCLUSION_RATIO   # backward-compatible alias

    # Unit that YOUNGS_MODULUS_MATRIX and TRACTION_MAGNITUDE are expressed in.
    # Mesh coordinates are in mm; the FEM converts both to SI internally.
    POSTPROC_OUTPUT_UNIT = 'kPa'

    # ── Loading: the APPLIED TORQUE about the rod axis ────────────────────────
    # This is the user-facing load parameter. Units are
    # POSTPROC_OUTPUT_UNIT * mm^3; with the default 'kPa' that is kPa*mm^3,
    # which equals exactly 1 micro-newton-metre:
    #     1 kPa*mm^3 = 1e3 Pa * 1e-9 m^3 = 1e-6 N*m = 1 uN*m
    #
    # The traction field that delivers it, t(X) = (tau/R)(-(y-cy), (x-cx), 0),
    # is DERIVED from this value at run time by ``traction_for_torque``, which
    # solves for tau such that the torque ASSEMBLED OVER THE DISCRETISED loaded
    # face equals APPLIED_TORQUE exactly. Scaling off the analytic tau*pi*R^3/2
    # instead would leave the realised torque a fraction of a percent low on a
    # polygonal rim, and mesh-dependent — two meshes would then be loaded
    # differently, which is exactly what a load parameter must not do.
    #
    # CALIBRATION. 1.1 uN*m was chosen so the rod reaches ~60 deg of twist. The
    # linear estimate needs the DEAD-LOAD correction: a traction fixed in the
    # reference frame delivers an effective torque M_z*cos(phi) once the face
    # has rotated by phi, so equilibrium solves
    #     (GJ/H) phi = M_z cos(phi)
    # rather than the textbook (GJ/H) phi = M_z. Ignoring the cosine
    # under-predicts the required torque by nearly a factor of two.
    # That same cosine bounds how large a twist a dead traction may be used
    # for. The load also carries a sin(phi) component that points radially
    # OUTWARD in the deformed frame, which balloons the free end. Both are
    # negligible while phi stays modest and take over once it does not:
    #
    #   M_z    twist    gamma   rim bulge at z=H   cos(phi)
    #   0.10   10.4 deg  0.091      +0.39 %          0.984
    #   0.19   19.1 deg  0.167      +1.35 %          0.945
    #   0.25   24.3 deg  0.212      +2.25 %          0.911   <- default
    #   0.31   29.3 deg  0.256      +3.30 %          0.872
    #   0.45   38.9 deg  0.340      +6.16 %          0.778
    #   1.10   62.7 deg  0.547     +21.20 %          0.458   <- unusable
    #
    # 0.25 is chosen as the largest load that is still unambiguously a torque.
    # Measured on the shipped mesh (NH): phi = 24.3 deg, rim shear
    # gamma = phi*R/H = 0.21 — 4-10x past the ~2-5 % linear-elastic range, so
    # the Neo-Hookean model is doing real work — while the dead-load artefact
    # stays at 2 % (the same regime as the cube-torsion case, which reached
    # 11 deg / +1.65 %) and 91 % of the applied moment still drives the twist.
    APPLIED_TORQUE = 0.25           # kPa*mm^3  ==  uN*m

    # Rim traction tau corresponding to APPLIED_TORQUE on the mesh actually in
    # use. NOT a user input: it is overwritten at run time (see main() /
    # run_pipeline.run_training) once the mesh is known. It is kept on
    # the config because the energy calculator, the output scaling and the
    # checkpoint all report it.
    TRACTION_MAGNITUDE = None

    # ── Output displacement scale (conditioning of the energy loss) ───────────
    # The GNN's raw output magnitude at initialisation is O(1e-2), set by weight
    # init and independent of the physics. When that disagrees with the true
    # displacement by orders of magnitude, the two terms of Pi = E_int - W_ext
    # are wildly unbalanced at init and the only meaningful gradient is
    # "shrink u", while W_ext — the term that creates the deformation SHAPE — is
    # numerically negligible. Training then collapses to a near-trivial field
    # regardless of epoch count.
    #
    # Multiplying the network output by a fixed characteristic displacement lets
    # the network learn O(1) values while the PHYSICAL strain is unchanged.
    #
    #   None  -> auto: (tau / E) * L_char, with tau the torque-derived rim
    #            traction and L_char the rod height taken from the mesh. A
    #            nominal strain times a length is a displacement — in torsion
    #            the rim displacement is ~gamma*H, so this lands within a factor
    #            of a few of the true magnitude.
    #   1.0   -> disabled
    #   float -> explicit
    # Results are insensitive to the exact value; only the ORDER OF MAGNITUDE
    # matters.
    OUTPUT_DISPLACEMENT_SCALE = None

    # ── GNN architecture ──────────────────────────────────────
    HIDDEN_DIM  = 64      # width of every hidden node-feature vector (channels)
    NUM_LAYERS  = 4       # message-passing steps; sets the receptive field in hops

    # ── Training ──────────────────────────────────────────────
    LEARNING_RATE = 1e-3  # Adam step size (dimensionless, on the scaled output)
    NUM_EPOCHS    = 30000 # full-batch energy-minimisation steps
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

    # Numerical stability: floor on a tetrahedron volume (mm^3). Prevents a
    # degenerate/sliver element from contributing a zero or negative weight to
    # the energy sum, which would make the loss non-differentiable there.
    MIN_VOLUME = 1e-16


def _normalized_material_model(config):
    """Read the material model off a config as an upper-case string.

    Args:
        config: Config-like object; a missing ``MATERIAL_MODEL`` defaults to
            Neo-Hookean.

    Returns:
        str: Either ``'NH'`` or ``'LE'``.
    """
    return str(getattr(config, 'MATERIAL_MODEL', 'NH')).upper()


def effective_output_displacement_scale(config, char_length=None):
    """Resolve OUTPUT_DISPLACEMENT_SCALE (see Config for the rationale).

    ``None`` -> (TRACTION_MAGNITUDE / E) * char_length, which puts the network's
    learned output at O(1) without perturbing the physical strain. ``char_length``
    defaults to the rod height recorded on the config (set from the mesh by
    ``main``); if that is unavailable it falls back to 1.0, which just makes the
    scale a nominal strain as in the 2-D version.
    """
    scale = getattr(config, 'OUTPUT_DISPLACEMENT_SCALE', None)
    if scale is None:
        E = float(config.YOUNGS_MODULUS_MATRIX)
        if E <= 0.0:
            raise ValueError("Young's modulus must be positive.")
        if char_length is None:
            char_length = float(getattr(config, 'ROD_HEIGHT', 1.0) or 1.0)
        tau = getattr(config, 'TRACTION_MAGNITUDE', None)
        if tau is None:
            raise ValueError(
                "TRACTION_MAGNITUDE is unset. It is derived from APPLIED_TORQUE "
                "once the mesh is known — call MeshGeometry.traction_for_torque "
                "and assign it before building the model.")
        return float(tau) / E * float(char_length)
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
    """Hard-imposed Dirichlet conditions for the clamped/twisted rod.

                 z = H  ── loaded face: tangential traction (torque), free DOFs
                        │
                        │      composite rod
                        │
                 z = 0  ── CLAMPED: ux = uy = uz = 0

    The clamped end face removes all six rigid-body modes, so no extra
    constraint is needed. The condition is imposed *hard* by zeroing the network
    output at every constrained DOF, so it holds exactly at every epoch — the
    network never has to learn it and no penalty term competes with the energy.

    The clamped node set comes from the Dirichlet flag in ``node_features``
    (column 0), which the mesh processor set from the z = 0 layer. Falling back
    to a coordinate query here would risk a different tolerance than the mesh
    processor used and silently free or over-constrain a ring of nodes.
    """

    def __init__(self, nodes, features, tolerance=1e-6):
        """Build the clamped/loaded node sets and the free-DOF mask.

        Args:
            nodes: Reference nodal coordinates, shape (N, 3), in mm.
            features: Node feature matrix, shape (N, F); column 0 flags clamped
                nodes and column 1 flags loaded-face nodes.
            tolerance: Coordinate tolerance retained for reporting; the node
                sets themselves come from the feature flags, not from a
                coordinate query.

        Raises:
            ValueError: If no clamped node is flagged, which would leave the
                energy minimisation singular under rigid-body motion.
        """
        self.num_nodes = len(nodes)
        self.tolerance = tolerance

        z = nodes[:, 2]
        z_min, z_max = z.min(), z.max()

        print("\n" + "="*60)
        print("BOUNDARY CONDITIONS SETUP")
        print("="*60)
        print(f"Domain: X=[{nodes[:, 0].min():.4f}, {nodes[:, 0].max():.4f}], "
              f"Y=[{nodes[:, 1].min():.4f}, {nodes[:, 1].max():.4f}], "
              f"Z=[{z_min:.4f}, {z_max:.4f}]")

        self.fixed_nodes  = np.where(features[:, 0] > 0.5)[0]
        self.loaded_nodes = np.where(features[:, 1] > 0.5)[0]

        if len(self.fixed_nodes) == 0:
            raise ValueError(
                "No clamped nodes found (node_features column 0 is all zero). "
                "Re-run the meshing stage — without a Dirichlet set the energy "
                "minimisation is singular under rigid-body motion.")

        # (N, 1) multiplicative mask: 0 on clamped nodes, 1 elsewhere. One column
        # broadcast over all three components, because the clamp fixes the whole
        # vector rather than a single direction (unlike the 2-D roller BCs).
        self.free_mask = np.ones((self.num_nodes, 1), dtype=np.float64)
        self.free_mask[self.fixed_nodes, 0] = 0.0

        print(f"\nClamped face (z={z_min:.4f}): {len(self.fixed_nodes)} nodes "
              f"(ux = uy = uz = 0)")
        print(f"Loaded face  (z={z_max:.4f}): {len(self.loaded_nodes)} nodes "
              f"(tangential traction -> torque)")
        print("Lateral surface: free (traction-free)")

        self._mask_tensor  = None
        self._cached_device = None

    def apply(self, displacements):
        """Zero the displacement at every clamped degree of freedom.

        The mask tensor is cached per device, so repeated calls in the training
        loop do not re-upload it.

        Args:
            displacements: Predicted displacements, shape (N, 3).

        Returns:
            Tensor: Displacements of shape (N, 3), identically zero on the
            clamped nodes.
        """
        dev = displacements.device
        if self._cached_device != dev:
            self._mask_tensor = torch.tensor(
                self.free_mask, dtype=torch.float64, device=dev)
            self._cached_device = dev
        return displacements * self._mask_tensor


# ============================================================
# GEOMETRY AND MESH UTILITIES
# ============================================================

class MeshGeometry:
    """Geometric properties of the tetrahedral mesh."""

    @staticmethod
    def compute_element_volumes(nodes, elements):
        """Tetrahedron volumes  |det[X1-X0, X2-X0, X3-X0]| / 6."""
        print("\nComputing element volumes...")
        p = nodes[elements]                              # (E, 4, 3)
        d = p[:, 1:, :] - p[:, :1, :]                    # (E, 3, 3)
        vols = np.maximum(np.abs(np.linalg.det(d)) / 6.0, Config.MIN_VOLUME)

        print(f"  Element volumes - Min: {vols.min():.6e}, Max: {vols.max():.6e}, "
              f"Mean: {vols.mean():.6e}")
        print(f"  Total volume: {vols.sum():.6f}")
        return vols.astype(np.float64)

    @staticmethod
    def compute_facet_areas(nodes, facets):
        """Triangle areas of ``[elem_id, n1, n2, n3]`` facet rows."""
        p = nodes[facets[:, 1:]]                         # (F, 3, 3)
        areas = 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        return areas.astype(np.float64)

    @staticmethod
    def traction_for_torque(nodes, loading_facets, torque, axis, plate_radius,
                            verbose=True):
        """
        Invert the load: find the rim traction ``tau`` that applies ``torque``.

        The traction field is linear in tau, so the assembled torque is too:
        assemble the nodal load once at tau = 1, read off the torque M_unit it
        produces, and scale.

            tau = torque / M_unit

        Solving against the ASSEMBLED torque rather than the analytic
        tau*pi*R^3/2 is what makes the requested torque the one actually
        delivered. The rim of the mesh is a polygon inscribed in the circle, so
        its second moment is ~0.7 % below the analytic disk value here, and that
        deficit shrinks as the mesh is refined — scaling off the analytic value
        would therefore load two different meshes with two different torques,
        which is exactly what a load parameter must not do.

        Returns (tau, M_unit).
        """
        nodal_unit, M_unit = MeshGeometry.compute_nodal_load(
            nodes, loading_facets, 1.0, axis, plate_radius, verbose=False)
        if abs(M_unit) < 1e-30:
            raise ValueError(
                "The loaded face assembles zero torque — check that "
                "loading_surface_facets.npy is non-empty and that the rod axis "
                "in mesh_summary.json is correct.")
        tau = float(torque) / M_unit
        if verbose:
            M_analytic_unit = np.pi * float(plate_radius) ** 3 / 2.0
            print("\nInverting torque -> rim traction...")
            print(f"  Requested torque : {float(torque):.6f}")
            print(f"  Torque at tau=1  : {M_unit:.6f}  "
                  f"(analytic pi*R^3/2 = {M_analytic_unit:.6f}, "
                  f"mesh is {100*(M_unit/M_analytic_unit - 1):+.3f} % of it)")
            print(f"  -> tau at r=R    : {tau:.6f}")
        return tau, M_unit

    @staticmethod
    def compute_nodal_load(nodes, loading_facets, traction, axis, plate_radius,
                           verbose=True):
        """
        Assemble the consistent nodal force vector of the torsion traction.

        The applied field is the dead load
            t(X) = (tau / R) * ( -(y - cy), (x - cx), 0 )
        which is LINEAR in position and vanishes on the axis, so it needs no
        special case at r = 0 (a normalised e_theta field would be undefined
        there).

        Over one linear triangle both t and u are linear, so the exact external
        work is  Int t.u dA = Sum_i u_i . f_i  with the consistent nodal force

            f_i = A/12 * (2 t_i + t_j + t_k) = A/12 * (t_i + (t_i+t_j+t_k))

        This is exact, not a midpoint approximation — which matters because a
        traction growing linearly with radius is precisely the case where a
        one-point rule loses the r^2 weighting that carries the torque.

        Returns
        -------
        nodal_load : (N, 3)  force per node; W_ext = sum(nodal_load * u).
        """
        if verbose:
            print("\nAssembling torsion traction nodal load...")
        N = len(nodes)
        cx, cy = float(axis[0]), float(axis[1])
        scale = float(traction) / float(plate_radius)

        # Traction vector at every node (only loaded-face rows are ever used).
        t = np.zeros((N, 3), dtype=np.float64)
        t[:, 0] = -(nodes[:, 1] - cy) * scale
        t[:, 1] = (nodes[:, 0] - cx) * scale

        facets = np.asarray(loading_facets, dtype=np.int64).reshape(-1, 4)
        areas = MeshGeometry.compute_facet_areas(nodes, facets)

        nodal_load = np.zeros((N, 3), dtype=np.float64)
        n1, n2, n3 = facets[:, 1], facets[:, 2], facets[:, 3]
        s = t[n1] + t[n2] + t[n3]                        # (F, 3)
        w = (areas / 12.0)[:, None]
        np.add.at(nodal_load, n1, w * (t[n1] + s))
        np.add.at(nodal_load, n2, w * (t[n2] + s))
        np.add.at(nodal_load, n3, w * (t[n3] + s))

        # Diagnostics. The resultant force must vanish (pure couple) and the
        # resultant moment must equal (tau/R)*Int r^2 dA — both are properties of
        # the assembly, so checking them here catches a bad facet set or a
        # mis-set axis before any training time is spent.
        f_res = nodal_load.sum(axis=0)
        M_z = float(np.sum((nodes[:, 0] - cx) * nodal_load[:, 1]
                           - (nodes[:, 1] - cy) * nodal_load[:, 0]))
        if verbose:
            M_analytic = float(traction) * np.pi * float(plate_radius) ** 3 / 2.0
            print(f"  Loading facets   : {len(facets)}  (total area {areas.sum():.6f})")
            print(f"  Resultant force  : [{f_res[0]:.3e}, {f_res[1]:.3e}, {f_res[2]:.3e}]"
                  f"  (must be ~0: pure couple)")
            print(f"  Applied torque Mz: {M_z:.6e}   "
                  f"(analytic tau*pi*R^3/2 = {M_analytic:.6e}, "
                  f"ratio {M_z / M_analytic if M_analytic else float('nan'):.4f})")
        return nodal_load, M_z


# ============================================================
# ENERGY CALCULATIONS
# ============================================================

class CompositeEnergyCalculator:
    """
    Total potential energy of the 3-D composite rod.

    Fully three-dimensional — there is no plane-stress / plane-strain reduction,
    so F, the strain and the Cauchy stress are all genuine 3x3 tensors and every
    stress component (including sigma_zz, sigma_xz, sigma_yz, which torsion
    makes large) is a real unknown rather than a recovered by-product.

    'NH'  — compressible Neo-Hookean, finite strain
                Psi = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2
            with I1 = tr(F^T F) and J = det F.

    'LE'  — linear-elastic energy
                Psi = lambda/2 (tr eps)^2 + mu tr(eps^2)
            evaluated on the SMALL strain eps = sym(grad u) = 1/2 (F + F^T) - I,
            which is exactly the measure the FEM's LE branch uses. The two
            solvers therefore discretise the same functional and the comparison
            is a true code-to-code check. (It is still not PHYSICALLY valid at
            this load: small-strain kinematics cannot represent a finite
            rotation, so both solvers are wrong in the same way. Use NH for the
            physics.)

    Lame parameters are the standard 3-D ones,
        mu = E/(2(1+nu)),  lambda = E nu /((1+nu)(1-2nu)).
    """

    def __init__(self, E_matrix, nu_matrix, nu_inclusion,
                 element_material_ids, material_model='NH', output_unit='kPa'):
        """Precompute the matrix Lame parameters and the inclusion element mask.

        Args:
            E_matrix: Matrix Young's modulus, in ``output_unit``.
            nu_matrix: Matrix Poisson ratio (dimensionless).
            nu_inclusion: Inclusion Poisson ratio (dimensionless).
            element_material_ids: Per-element material tag, shape (E,), where
                0 = matrix, 1 = inclusion, 2 = interface (given matrix
                properties).
            material_model: ``'NH'`` for Neo-Hookean or ``'LE'`` for linear
                elasticity.
            output_unit: Stress unit that ``E_matrix`` is expressed in, used
                for reporting only.
        """
        self.E_matrix             = E_matrix
        self.nu_matrix            = nu_matrix
        self.nu_inclusion         = nu_inclusion
        self.material_model       = material_model.upper()
        self.element_material_ids = element_material_ids
        self.output_unit          = output_unit
        self._inclusion_mask      = element_material_ids == 1

        self.mu_matrix, self.lam_matrix = self._element_lame(E_matrix, nu_matrix)

        self._lame_cache = {}

        print("\n" + "="*60)
        print(f"COMPOSITE MATERIAL PROPERTIES — {self.material_model} (3-D)")
        print("="*60)
        print(f"MATRIX:")
        print(f"  Young's Modulus E: {E_matrix:.4g} {self.output_unit}")
        print(f"  Poisson's Ratio: {nu_matrix:.4f}")
        print(f"  Shear Modulus mu: {self.mu_matrix:.4g} {self.output_unit}")
        print(f"  Lame lambda (3-D): {self.lam_matrix:.4g} {self.output_unit}")
        print(f"\nINCLUSION:")
        print("  Young's Modulus E: set by inclusion_ratio at call time")
        print(f"  Poisson's Ratio: {nu_inclusion:.4f}")
        print(f"\nELEMENT DISTRIBUTION:")
        print(f"  Matrix tets   : {np.sum(element_material_ids == 0)}")
        print(f"  Inclusion tets: {np.sum(element_material_ids == 1)}")
        print(f"  Interface tets: {np.sum(element_material_ids == 2)} (matrix properties)")

    # ── Lamé parameter helpers ────────────────────────────────

    @staticmethod
    def _lame(E, nu):
        """Standard 3-D Lame parameters (mu, lambda)."""
        mu  = E / (2.0 * (1.0 + nu))
        lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return mu, lam

    def _element_lame(self, E, nu):
        """Instance-level wrapper around :meth:`_lame`.

        Args:
            E: Young's modulus, in the configured output unit.
            nu: Poisson ratio (dimensionless).

        Returns:
            tuple: ``(mu, lambda)`` in the same unit as ``E``.
        """
        return self._lame(E, nu)

    def get_element_lame(self, inclusion_ratio):
        """Per-element Lame parameters for a given inclusion_ratio."""
        E_inclusion = self.E_matrix * inclusion_ratio
        mu_inc, lam_inc = self._element_lame(E_inclusion, self.nu_inclusion)
        element_mu  = np.where(self._inclusion_mask, mu_inc,  self.mu_matrix)
        element_lam = np.where(self._inclusion_mask, lam_inc, self.lam_matrix)
        return element_mu, element_lam, E_inclusion

    # ── Deformation-gradient and strain helpers ───────────────

    def compute_deformation_gradient(self, ref_coords, cur_coords, elements):
        """
        F = Ds @ Dm^-1  for each linear tetrahedron, where Dm and Ds hold the
        three edge vectors from the first vertex in the reference and current
        configurations. F is constant per element (linear shape functions).

        Returns (E, 3, 3).
        """
        ref_elem = ref_coords[elements]        # (E, 4, 3)
        cur_elem = cur_coords[elements]

        Dm = torch.stack([ref_elem[:, 1] - ref_elem[:, 0],
                          ref_elem[:, 2] - ref_elem[:, 0],
                          ref_elem[:, 3] - ref_elem[:, 0]], dim=2)   # (E,3,3)
        Ds = torch.stack([cur_elem[:, 1] - cur_elem[:, 0],
                          cur_elem[:, 2] - cur_elem[:, 0],
                          cur_elem[:, 3] - cur_elem[:, 0]], dim=2)
        return torch.matmul(Ds, torch.inverse(Dm))

    # Backwards-compatible alias (the 2-D name used by older call sites).
    compute_deformation_gradient_2d = compute_deformation_gradient

    def compute_green_lagrange_strain(self, F):
        """Green-Lagrange strain E = 1/2 (F^T F - I), 3x3."""
        FTF = torch.matmul(F.transpose(-2, -1), F)
        I   = torch.eye(3, dtype=F.dtype, device=F.device).unsqueeze(0)
        return 0.5 * (FTF - I)

    def compute_small_strain(self, F):
        """Small (engineering) strain eps = sym(grad u) = 1/2 (F + F^T) - I."""
        I = torch.eye(3, dtype=F.dtype, device=F.device).unsqueeze(0)
        return 0.5 * (F + F.transpose(-2, -1)) - I

    def compute_strain_tensor(self, F):
        """The strain measure the ACTIVE material model is defined on.

        'LE' -> small strain sym(grad u).  This must match the FEM, which solves
        sigma = lambda tr(sym grad u) I + 2 mu sym(grad u); using Green-Lagrange
        here instead would have the GNN minimise the St. Venant-Kirchhoff energy
        while the reference stayed small-strain, and the two agree only at small
        strain. At the ~50 % shear this problem is posed at they differ grossly,
        which would show up as a large "network error" that is really a
        constitutive mismatch between the two solvers.

        'NH' -> Green-Lagrange, which is the natural finite-strain measure. The
        Neo-Hookean energy and stress are written directly in terms of F, so
        this value is diagnostic only and never enters the NH energy.
        """
        if self.material_model == 'NH':
            return self.compute_green_lagrange_strain(F)
        return self.compute_small_strain(F)

    # ── Strain-energy densities ───────────────────────────────

    def compute_strain_energy_density_LE(self, strain, element_mu, element_lam):
        """Psi = lambda/2 (tr eps)^2 + mu tr(eps^2), full 3-D trace."""
        tr_e   = strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2]
        e_sq   = torch.matmul(strain, strain)
        tr_esq = e_sq[:, 0, 0] + e_sq[:, 1, 1] + e_sq[:, 2, 2]
        return 0.5 * element_lam * tr_e**2 + element_mu * tr_esq

    def compute_strain_energy_density_NH(self, F, element_mu, element_lam):
        """
        Compressible Neo-Hookean energy density, full 3-D:

            W = mu/2 (I1 - 3) - mu ln J + lambda/2 (ln J)^2
            I1 = tr(F^T F),   J = det F

        No out-of-plane condensation is involved — F33 is a genuine unknown, not
        a kinematic constant as it was in the plane-strain formulation.
        """
        J = torch.linalg.det(F)
        # Guard the logarithm. A tet that inverts mid-training (J <= 0) would
        # otherwise produce NaN and poison every subsequent gradient; clamping
        # leaves such an element with a large finite energy, which the minimiser
        # then pushes back out of inversion.
        J = torch.clamp(J, min=1e-8)

        FTF = torch.matmul(F.transpose(-2, -1), F)
        I1  = FTF[:, 0, 0] + FTF[:, 1, 1] + FTF[:, 2, 2]
        lnJ = torch.log(J)

        return 0.5 * element_mu * (I1 - 3.0) - element_mu * lnJ \
            + 0.5 * element_lam * lnJ**2

    # ── Internal energy ───────────────────────────────────────

    def _get_cached_lame(self, inclusion_ratio, dev):
        """Return per-element Lame tensors, memoised by ratio and device.

        The cache is bounded at 128 entries so a sweep over many ratios cannot
        grow it without limit.

        Args:
            inclusion_ratio: E_inclusion / E_matrix (dimensionless).
            dev: Torch device the tensors are needed on.

        Returns:
            tuple: ``(element_mu, element_lambda)``, each of shape (E,) and
            dtype float64.
        """
        ratio_key = (round(inclusion_ratio, 6), str(dev))
        if ratio_key not in self._lame_cache:
            emu_np, elam_np, _ = self.get_element_lame(inclusion_ratio)
            self._lame_cache[ratio_key] = (
                torch.tensor(emu_np,  dtype=torch.float64, device=dev),
                torch.tensor(elam_np, dtype=torch.float64, device=dev),
            )
            if len(self._lame_cache) > 128:
                del self._lame_cache[next(iter(self._lame_cache))]
        return self._lame_cache[ratio_key]

    def compute_internal_energy(self, ref_coords, cur_coords, elements,
                                element_volumes, inclusion_ratio=1.0):
        """Integrate the strain-energy density over the whole body.

        Args:
            ref_coords: Reference nodal coordinates, shape (N, 3).
            cur_coords: Deformed nodal coordinates, shape (N, 3).
            elements: Tetrahedron connectivity, shape (E, 4).
            element_volumes: Reference element volumes, shape (E,).
            inclusion_ratio: E_inclusion / E_matrix (dimensionless).

        Returns:
            tuple: ``(E_internal, F, strain)`` where ``E_internal`` is a scalar
            tensor, ``F`` is the deformation gradient of shape (E, 3, 3) and
            ``strain`` is the active model's strain measure, shape (E, 3, 3).
            For the Neo-Hookean model ``strain`` is diagnostic only and never
            enters the energy.
        """
        F   = self.compute_deformation_gradient(ref_coords, cur_coords, elements)
        dev = ref_coords.device
        emu, elam = self._get_cached_lame(inclusion_ratio, dev)

        if self.material_model == 'NH':
            psi    = self.compute_strain_energy_density_NH(F, emu, elam)
            strain = self.compute_strain_tensor(F)   # for debug / viz only
        else:
            strain = self.compute_strain_tensor(F)
            psi    = self.compute_strain_energy_density_LE(strain, emu, elam)

        E_internal = torch.sum(psi * element_volumes)
        return E_internal, F, strain

    # ── External work ─────────────────────────────────────────

    def compute_external_work(self, displacements, nodal_load):
        """
        W_ext = Int t.u dS = sum_i f_i . u_i

        ``nodal_load`` is the exact consistent nodal force vector of the
        tangential traction (see MeshGeometry.compute_nodal_load), so this is an
        exact surface integral for a linear displacement field, not a quadrature
        approximation. It is also linear in u, which is what makes Pi a genuine
        potential and the dead load conservative.
        """
        return torch.sum(nodal_load * displacements)

    # ── Total potential energy ────────────────────────────────

    def compute_total_potential_energy(self, ref_coords, displacements, elements,
                                       element_volumes, nodal_load,
                                       inclusion_ratio=1.0):
        """Evaluate Pi(u) = E_int(u) - W_ext(u), the quantity training minimises.

        Args:
            ref_coords: Reference nodal coordinates, shape (N, 3).
            displacements: Predicted displacements, shape (N, 3), already
                masked by the boundary conditions.
            elements: Tetrahedron connectivity, shape (E, 4).
            element_volumes: Reference element volumes, shape (E,).
            nodal_load: Consistent nodal force vector, shape (N, 3).
            inclusion_ratio: E_inclusion / E_matrix (dimensionless).

        Returns:
            tuple: ``(Pi, E_int, W_ext, F, strain)``. The first three are
            scalar tensors; ``F`` and ``strain`` have shape (E, 3, 3).
        """
        cur_coords = ref_coords + displacements
        E_int, F, strain = self.compute_internal_energy(
            ref_coords, cur_coords, elements, element_volumes, inclusion_ratio)
        W_ext = self.compute_external_work(displacements, nodal_load)
        Pi    = E_int - W_ext
        return Pi, E_int, W_ext, F, strain

    # ── Cauchy stress ─────────────────────────────────────────

    def cauchy_stress_LE(self, strain, emu, elam):
        """sigma = lambda tr(eps) I + 2 mu eps  (E,3,3)."""
        tr_e = strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2]
        I    = torch.eye(3, dtype=strain.dtype, device=strain.device
                         ).unsqueeze(0).expand(len(strain), -1, -1)
        return elam[:, None, None] * tr_e[:, None, None] * I \
            + 2.0 * emu[:, None, None] * strain

    def cauchy_stress_NH(self, F, emu, elam):
        """
        Compressible Neo-Hookean Cauchy stress:

            sigma = (1/J) ( mu (B - I) + lambda ln J I ),   B = F F^T
        """
        J = torch.clamp(torch.linalg.det(F), min=1e-8)
        B = torch.matmul(F, F.transpose(-2, -1))
        I = torch.eye(3, dtype=F.dtype, device=F.device
                      ).unsqueeze(0).expand(len(F), -1, -1)
        lnJ = torch.log(J)
        return (1.0 / J[:, None, None]) * (
            emu[:, None, None] * (B - I) + elam[:, None, None] * lnJ[:, None, None] * I)

    @staticmethod
    def von_mises(sxx, syy, szz, sxy, syz, sxz):
        """
        Full 3-D von Mises invariant:

            sigma_vm = sqrt( 1/2[(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2]
                             + 3(sxy^2 + syz^2 + sxz^2) )

        In torsion the out-of-plane shears sigma_xz / sigma_yz carry most of the
        load, so all three shear terms must be present — the 2-D form that kept
        only sigma_xy would under-report the invariant by nearly the whole
        torsional contribution.
        """
        return torch.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
        )

    def stress_tensor(self, F, strain, emu, elam):
        """Full 3x3 Cauchy stress for the active material model."""
        if self.material_model == 'NH':
            return self.cauchy_stress_NH(F, emu, elam)
        return self.cauchy_stress_LE(strain, emu, elam)

    def stress_components(self, F, strain, emu, elam):
        """(sxx, syy, szz, sxy, syz, sxz, von_mises) for the active model."""
        sigma = self.stress_tensor(F, strain, emu, elam)
        sxx = sigma[:, 0, 0]
        syy = sigma[:, 1, 1]
        szz = sigma[:, 2, 2]
        sxy = sigma[:, 0, 1]
        syz = sigma[:, 1, 2]
        sxz = sigma[:, 0, 2]
        return sxx, syy, szz, sxy, syz, sxz, \
            self.von_mises(sxx, syy, szz, sxy, syz, sxz)


# ============================================================
# CYLINDRICAL STRESS COMPONENTS
# ============================================================

def cylindrical_components(sigma_cart, centroids, axis):
    """
    Rotate a Cartesian stress field into rod (cylindrical) coordinates.

    Returns (s_rr, s_tt, s_zz, s_tz, s_rz, s_rt) as numpy arrays.

    ``s_tz`` (sigma_theta_z) is THE torsion stress: for a homogeneous shaft it is
    the only non-zero component and grows linearly with radius. In this composite
    it also picks up the 5-fold pattern of the petal inclusion, which is what
    makes it the most informative field to plot.

    On the axis (r = 0) the radial/tangential directions are undefined; e_r is
    set to x-hat there, which is arbitrary but harmless because the deviatoric
    invariant is rotation-independent and r = 0 is a single line of elements.
    """
    sigma = np.asarray(sigma_cart, dtype=np.float64)
    cx, cy = float(axis[0]), float(axis[1])
    dx = centroids[:, 0] - cx
    dy = centroids[:, 1] - cy
    r  = np.hypot(dx, dy)
    safe = r > 1e-12
    c = np.where(safe, dx / np.maximum(r, 1e-30), 1.0)
    s = np.where(safe, dy / np.maximum(r, 1e-30), 0.0)

    e_r = np.stack([c, s, np.zeros_like(c)], axis=1)
    e_t = np.stack([-s, c, np.zeros_like(c)], axis=1)
    e_z = np.stack([np.zeros_like(c), np.zeros_like(c), np.ones_like(c)], axis=1)

    def project_stress(a, b):
        """Contract the stress tensor onto a pair of unit directions: a.sigma.b."""
        return np.einsum('ei,eij,ej->e', a, sigma, b)

    return (project_stress(e_r, e_r), project_stress(e_t, e_t),
            project_stress(e_z, e_z), project_stress(e_t, e_z),
            project_stress(e_r, e_z), project_stress(e_r, e_t))


def twist_angle_deg(nodes, disp, axis, node_mask=None, min_radius_frac=0.3,
                    plate_radius=None):
    """
    Rotation of the material about the rod axis, in degrees, per node.

    Computed as the change in polar angle of the node about the axis, wrapped to
    (-pi, pi]. Nodes closer to the axis than ``min_radius_frac * R`` are excluded
    from the summary statistics: there the angle is the ratio of two small
    numbers and is dominated by noise, even though the physical twist is the
    same.

    Returns (per_node_deg, mean_deg, max_deg).
    """
    cx, cy = float(axis[0]), float(axis[1])
    x0 = nodes[:, 0] - cx
    y0 = nodes[:, 1] - cy
    r0 = np.hypot(x0, y0)
    th0 = np.arctan2(y0, x0)
    th1 = np.arctan2(y0 + disp[:, 1], x0 + disp[:, 0])
    dth = np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0))
    deg = np.degrees(dth)

    R = plate_radius if plate_radius is not None else float(r0.max())
    sel = r0 > min_radius_frac * R
    if node_mask is not None:
        sel = sel & node_mask
    if not sel.any():
        return deg, float('nan'), float('nan')
    return deg, float(deg[sel].mean()), float(deg[sel][np.argmax(np.abs(deg[sel]))])


# ============================================================
# GNN ARCHITECTURE
# ============================================================

# Width of the geometric edge feature vector: [dx, dy, dz, |d|].
EDGE_ATTR_DIM = 4


class GNNLayer(MessagePassing):
    """
    Single GNN layer with message passing.

    The message is built from the receiver state x_i, the *difference*
    x_j - x_i, and the geometric edge attribute [dx, dy, dz, |d|] (the
    reference-configuration offset of the sender from the receiver, and its
    length).

    Passing x_j - x_i together with the edge vector — rather than the raw x_j —
    is what lets a layer represent a finite-difference stencil: a directional
    derivative needs both the state difference and the physical separation it is
    taken over. With only [x_i, x_j] the layer can do little more than
    mean-smooth its neighbourhood, so stacking depth blurs the field instead of
    widening the effective stencil.
    """

    def __init__(self, in_dim, out_dim, edge_dim=EDGE_ATTR_DIM):
        """Build the message and update multilayer perceptrons.

        Args:
            in_dim: Width of the incoming node features.
            out_dim: Width of the produced node features.
            edge_dim: Width of the geometric edge attribute, 4 for
                ``[dx, dy, dz, |d|]``.
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
    """Graph Neural Network predicting the 3-component displacement field."""

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
        self.output_layer = nn.Linear(hidden_dim, 3)

        print("\n" + "="*60)
        print("GNN ARCHITECTURE")
        print("="*60)
        print(f"Input dimension: {input_dim}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"Number of GNN layers: {num_layers}")
        print(f"Output dimension: 3 (ux, uy, uz displacements)")
        print(f"Output displacement scale: {self.output_scale:.6g}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")

    def forward(self, x, edge_index, edge_attr=None):
        """Predict the nodal displacement field.

        Args:
            x: Node features, shape (N, input_dim).
            edge_index: Graph connectivity, shape (2, E).
            edge_attr: Geometric edge features, shape (E, 4).

        Returns:
            Tensor: Displacements ``(ux, uy, uz)``, shape (N, 3), already
            multiplied by ``output_scale``. Clamped nodes are NOT zeroed here —
            that is applied separately by :meth:`BoundaryConditions.apply`.
        """
        h = F.relu(self.input_embed(x))
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)
        return self.output_layer(h) * self.output_scale


# ============================================================
# DATA LOADING
# ============================================================

def load_mesh_summary(data_dir):
    """Read the rod geometry the meshing stage recorded.

    The torsion axis and radius define the traction field, so they must come
    from the mesh that was actually built. Re-deriving the axis from a bounding
    box downstream would be wrong for a petal, whose bbox is not centred on the
    shape.
    """
    path = Path(data_dir) / 'mesh_summary.json'
    if not path.exists():
        raise FileNotFoundError(
            f"mesh_summary.json not found in {data_dir}. It carries the rod "
            f"axis/radius/height that the traction field is defined from — "
            f"re-run the meshing stage.")
    with open(path) as f:
        return json.load(f)


def load_mesh_data(data_dir):
    """Load every mesh array the training stage needs.

    Args:
        data_dir: Directory written by the meshing stage.

    Returns:
        tuple: ``(nodes, elements, features, loading_facets, topology,
        element_material_ids)`` with shapes (N, 3), (E, 4), (N, F), (L, 4),
        (N, max_degree) and (E,) respectively. ``topology`` is padded with -1
        where a node has fewer neighbours than the maximum degree, and
        ``loading_facets`` rows are ``[elem_id, n1, n2, n3]``.
    """
    print("\n" + "="*60)
    print("LOADING COMPOSITE ROD MESH DATA")
    print("="*60)

    data_dir = Path(data_dir)
    nodes                = np.load(data_dir / 'nodes.npy')
    elements             = np.load(data_dir / 'elements.npy')
    features             = np.load(data_dir / 'node_features.npy')
    loading_facets       = np.load(data_dir / 'loading_surface_facets.npy')
    topology             = np.load(data_dir / 'node_topology.npy')
    element_material_ids = np.load(data_dir / 'element_material_ids.npy')

    print(f"Loaded composite rod mesh with:")
    print(f"  Nodes: {len(nodes)}  (3-D)")
    print(f"  Tetrahedra: {len(elements)}")
    print(f"  Loading facets (z=H triangles): {len(loading_facets)}")
    print(f"  Material distribution:")
    print(f"    - Matrix tets   : {np.sum(element_material_ids == 0)}")
    print(f"    - Inclusion tets: {np.sum(element_material_ids == 1)}")
    print(f"    - Interface tets: {np.sum(element_material_ids == 2)}")

    return nodes, elements, features, loading_facets, topology, element_material_ids


def build_graph_data(nodes, elements, features, topology):
    """Assemble the PyTorch Geometric graph the GNN consumes.

    Node coordinates are min-max normalised to the unit cube so the input
    features are O(1); the geometric edge attributes are built from the same
    normalised coordinates, which keeps them mesh-size-consistent.

    Args:
        nodes: Reference nodal coordinates, shape (N, 3), in mm.
        elements: Tetrahedron connectivity, shape (E, 4). Accepted for
            signature symmetry with the other loaders; the graph is built from
            ``topology``.
        features: Node feature matrix, shape (N, F); columns 2:4 are the
            material one-hot and columns 4:7 the raw coordinates.
        topology: Neighbour lists padded with -1, shape (N, max_degree).

    Returns:
        Data: Graph with ``x`` of shape (N, 5), ``edge_index`` of shape (2, E)
        and ``edge_attr`` of shape (E, 4) holding ``[dx, dy, dz, |d|]``.
    """
    print("\nBuilding graph structure...")

    xyz    = features[:, 4:7].astype(np.float64)
    mat_id = features[:, 2:4].astype(np.float64)
    xyz_min = xyz.min(axis=0)
    xyz_range = xyz.max(axis=0) - xyz_min
    xyz_range[xyz_range < 1e-10] = 1.0
    xyz_norm = (xyz - xyz_min) / xyz_range

    node_features = torch.tensor(np.concatenate([xyz_norm, mat_id], axis=1),
                                 dtype=torch.float64)
    print(f"  Coords normalised to [0,1]^3  -> input dim {node_features.shape[1]}")

    # Vectorised edge list from the padded topology array (-1 = no neighbour).
    valid = topology >= 0
    src_nodes = np.repeat(np.arange(len(topology)), valid.sum(axis=1))
    dst_nodes = topology[valid]
    edge_index = torch.tensor(np.stack([src_nodes, dst_nodes]), dtype=torch.long)

    print(f"  Graph edges: {edge_index.shape[1]}")
    print(f"  Average node degree: {edge_index.shape[1] / len(nodes):.2f}")

    # Geometric edge attributes [dx, dy, dz, |d|] in the *reference*
    # configuration, from the same normalised coordinates the network sees as
    # input, so the feature is O(1) and mesh-size-consistent. Row k corresponds
    # to edge_index[:, k] = (dst, src): the offset of the sender from the receiver.
    dst, src = edge_index[0], edge_index[1]
    xyz_t = torch.tensor(xyz_norm, dtype=torch.float64)
    d_vec = xyz_t[src] - xyz_t[dst]                      # (E, 3)
    d_len = d_vec.norm(dim=1, keepdim=True)              # (E, 1)
    edge_attr = torch.cat([d_vec, d_len], dim=1)         # (E, 4)
    print(f"  Edge attributes: {tuple(edge_attr.shape)} [dx, dy, dz, |d|] "
          f"(mean |d| = {d_len.mean().item():.5f} in normalised units)")

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================
# TRAINING
# ============================================================

class Trainer:
    """Training loop for the physics-informed GNN on the composite rod."""

    def __init__(self, model, energy_calculator, bc_handler, graph_data,
                 ref_coords, elements, element_volumes, nodal_load,
                 nodes_np, elements_np, element_material_ids, config,
                 mesh_info=None):
        """Wire up the optimizer, logging and slice-plot triangulation.

        Args:
            model: The ``DisplacementGNN`` to train.
            energy_calculator: ``CompositeEnergyCalculator`` supplying Pi(u).
            bc_handler: ``BoundaryConditions`` imposing the clamp.
            graph_data: PyTorch Geometric graph holding ``x``, ``edge_index``
                and ``edge_attr``.
            ref_coords: Reference nodal coordinates as a tensor, shape (N, 3).
            elements: Tetrahedron connectivity as a tensor, shape (E, 4).
            element_volumes: Reference element volumes as a tensor, shape (E,).
            nodal_load: Consistent nodal force vector as a tensor, shape (N, 3).
            nodes_np: Reference nodal coordinates as a NumPy array, shape
                (N, 3), used by the plotting and twist diagnostics.
            elements_np: Tetrahedron connectivity as a NumPy array, shape (E, 4).
            element_material_ids: Per-element material tag, shape (E,).
            config: Config-like object holding the hyperparameters and paths.
            mesh_info: Parsed ``mesh_summary.json``; supplies the rod axis,
                radius, height and extrusion layout. Slice plots are disabled
                when the layout keys are absent.
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
        self.element_volumes = element_volumes
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
                        'max_disp': [], 'twist_deg': []}

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

        # Cross-section triangulations for the slice plots. The mesh is a
        # straight extrusion with layer-major node ids, so a z-layer is the
        # first N_section nodes offset by layer*N_section, and the cross-section
        # connectivity is recoverable from the bottom layer of tets.
        self._setup_slice_plotting()

        print("\n" + "="*60)
        print("TRAINER INITIALIZED")
        print("="*60)
        print(f"Optimizer: Adam (lr={config.LEARNING_RATE})")
        print(f"Material model: {config.MATERIAL_MODEL}")
        print(f"Applied torque: {config.APPLIED_TORQUE} "
              f"{config.POSTPROC_OUTPUT_UNIT}*mm^3  "
              f"(rim traction {config.TRACTION_MAGNITUDE:.6g} "
              f"{config.POSTPROC_OUTPUT_UNIT})")
        print(f"TensorBoard logs: {tb_dir}")

    def _setup_slice_plotting(self):
        """Build the cross-section triangulation used by the field plots."""
        info = self.mesh_info
        self._n_sec_nodes = info.get('n_section_nodes')
        self._n_sec_elems = info.get('n_section_elements')
        self._n_layers    = info.get('n_layers')
        self._axis        = (info.get('axis_x', 0.0), info.get('axis_y', 0.0))
        self._radius      = info.get('plate_radius')
        self._height      = info.get('height')
        self._triang = None
        self._section_tris = None

        if not (self._n_sec_nodes and self._n_sec_elems and self._n_layers):
            print("  (mesh_summary lacks the extrusion layout — slice plots disabled)")
            return

        N2 = self._n_sec_nodes
        T2 = self._n_sec_elems
        # The first tet of each prism in layer 0 is the one built as
        # (b0, b1, b2, t0) — three bottom-layer vertices and one top. Recover
        # the cross-section triangle as its vertices BELOW N2 rather than as its
        # first three columns: extrude_to_3d re-orders vertices when it flips a
        # negative-volume tet, so column position is not reliable, but layer
        # membership is.
        first_tets = self.elements_np[0:T2 * 3:3]
        if len(first_tets) != T2:
            print("  (unexpected extrusion ordering — slice plots disabled)")
            return
        in_layer0 = first_tets < N2
        if not np.all(in_layer0.sum(axis=1) == 3):
            print("  (unexpected extrusion ordering — slice plots disabled)")
            return
        section_tris = first_tets[in_layer0].reshape(T2, 3)
        self._section_tris = section_tris
        pts = self.nodes_np[:N2]
        self._triang = mtri.Triangulation(pts[:, 0], pts[:, 1], section_tris)

    def _layer_node_slice(self, layer):
        """Node indices of one z-layer (layer-major ordering)."""
        N2 = self._n_sec_nodes
        return np.arange(N2) + int(layer) * N2

    def _layer_cell_values(self, cell_vals, layer):
        """Average the 3 tets of every prism in ``layer`` -> per cross-section
        triangle value, so a 3-D element field can be drawn on the 2-D slice."""
        T2 = self._n_sec_elems
        block = np.asarray(cell_vals)[layer * T2 * 3:(layer + 1) * T2 * 3]
        return block.reshape(T2, 3).mean(axis=1)

    def _make_input(self):
        """Return the node-feature matrix fed to the model, shape (N, input_dim)."""
        return self.graph_data.x

    def train_step(self, inclusion_ratio):
        """Take one full-batch gradient step on the total potential energy.

        Args:
            inclusion_ratio: E_inclusion / E_matrix (dimensionless).

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
            self.ref_coords, displacements, self.elements, self.element_volumes,
            self.nodal_load, inclusion_ratio=inclusion_ratio
        )

        Pi.backward()
        self.optimizer.step()

        loss_val  = Pi.item()
        E_int_val = E_int.item()
        W_ext_val = W_ext.item()
        max_disp  = torch.max(torch.abs(displacements)).item()

        del _F, _strain, displacements, displacements_raw, x_input, Pi, E_int, W_ext
        return loss_val, E_int_val, W_ext_val, max_disp

    def _current_twist_deg(self):
        """Mean twist of the loaded face, in degrees — the headline output."""
        with torch.no_grad():
            disp = self.bc.apply(self._orig_model(
                self.graph_data.x, self.graph_data.edge_index,
                getattr(self.graph_data, 'edge_attr', None))).cpu().numpy()
        mask = np.zeros(len(self.nodes_np), dtype=bool)
        mask[self.bc.loaded_nodes] = True
        _, mean_deg, _ = twist_angle_deg(
            self.nodes_np, disp, self._axis, node_mask=mask,
            plate_radius=self._radius)
        return mean_deg

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
        # training time — that number must reflect optimization compute only.
        self._checkpoint_io_time = 0.0
        inclusion_ratio = float(self.config.INCLUSION_RATIO)

        for epoch in range(self.config.NUM_EPOCHS):
            loss, E_int, W_ext, max_disp = self.train_step(inclusion_ratio)

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
                twist = self._current_twist_deg()
                self.history['twist_deg'].append(twist)
                self.writer.add_scalar('Torsion/twist_deg', twist, epoch)
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
                print(f"  Mean twist of loaded face: {twist:.3f} deg")

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
            'inclusion_ratio':     float(self.config.INCLUSION_RATIO),
            # Needed to rebuild an identical model at inference time; the scale
            # is not a parameter, so it does NOT live in model_state_dict, and
            # rebuilding without it would silently rescale the entire field.
            'output_displacement_scale': self._orig_model.output_scale,
            # Architecture, so Stage 3 can rebuild the right shape in a fresh
            # process rather than falling back to the live Config defaults.
            'input_dim':   int(self._orig_model.input_dim),
            'hidden_dim':  int(self._orig_model.hidden_dim),
            'num_layers':  int(self._orig_model.num_layers),
            'applied_torque':      float(self.config.APPLIED_TORQUE),
            # Derived from applied_torque on the training mesh; recorded so a
            # comparison run can report the load the model was actually trained
            # under without re-deriving it.
            'traction_magnitude':  float(self.config.TRACTION_MAGNITUDE),
            'poissons_ratio_matrix':    float(self.config.POISSONS_RATIO_MATRIX),
            'poissons_ratio_inclusion': float(self.config.POISSONS_RATIO_INCLUSION),
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

    def _eval_at_ratio(self, inclusion_ratio):
        """Inference: nodal displacement + element-centred stress arrays."""
        with torch.no_grad():
            x_input           = self._make_input()
            displacements_raw = self._orig_model(
                x_input, self.graph_data.edge_index,
                getattr(self.graph_data, 'edge_attr', None))
            displacements     = self.bc.apply(displacements_raw)

            cur_coords = self.ref_coords + displacements
            F_def      = self.energy_calc.compute_deformation_gradient(
                             self.ref_coords, cur_coords, self.elements)
            strain     = self.energy_calc.compute_strain_tensor(F_def)

            dev       = F_def.device
            emu, elam = self.energy_calc._get_cached_lame(inclusion_ratio, dev)

            sigma = self.energy_calc.stress_tensor(F_def, strain, emu, elam)
            sxx, syy, szz, sxy, syz, sxz, vm = \
                self.energy_calc.stress_components(F_def, strain, emu, elam)
            J = torch.linalg.det(F_def)

            out = dict(
                disp=displacements.cpu().numpy(),
                sigma=sigma.cpu().numpy(),
                von_mises=vm.cpu().numpy(),
                J=J.cpu().numpy(),
            )
            del x_input, displacements_raw, displacements, cur_coords
            del F_def, strain, sigma, sxx, syy, szz, sxy, syz, sxz, vm, J
        return out

    def _save_field_plots(self, epoch, save_dir):
        """Slice plots of the fields on the loaded face and at mid-height."""
        save_dir = Path(save_dir)
        self._orig_model.eval()

        if self._triang is None:
            return

        ratio = float(self.config.INCLUSION_RATIO)
        res = self._eval_at_ratio(ratio)
        disp = res['disp']
        unit = self.config.POSTPROC_OUTPUT_UNIT

        top_layer = self._n_layers
        mid_layer = self._n_layers // 2
        top_nodes = self._layer_node_slice(top_layer)
        pts = self.nodes_np[:self._n_sec_nodes]

        # ── Displacement on the loaded face ──
        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
        # Statistics over the LOADED FACE only. A whole-body mean is dragged
        # down by the clamped end, where the twist is zero by construction, and
        # would report roughly half the actual end rotation in the title.
        loaded_mask = np.zeros(len(self.nodes_np), dtype=bool)
        loaded_mask[self.bc.loaded_nodes] = True
        deg, mean_deg, _ = twist_angle_deg(
            self.nodes_np, disp, self._axis, node_mask=loaded_mask,
            plate_radius=self._radius)
        fig.suptitle(
            f'Loaded face (z=H) — Epoch {epoch}  [{self.config.MATERIAL_MODEL}]  '
            f'mean twist {mean_deg:.2f} deg',
            fontsize=13, fontweight='bold')

        for ax, vals, label in [
            (axes[0, 0], disp[top_nodes, 0], 'ux (mm)'),
            (axes[0, 1], disp[top_nodes, 1], 'uy (mm)'),
            (axes[1, 0], disp[top_nodes, 2], 'uz — WARPING (mm)'),
            (axes[1, 1], deg[top_nodes],     'twist angle (deg)'),
        ]:
            tc = ax.tripcolor(self._triang, vals, cmap='jet', shading='gouraud')
            plt.colorbar(tc, ax=ax, label=label)
            ax.set_title(label); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'displacement_face_epoch_{epoch}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ── Stress at mid-height ──
        sigma = res['sigma']
        centroids = self.nodes_np[self.elements_np].mean(axis=1)
        s_rr, s_tt, s_zz, s_tz, s_rz, s_rt = cylindrical_components(
            sigma, centroids, self._axis)

        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
        fig.suptitle(
            f'Stress at mid-height z=H/2 — Epoch {epoch} '
            f'[{self.config.MATERIAL_MODEL}, E_mat={self.energy_calc.E_matrix:.3g} {unit}, '
            f'E_inc={self.energy_calc.E_matrix * ratio:.3g} {unit}]',
            fontsize=13, fontweight='bold')

        for ax, vals, title in [
            (axes[0, 0], s_tz,             'sigma_theta_z (torsion shear)'),
            (axes[0, 1], res['von_mises'], 'von Mises'),
            (axes[1, 0], s_zz,             'sigma_zz (axial)'),
            (axes[1, 1], s_rz,             'sigma_rz'),
        ]:
            slice_vals = self._layer_cell_values(vals, mid_layer)
            tc = ax.tripcolor(self._triang, facecolors=slice_vals,
                              cmap='jet', shading='flat')
            plt.colorbar(tc, ax=ax, label=unit)
            ax.set_title(title); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'stress_midheight_epoch_{epoch}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

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
                      element_material_ids, history, config, mesh_info,
                      inclusion_ratio=1.0):
    """Training history + a 3-D view of the twisted rod."""
    print("\n" + "="*60)
    print(f"GENERATING VISUALIZATIONS (ratio={inclusion_ratio:.4f})")
    print("="*60)

    model.eval()
    with torch.no_grad():
        disp = bc_handler.apply(model(
            _make_model_input(graph_data), graph_data.edge_index,
            getattr(graph_data, 'edge_attr', None))).cpu().numpy()

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(exist_ok=True, parents=True)

    axis = (mesh_info.get('axis_x', 0.0), mesh_info.get('axis_y', 0.0))
    R = mesh_info.get('plate_radius')
    # Restrict the summary statistics to the LOADED face. A whole-body mean is
    # dominated by the clamped end, where the twist is zero by construction, and
    # would report roughly half the actual end rotation.
    loaded_mask = np.zeros(len(nodes), dtype=bool)
    loaded_mask[bc_handler.loaded_nodes] = True
    deg, mean_deg, max_deg = twist_angle_deg(
        nodes, disp, axis, node_mask=loaded_mask, plate_radius=R)

    # ── Deformed rod (undeformed vs deformed surface nodes) ──
    incl = element_material_ids == 1
    node_incl = np.zeros(len(nodes), dtype=bool)
    if incl.any():
        node_incl[np.unique(elements[incl].ravel())] = True

    fig = plt.figure(figsize=(16, 7))
    for i, (coords, title) in enumerate([
            (nodes, 'Reference'),
            (nodes + disp, f'Deformed (true scale) — mean twist {mean_deg:.2f} deg')]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        step = max(1, len(nodes) // 6000)
        s = np.arange(0, len(nodes), step)
        ax.scatter(coords[s, 0], coords[s, 1], coords[s, 2],
                   c=node_incl[s], cmap='coolwarm', s=3, alpha=0.5)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'deformed_rod.png', dpi=200, bbox_inches='tight')
    print("  Saved: deformed_rod.png")
    plt.close(fig)

    # ── Training history ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
    ax.set_title('External Work (torque)', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_yscale('log')

    ax = axes[1, 1]
    if history['twist_deg']:
        xs = np.linspace(0, history['epoch'][-1] if history['epoch'] else 0,
                         len(history['twist_deg']))
        ax.plot(xs, history['twist_deg'], 'm-o', markersize=3, linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Mean twist of loaded face (deg)')
    ax.set_title('Twist angle', fontweight='bold'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    print("  Saved: training_history.png")
    plt.close(fig)
    gc.collect()

    print(f"  Mean twist of the loaded face : {mean_deg:.3f} deg")
    print(f"  Max  twist                    : {max_deg:.3f} deg")
    print(f"  Max |uz| (warping)            : {np.abs(disp[:, 2]).max():.5f} mm")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the full training stage end to end.

    Loads the mesh, derives the rim traction that delivers ``APPLIED_TORQUE``
    on that mesh, builds the graph, energy calculator and model, trains to
    convergence and writes checkpoints plus summary figures. Paths and
    hyperparameters are read from the module-level ``Config``.
    """
    print("="*60)
    print("PHYSICS-INFORMED GNN — 3-D COMPOSITE ROD IN TORSION")
    print("="*60)
    print(f"Device:         {Config.DEVICE}")
    print(f"Material model: {Config.MATERIAL_MODEL}")

    setup_cpu_limits(Config)

    mesh_info = load_mesh_summary(Config.DATA_DIR)
    nodes, elements, features, loading_facets, topology, element_material_ids = \
        load_mesh_data(Config.DATA_DIR)

    axis = (mesh_info['axis_x'], mesh_info['axis_y'])
    R    = mesh_info['plate_radius']
    H    = mesh_info['height']
    # Recorded on Config so effective_output_displacement_scale can size the
    # characteristic displacement from the actual rod, not a guess.
    Config.ROD_HEIGHT = H
    print(f"\nRod geometry: R={R}, H={H}, axis=({axis[0]:.6f}, {axis[1]:.6f})")

    element_volumes = MeshGeometry.compute_element_volumes(nodes, elements)

    # The user specifies a TORQUE; the rim traction that delivers it on this
    # particular mesh is derived here and recorded on the config, because the
    # energy calculator, the output scaling and the checkpoint all report it.
    Config.TRACTION_MAGNITUDE, _ = MeshGeometry.traction_for_torque(
        nodes, loading_facets, Config.APPLIED_TORQUE, axis, R)
    nodal_load, M_z = MeshGeometry.compute_nodal_load(
        nodes, loading_facets, Config.TRACTION_MAGNITUDE, axis, R)
    bc_handler = BoundaryConditions(nodes, features)

    ref_coords            = torch.tensor(nodes,           dtype=torch.float64, device=Config.DEVICE)
    elements_tensor       = torch.tensor(elements,        dtype=torch.long,    device=Config.DEVICE)
    element_volumes_t     = torch.tensor(element_volumes, dtype=torch.float64, device=Config.DEVICE)
    nodal_load_t          = torch.tensor(nodal_load,      dtype=torch.float64, device=Config.DEVICE)

    graph_data = build_graph_data(nodes, elements, features, topology).to(Config.DEVICE)

    print("\n" + "="*60)
    print("TRAINING SETUP")
    print("="*60)
    E_mat = float(Config.YOUNGS_MODULUS_MATRIX)
    unit  = Config.POSTPROC_OUTPUT_UNIT
    print(f"  Inclusion ratio  : {Config.INCLUSION_RATIO:.4f} (fixed)")
    print(f"  E_matrix         : {E_mat:.4g} {unit}")
    print(f"  E_inclusion      : {E_mat * Config.INCLUSION_RATIO:.4g} {unit}")
    print(f"  Applied torque   : {Config.APPLIED_TORQUE:.6g} {unit}*mm^3"
          f"   (= {Config.APPLIED_TORQUE:.6g} uN*m for unit=kPa)")
    print(f"  Realised torque  : {M_z:.6g} {unit}*mm^3   "
          f"(rel. err {abs(M_z / Config.APPLIED_TORQUE - 1):.2e})")
    print(f"  -> rim traction  : {Config.TRACTION_MAGNITUDE:.6g} {unit}  (derived)")

    energy_calc = CompositeEnergyCalculator(
        E_mat,
        Config.POISSONS_RATIO_MATRIX,
        Config.POISSONS_RATIO_INCLUSION,
        element_material_ids,
        material_model=Config.MATERIAL_MODEL,
        output_unit=unit,
    )

    input_dim = graph_data.x.shape[1]
    model = DisplacementGNN(
        input_dim=input_dim,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        output_scale=effective_output_displacement_scale(Config, char_length=H),
    ).double().to(Config.DEVICE)

    trainer = Trainer(
        model, energy_calc, bc_handler, graph_data,
        ref_coords, elements_tensor, element_volumes_t, nodal_load_t,
        nodes, elements, element_material_ids, Config, mesh_info=mesh_info
    )

    trainer.train()

    visualize_results(model, graph_data, bc_handler, nodes, elements,
                      element_material_ids, trainer.history, Config, mesh_info,
                      inclusion_ratio=float(Config.INCLUSION_RATIO))

    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
