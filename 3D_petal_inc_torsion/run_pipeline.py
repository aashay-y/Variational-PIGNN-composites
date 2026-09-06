"""
Master Orchestration Script — Composite Rod PI-GNN Pipeline (3-D torsion)
========================================================================
Runs some or all of the following stages in order:

  Stage 1 — Meshing   : cross-section collocation -> Delaunay -> extrude to
                        tetrahedra -> .npy files
  Stage 2 — Training  : train the PI-GNN on the rod mesh (energy minimisation)
  Stage 3 — Inference : GNN + FEniCSx FEM comparison, write VTU/CSV/TXT

Skipping logic
--------------
  --skip-meshing   : Skip Stage 1; use an existing mesh directory.
  --skip-training  : Skip Stages 1 AND 2; use an existing checkpoint.

If --skip-training is given, --skip-meshing is implied automatically.

============================================================
THE PROBLEM
============================================================

A circular composite rod (radius R, length H) with a straight prismatic petal
inclusion, CLAMPED at z = 0 and TWISTED at z = H by a tangential dead traction

    t(X) = (tau / R) * ( -(y - cy), (x - cx), 0 )

whose resultant force is zero and whose resultant moment is a pure torque
M_z = tau * pi * R^3 / 2 about the rod axis. The twist angle is an OUTPUT of the
applied torque, not a prescribed boundary condition.

Fully three-dimensional: no plane-stress or plane-strain assumption, three
displacement components per node, six independent stress components per element.

============================================================
UNIT CONVENTIONS
============================================================

JSON config (inclusion cross-section geometry)
  center_x, center_y   [mm]   inclusion centre (also the default torsion axis)
  base_radius, amplitude, num_petals, phase   petal shape parameters

Mesh / geometry
  plate_radius (R)     [mm]   rod outer radius
  height (H)           [mm]   rod length along z
  n_layers             [-]    element layers through the depth
  node coordinates     [mm]   stored as-is in nodes.npy and fed to GNN/FEM

GNN training  (train_pignn.py)
  YOUNGS_MODULUS_MATRIX      [kPa]  E of the matrix (default 1.5)
  POISSONS_RATIO_MATRIX      [-]    nu of matrix    (default 0.40)
  POISSONS_RATIO_INCLUSION   [-]    nu of inclusion (default 0.35)
  INCLUSION_RATIO            [-]    E_inc / E_mat   (default 5.0/1.5 = 3.3333)
  APPLIED_TORQUE       [kPa*mm^3]  torque about the rod axis (default 1.1).
                                    1 kPa*mm^3 = 1e-6 N*m = 1 uN*m exactly.
                                    The rim traction tau that delivers it is
                                    DERIVED from the mesh, not specified.
  Predicted displacements    [mm]   PHYSICAL — the network trains on the real
                                    material, so no post-hoc rescaling is applied.
  Stress (energy calculator) [kPa]

FEM  (fem_kernel.py, driven by R1 and by Stage 3)
  E, nu, tau                 [Pa]   config values x UNIT_TO_PA
  torque                     [N*m]  APPLIED_TORQUE x UNIT_TO_PA x 1e-9
  coordinates                [m]    mesh mm x 1e-3
  displacements              [m]    -> x1000 to report mm
  stress                     [Pa]   -> x PA_TO_UNIT to report the config unit

Derived outputs
  Applied torque             [kPa*mm^3]
  Twist angle                [deg]
  Warping uz                 [mm]

Mesh-density parameters (dimensionless ratios, not lengths)
  min_density_factor   [-]   min_spacing = inclusion char_length x factor
  max_density_factor   [-]   max_spacing = 2 x plate_radius x factor
  matrix_near_field    [-]   near-field zone = inclusion char_length x factor
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse the command-line options for the three-stage pipeline.

    Returns:
        argparse.Namespace: Geometry, meshing, training and inference options.
            Stage selection is carried by the ``skip_meshing`` and
            ``skip_training`` flags.
    """
    parser = argparse.ArgumentParser(
        description="Composite rod PI-GNN pipeline: mesh -> train -> FEM compare.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run from the petal config (mesh + train + compare)
  python run_pipeline.py --config petal_config.json \\
      --plate-radius 1.0 --height 2.0 --n-layers 10 \\
      --mesh-dir mesh_rod_petal --checkpoint-dir rod_checkpoints \\
      --results-dir rod_results --comparison-dir rod_comparison --epochs 30000

  # Reuse an existing mesh
  python run_pipeline.py --skip-meshing --mesh-dir mesh_rod_petal \\
      --epochs 30000

  # Compare only, from an existing checkpoint
  python run_pipeline.py --skip-training --mesh-dir mesh_rod_petal \\
      --checkpoint rod_checkpoints/model_best.pt

  # Mesh only (inspect before committing the epoch budget)
  python run_pipeline.py --config petal_config.json \\
      --mesh-dir mesh_rod_petal --skip-training --skip-comparison
""",
    )

    skip_grp = parser.add_argument_group("Skip flags")
    skip_grp.add_argument("--skip-meshing", action="store_true",
                          help="Skip Stage 1 and use an existing mesh directory.")
    skip_grp.add_argument("--skip-training", action="store_true",
                          help="Skip Stages 1 and 2 and use an existing checkpoint.")
    skip_grp.add_argument("--skip-comparison", action="store_true",
                          help="Skip Stage 3 (no FEM comparison).")

    parser.add_argument("--material-model", type=str, default="NH",
                        choices=["NH", "LE", "nh", "le"],
                        help="'NH' compressible Neo-Hookean, finite strain (default; "
                             "the physical choice at this load) or 'LE' small-strain "
                             "linear elasticity (reference option only — it cannot "
                             "represent the finite rotation this problem produces).")

    dirs_grp = parser.add_argument_group("Directory / path configuration")
    dirs_grp.add_argument("--config", type=Path, default=Path("petal_config.json"),
                          help="JSON inclusion cross-section config (Stage 1).")
    dirs_grp.add_argument("--mesh-dir", type=Path, default=Path("mesh_rod_petal"),
                          help="Mesh case directory (written by Stage 1, read by 2/3).")
    dirs_grp.add_argument("--checkpoint-dir", type=Path, default=Path("rod_checkpoints"),
                          help="Directory for training checkpoints.")
    dirs_grp.add_argument("--results-dir", type=Path, default=Path("rod_results"),
                          help="Directory for training result plots.")
    dirs_grp.add_argument("--comparison-dir", type=Path, default=Path("rod_comparison"),
                          help="Directory for Stage 3 outputs.")
    dirs_grp.add_argument("--checkpoint", type=Path, default=None,
                          help="Explicit checkpoint (.pt) for Stage 3. Defaults to "
                               "model_best.pt in --checkpoint-dir.")

    mesh_grp = parser.add_argument_group("Meshing parameters (Stage 1)")
    mesh_grp.add_argument("--plate-radius", type=float, default=1.0,
                          help="Rod outer radius R in mm (default: 1.0). Pass a "
                               "negative value to infer it from the inclusion extent.")
    mesh_grp.add_argument("--height", type=float, default=2.0,
                          help="Rod length H in mm (default: 2.0).")
    mesh_grp.add_argument("--n-layers", type=int, default=10,
                          help="Element layers through the depth (default: 10). The "
                               "generator reports dz / in-plane element size; keep it "
                               "near 1 to avoid sliver tetrahedra.")
    mesh_grp.add_argument("--matrix-near-field", type=float, default=2.0,
                          help="Matrix near-field factor (default: 2.0).")
    mesh_grp.add_argument("--inclusion-refinement", type=float, default=1.5,
                          help="Inclusion refinement factor (default: 1.5).")
    mesh_grp.add_argument("--min-density", type=float, default=0.16,
                          help="Min spacing factor at the interface (default: 0.16).")
    mesh_grp.add_argument("--max-density", type=float, default=0.075,
                          help="Max spacing factor in the far field (default: 0.075).")
    mesh_grp.add_argument("--growth-rate", type=float, default=None,
                          help="Element growth rate for the fine->coarse transition. "
                               "Default: auto (spans the near-field band).")

    train_grp = parser.add_argument_group("Training parameters (Stage 2)")
    train_grp.add_argument("--epochs", type=int, default=None,
                           help="Training epochs (default: Config.NUM_EPOCHS).")
    train_grp.add_argument("--lr", type=float, default=None,
                           help="Adam learning rate (default: Config.LEARNING_RATE).")
    train_grp.add_argument("--hidden-dim", type=int, default=None,
                           help="GNN hidden width (default: Config.HIDDEN_DIM).")
    train_grp.add_argument("--num-layers", type=int, default=None,
                           help="GNN message-passing layers (default: Config.NUM_LAYERS).")
    train_grp.add_argument("--seed", type=int, default=None,
                           help="Random seed for weight init (reproducibility).")
    train_grp.add_argument("--device", type=str, default=None,
                           choices=["cpu", "cuda"],
                           help="Device for training and inference.")

    phys_grp = parser.add_argument_group("Material / loading overrides")
    phys_grp.add_argument("--output-unit", type=str, default=None,
                          choices=["Pa", "kPa", "MPa", "GPa"],
                          help="Stress unit that --youngs-matrix is given in; --torque is "
                               "then in <unit>*mm^3.")
    phys_grp.add_argument("--youngs-matrix", type=float, default=None,
                          help="Matrix Young's modulus (default: 1.5 kPa).")
    phys_grp.add_argument("--nu-matrix", type=float, default=None,
                          help="Matrix Poisson ratio (default: 0.40).")
    phys_grp.add_argument("--nu-inclusion", type=float, default=None,
                          help="Inclusion Poisson ratio (default: 0.35).")
    phys_grp.add_argument("--inclusion-ratio", type=float, default=None,
                          help="E_inclusion / E_matrix (default: 5.0/1.5 = 3.3333).")
    phys_grp.add_argument("--torque", type=float, default=None,
                          help="APPLIED TORQUE about the rod axis, in "
                               "<output-unit>*mm^3. With the default kPa that is "
                               "kPa*mm^3, which equals exactly 1 micro-newton-metre "
                               "(1 kPa*mm^3 = 1e-6 N*m = 1 uN*m). Default: 0.25 uN*m, "
                               "calibrated for ~24 deg of twist on the shipped rod. "
                               "The rim traction that delivers it is derived from "
                               "the mesh at run time.")

    inf_grp = parser.add_argument_group("Inference parameters (Stage 3)")
    inf_grp.add_argument("--threads", type=int, default=None, metavar="N",
                         help="Pin BOTH the GNN and the FEM to N CPU cores for a fair "
                              "same-core timing comparison.")
    return parser.parse_args()


def _banner(title: str, width: int = 70) -> None:
    """Print a full-width, centred stage banner.

    Args:
        title: Text to centre inside the rule.
        width: Total banner width in characters.
    """
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width)


def _section(title: str) -> None:
    """Print a minor section heading.

    Args:
        title: Text of the heading.
    """
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


REQUIRED_MESH_FILES = [
    "nodes.npy", "elements.npy", "node_features.npy",
    "loading_surface_facets.npy", "node_topology.npy",
    "element_material_ids.npy",
    # Carries the rod axis/radius/height that the traction field is defined
    # from; without it neither the trainer nor the FEM can build the load.
    "mesh_summary.json",
]


# ============================================================
# STAGE 1 — MESHING
# ============================================================

def run_meshing(args: argparse.Namespace) -> None:
    """Run Stage 1: build the composite rod mesh and its graph arrays.

    Args:
        args: Parsed CLI options; supplies the inclusion config path, the rod
            geometry and the meshing density factors.

    Raises:
        FileNotFoundError: If the inclusion config JSON does not exist.
    """
    _banner("STAGE 1: MESH GENERATION")

    if not args.config.exists():
        raise FileNotFoundError(
            f"Config file not found: {args.config}\n"
            "Provide --config pointing to a valid JSON inclusion config."
        )

    from meshing_pipeline import run_meshing_pipeline

    # A negative radius means "infer from the inclusion extent".
    plate_radius = None if (args.plate_radius or 0) < 0 else args.plate_radius

    run_meshing_pipeline(
        config_file=str(args.config),
        plate_radius=plate_radius,
        height=args.height,
        n_layers=args.n_layers,
        matrix_near_field_factor=args.matrix_near_field,
        inclusion_refinement_factor=args.inclusion_refinement,
        min_density_factor=args.min_density,
        max_density_factor=args.max_density,
        growth_rate=args.growth_rate,
        output_dir=str(args.mesh_dir),
        run_mesh_processor=True,
    )

    print(f"\n  Mesh generated in: {args.mesh_dir}")

    missing = [f for f in REQUIRED_MESH_FILES if not (args.mesh_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Meshing completed but the following files are missing in {args.mesh_dir}:\n"
            + "\n".join(f"  {f}" for f in missing)
        )
    print("  All required mesh files verified.")


# ============================================================
# CONFIG OVERRIDES
# ============================================================

def apply_config_overrides(args: argparse.Namespace):
    """
    Patch the trainer Config from the CLI.

    MUST run before Stage 3 imports evaluate_gnn_vs_fem:
    that module snapshots UNIT_TO_PA, FEM_E_MATRIX, FEM_E_INCLUSION, FEM_TAU and
    FIXED_INCLUSION_RATIO into module-level constants **at import time**, reading
    them off TrainConfig. Patching the config after that import would leave the
    FEM reference on the old material properties while the GNN trained on the new
    ones — a silent, total mismatch. Calling this from main() also makes the
    physics flags work under --skip-training.
    """
    import train_pignn as _trainer_mod
    cfg = _trainer_mod.Config

    cfg.MATERIAL_MODEL  = args.material_model.upper()
    cfg.DATA_DIR        = str(args.mesh_dir)
    cfg.CHECKPOINT_DIR  = str(args.checkpoint_dir)
    cfg.RESULTS_DIR     = str(args.results_dir)

    if args.epochs       is not None:  cfg.NUM_EPOCHS      = args.epochs
    if args.lr           is not None:  cfg.LEARNING_RATE   = args.lr
    if args.hidden_dim   is not None:  cfg.HIDDEN_DIM      = args.hidden_dim
    if args.num_layers   is not None:  cfg.NUM_LAYERS      = args.num_layers
    if args.inclusion_ratio is not None: cfg.INCLUSION_RATIO = args.inclusion_ratio
    if args.device       is not None:  cfg.DEVICE          = args.device

    # Material / loading. The unit must be set first: the modulus and traction
    # are expressed in it.
    if args.output_unit   is not None: cfg.POSTPROC_OUTPUT_UNIT   = args.output_unit
    if args.youngs_matrix is not None: cfg.YOUNGS_MODULUS_MATRIX  = args.youngs_matrix
    if args.nu_matrix     is not None: cfg.POISSONS_RATIO_MATRIX  = args.nu_matrix
    if args.nu_inclusion  is not None: cfg.POISSONS_RATIO_INCLUSION = args.nu_inclusion
    if args.torque        is not None: cfg.APPLIED_TORQUE         = args.torque

    return cfg


# ============================================================
# STAGE 2 — TRAINING
# ============================================================

def run_training(args: argparse.Namespace) -> None:
    """Run Stage 2: train the PI-GNN by minimising the total potential energy.

    The rim traction that delivers the requested torque on this particular mesh
    is derived here and written onto the trainer ``Config`` before the model is
    built, because the energy calculator, the output scaling and the checkpoint
    all report it.

    Args:
        args: Parsed CLI options; the trainer ``Config`` has already been
            patched with the resolved paths by ``apply_config_overrides``.
    """
    _banner("STAGE 2: GNN TRAINING")

    import train_pignn as _trainer_mod
    cfg = _trainer_mod.Config   # already patched in main()

    unit = cfg.POSTPROC_OUTPUT_UNIT
    print(f"  Material model  : {cfg.MATERIAL_MODEL}")
    print(f"  Mesh dir        : {cfg.DATA_DIR}")
    print(f"  Checkpoint dir  : {cfg.CHECKPOINT_DIR}")
    print(f"  Results dir     : {cfg.RESULTS_DIR}")
    print(f"  Epochs          : {cfg.NUM_EPOCHS}")
    print(f"  Learning rate   : {cfg.LEARNING_RATE}")
    print(f"  Hidden dim      : {cfg.HIDDEN_DIM}")
    print(f"  GNN layers      : {cfg.NUM_LAYERS}")
    print(f"  Inclusion ratio : {cfg.INCLUSION_RATIO}")
    print(f"  E_matrix        : {cfg.YOUNGS_MODULUS_MATRIX} {unit}")
    print(f"  E_inclusion     : {cfg.YOUNGS_MODULUS_MATRIX * cfg.INCLUSION_RATIO} {unit}")
    print(f"  nu mat / inc    : {cfg.POISSONS_RATIO_MATRIX} / {cfg.POISSONS_RATIO_INCLUSION}")
    print(f"  Applied torque  : {cfg.APPLIED_TORQUE} {unit}*mm^3"
          + ("  (= uN*m)" if str(unit).upper() == 'KPA' else ""))
    print(f"  Device          : {cfg.DEVICE}")

    import torch

    # Seed BEFORE the model is constructed so weight init is reproducible.
    if args.seed is not None:
        import random as _random
        import numpy as _np
        _random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"  Random seed     : {args.seed}")

    _trainer_mod.setup_cpu_limits(cfg)

    mesh_info = _trainer_mod.load_mesh_summary(cfg.DATA_DIR)
    nodes, elements, features, loading_facets, topology, element_material_ids = \
        _trainer_mod.load_mesh_data(cfg.DATA_DIR)

    axis = (mesh_info["axis_x"], mesh_info["axis_y"])
    R, H = mesh_info["plate_radius"], mesh_info["height"]
    # Recorded on Config so effective_output_displacement_scale sizes the
    # characteristic displacement from the actual rod, not a guess.
    cfg.ROD_HEIGHT = H
    print(f"  Rod geometry    : R={R}, H={H}, axis=({axis[0]:.6f}, {axis[1]:.6f})")

    element_volumes = _trainer_mod.MeshGeometry.compute_element_volumes(nodes, elements)

    # The user specifies a TORQUE; solve for the rim traction that delivers it
    # on this mesh and record it on the config (the energy calculator, the
    # output scaling and the checkpoint all report it).
    cfg.TRACTION_MAGNITUDE, _ = _trainer_mod.MeshGeometry.traction_for_torque(
        nodes, loading_facets, cfg.APPLIED_TORQUE, axis, R)
    nodal_load, M_z = _trainer_mod.MeshGeometry.compute_nodal_load(
        nodes, loading_facets, cfg.TRACTION_MAGNITUDE, axis, R)
    bc_handler = _trainer_mod.BoundaryConditions(nodes, features)

    ref_coords        = torch.tensor(nodes,           dtype=torch.float64, device=cfg.DEVICE)
    elements_tensor   = torch.tensor(elements,        dtype=torch.long,    device=cfg.DEVICE)
    element_volumes_t = torch.tensor(element_volumes, dtype=torch.float64, device=cfg.DEVICE)
    nodal_load_t      = torch.tensor(nodal_load,      dtype=torch.float64, device=cfg.DEVICE)

    graph_data = _trainer_mod.build_graph_data(
        nodes, elements, features, topology).to(cfg.DEVICE)

    energy_calc = _trainer_mod.CompositeEnergyCalculator(
        cfg.YOUNGS_MODULUS_MATRIX,
        cfg.POISSONS_RATIO_MATRIX,
        cfg.POISSONS_RATIO_INCLUSION,
        element_material_ids,
        material_model=cfg.MATERIAL_MODEL,
        output_unit=unit,
    )

    input_dim = graph_data.x.shape[1]
    model = _trainer_mod.DisplacementGNN(
        input_dim=input_dim,
        hidden_dim=cfg.HIDDEN_DIM,
        num_layers=cfg.NUM_LAYERS,
        output_scale=_trainer_mod.effective_output_displacement_scale(
            cfg, char_length=H),
    ).double().to(cfg.DEVICE)

    trainer = _trainer_mod.Trainer(
        model, energy_calc, bc_handler, graph_data,
        ref_coords, elements_tensor, element_volumes_t, nodal_load_t,
        nodes, elements, element_material_ids, cfg, mesh_info=mesh_info,
    )

    trainer.train()

    _trainer_mod.visualize_results(
        model, graph_data, bc_handler, nodes, elements,
        element_material_ids, trainer.history, cfg, mesh_info,
        inclusion_ratio=float(cfg.INCLUSION_RATIO),
    )

    print(f"\n  Training complete.  Checkpoints in: {cfg.CHECKPOINT_DIR}")


# ============================================================
# STAGE 3 — INFERENCE / COMPARISON
# ============================================================

def run_comparison(args: argparse.Namespace, material_model: str) -> None:
    """Run Stage 3: GNN inference and the reference FEM comparison.

    Args:
        args: Parsed CLI options; ``args.threads`` pins both solvers to the
            same core budget for a fair timing comparison.
        material_model: ``'NH'`` or ``'LE'``, selecting the constitutive model
            the FEM reference is solved with.
    """
    _banner("STAGE 3: GNN INFERENCE + FEM COMPARISON")

    # Pin the CPU core budget BEFORE importing the comparison module: OpenMP,
    # the BLAS backends and MUMPS latch their thread count at library-init time,
    # which happens the moment dolfinx/PETSc is imported (inside _cmp below).
    # Setting it here guarantees both the GNN and FEM solves run on N cores for
    # a fair same-hardware timing. Training already ran (Stage 2), so this
    # cannot affect it.
    if getattr(args, "threads", None):
        import os
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                   "COMPOSITE_NUM_THREADS"):
            os.environ[_v] = str(args.threads)

    import evaluate_gnn_vs_fem as _cmp

    # Patch TrainConfig so the comparison module picks up the right paths.
    _cmp.TrainConfig.DATA_DIR       = str(args.mesh_dir)
    _cmp.TrainConfig.CHECKPOINT_DIR = str(args.checkpoint_dir)
    _cmp.TrainConfig.RESULTS_DIR    = str(args.results_dir)
    _cmp.TrainConfig.MATERIAL_MODEL = material_model
    _cmp.MATERIAL_MODEL             = material_model

    if args.device:
        device_str = args.device
    else:
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    _cmp.run_inference(
        model_path=args.checkpoint,
        input_dir=args.mesh_dir,
        output_dir=args.comparison_dir,
        device_str=device_str,
        material_model_override=material_model,
    )

    print(f"\n  Comparison outputs in: {args.comparison_dir}")


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_mesh_dir(mesh_dir: Path) -> None:
    """Ensure all required mesh files exist in mesh_dir."""
    missing = [f for f in REQUIRED_MESH_FILES if not (mesh_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Mesh directory '{mesh_dir}' is missing required files:\n"
            + "\n".join(f"  {f}" for f in missing)
            + "\nRun without --skip-meshing to generate them, or point --mesh-dir at "
              "an existing mesh directory."
        )


def _validate_checkpoint(checkpoint: Path | None, checkpoint_dir: Path) -> None:
    """Ensure at least one checkpoint is resolvable."""
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return

    candidates = (list(checkpoint_dir.glob("model_best.pt"))
                  + list(checkpoint_dir.glob("model_epoch_*.pt")))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in '{checkpoint_dir}'. "
            "Run without --skip-training to create them."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Drive the pipeline: meshing, training, then FEM comparison.

    Stages are skipped according to ``--skip-meshing`` / ``--skip-training``;
    skipping training implies skipping meshing, so an existing mesh and
    checkpoint are reused.

    Returns:
        int: Process exit status, 0 on success and non-zero if a stage failed.
    """
    args = parse_args()

    # --skip-training implies --skip-meshing
    if args.skip_training:
        args.skip_meshing = True

    mat_model = args.material_model.upper()

    # Before ANY stage — see apply_config_overrides for why the ordering matters.
    apply_config_overrides(args)

    _banner("COMPOSITE ROD PI-GNN PIPELINE (3-D TORSION)", width=70)
    print(f"\n  Material model  : {mat_model}")
    print(f"  Problem type    : 3-D composite rod (2-phase), clamped-twisted")
    print(f"  Mesh dir        : {args.mesh_dir}")
    print(f"  Checkpoint dir  : {args.checkpoint_dir}")
    print(f"  Results dir     : {args.results_dir}")
    print(f"  Comparison dir  : {args.comparison_dir}")
    print(f"  Skip meshing    : {args.skip_meshing}")
    print(f"  Skip training   : {args.skip_training}")
    print(f"  Skip comparison : {args.skip_comparison}")

    if mat_model != "NH":
        print("\n  NOTE: --material-model LE selected. The rod twists tens of "
              "degrees\n        under this load, where small-strain kinematics "
              "cannot represent\n        finite rotation. Use LE as a code-to-code "
              "check only.")

    t0 = time.time()

    # -- Stage 1 --------------------------------------------
    if not args.skip_meshing:
        _section("Stage 1 - Meshing")
        try:
            run_meshing(args)
        except Exception as exc:
            print(f"\n[ERROR] Meshing failed: {exc}")
            import traceback; traceback.print_exc()
            return 1
    else:
        _section("Stage 1 - Meshing (SKIPPED)")
        print(f"  Using existing mesh in: {args.mesh_dir}")
        try:
            _validate_mesh_dir(args.mesh_dir)
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            return 1

    # -- Stage 2 --------------------------------------------
    if not args.skip_training:
        _section("Stage 2 - Training")
        try:
            run_training(args)
        except Exception as exc:
            print(f"\n[ERROR] Training failed: {exc}")
            import traceback; traceback.print_exc()
            return 1
    else:
        _section("Stage 2 - Training (SKIPPED)")
        print(f"  Using checkpoints in: {args.checkpoint_dir}")
        if args.checkpoint:
            print(f"  Explicit checkpoint: {args.checkpoint}")
        try:
            _validate_checkpoint(args.checkpoint, args.checkpoint_dir)
        except FileNotFoundError as exc:
            print(f"\n[ERROR] {exc}")
            return 1

    # -- Stage 3 --------------------------------------------
    if not args.skip_comparison:
        _section("Stage 3 - GNN vs FEM Comparison")
        try:
            run_comparison(args, mat_model)
        except Exception as exc:
            print(f"\n[ERROR] Comparison failed: {exc}")
            import traceback; traceback.print_exc()
            return 1
    else:
        _section("Stage 3 - Comparison (SKIPPED)")

    elapsed = time.time() - t0
    _banner("PIPELINE COMPLETE", width=70)
    print(f"\n  Total elapsed time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"\n  Outputs:")
    if not args.skip_meshing:
        print(f"    Mesh        -> {args.mesh_dir}/")
    if not args.skip_training:
        print(f"    Checkpoints -> {args.checkpoint_dir}/")
        print(f"    Results     -> {args.results_dir}/")
    if not args.skip_comparison:
        print(f"    Comparison  -> {args.comparison_dir}/")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
