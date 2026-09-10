"""
Master Orchestration Script — Hole-Plate PI-GNN Pipeline (2-D plane stress)
===========================================================================
Runs some or all of the following stages in order:

  Stage 1 — Meshing   : collocation -> Delaunay -> carve out the voids ->
                        .npy files + composite_mesh.vtu
  Stage 2 — Training  : train the PI-GNN on the plate mesh (energy minimisation)
  Stage 3 — Inference : GNN + FEniCSx FEM comparison, write VTU/CSV/TXT

Skipping logic
--------------
  --skip-meshing   : Skip Stage 1; use an existing mesh directory.
  --skip-training  : Skip Stages 1 AND 2; use an existing checkpoint.

If --skip-training is given, --skip-meshing is implied automatically.

============================================================
THE PROBLEM
============================================================

A square plate (side L) with circular hole(s) cut out of it, held by a ROLLER on
x = x_min (ux = 0) plus a single PIN at the top-left corner (uy = 0), and loaded
on x = x_max by the uniform dead traction

    t = (T, 0)

Every hole rim is an exterior traction-free free surface — the void interiors
are removed at mesh time, so the plate is a single homogeneous matrix phase.

Two-dimensional PLANE STRESS throughout: two displacement components per node,
three independent in-plane stress components per element, sigma_33 = 0 imposed
by the constitutive law (LE, reduced lambda_ps) or by the out-of-plane stretch
F33 (NH, rigorous).

============================================================
UNIT CONVENTIONS
============================================================

JSON config (hole geometry)
  center_x, center_y   [mm]   hole centre
  radius               [mm]   hole radius

Mesh / geometry
  plate_size (L)       [mm]   square plate side length
  node coordinates     [mm]   stored as-is in nodes.npy and fed to GNN/FEM

GNN training  (train_pignn.py)
  YOUNGS_MODULUS_MATRIX      [MPa]  E of the matrix (default 210e3 = 210 GPa)
  POISSONS_RATIO_MATRIX      [-]    nu of the matrix (default 0.3)
  TRACTION_MAGNITUDE         [MPa]  uniform right-edge traction (default 1.0)
  Predicted displacements    [mm]   PHYSICAL — the network trains on the real
                                    material, so no post-hoc rescaling is applied.
  Stress (energy calculator) [MPa]

FEM  (fem_kernel.py, driven by fem_reference_solver and by Stage 3)
  E, nu, T                   [Pa]   config values x UNIT_TO_PA
  coordinates                [m]    mesh mm x 1e-3
  displacements              [m]    -> x1000 to report mm
  stress                     [Pa]   -> x PA_TO_UNIT to report the config unit

Derived outputs
  Applied force Fx           [MPa*mm per unit thickness]
  Kt = max(sigma_xx) / T     [-]    stress concentration at the hole rim

Mesh-density parameters (dimensionless ratios, not lengths)
  min_density_factor   [-]   min_spacing = hole char_length x factor
  max_density_factor   [-]   max_spacing = plate_size x factor
  matrix_near_field    [-]   near-field zone = hole char_length x factor
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
        description="Hole-plate PI-GNN pipeline: mesh -> train -> FEM compare.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run from the hole config (mesh + train + compare)
  python run_pipeline.py --config hole_config.json --plate-size 20 \\
      --mesh-dir mesh_plate_hole --checkpoint-dir plate_checkpoints \\
      --results-dir plate_results --comparison-dir plate_comparison \\
      --epochs 20000 --seed 0

  # Reuse an existing mesh
  python run_pipeline.py --skip-meshing --mesh-dir mesh_plate_hole \\
      --epochs 20000

  # Compare only, from an existing checkpoint
  python run_pipeline.py --skip-training --mesh-dir mesh_plate_hole \\
      --checkpoint plate_checkpoints/model_best.pt

  # Mesh only (inspect before committing the epoch budget)
  python run_pipeline.py --config hole_config.json \\
      --mesh-dir mesh_plate_hole --skip-training --skip-comparison
""",
    )

    skip_grp = parser.add_argument_group("Skip flags")
    skip_grp.add_argument("--skip-meshing", action="store_true",
                          help="Skip Stage 1 and use an existing mesh directory.")
    skip_grp.add_argument("--skip-training", action="store_true",
                          help="Skip Stages 1 and 2 and use an existing checkpoint.")
    skip_grp.add_argument("--skip-comparison", action="store_true",
                          help="Skip Stage 3 (no FEM comparison).")

    parser.add_argument("--material-model", type=str, default="LE",
                        choices=["LE", "NH", "le", "nh"],
                        help="'LE' small-strain linear elasticity, plane stress "
                             "(default; the physical choice at this load) or 'NH' "
                             "compressible Neo-Hookean, finite strain with a "
                             "rigorous plane-stress condensation.")

    dirs_grp = parser.add_argument_group("Directory / path configuration")
    dirs_grp.add_argument("--config", type=Path, default=Path("hole_config.json"),
                          help="JSON hole geometry config (Stage 1).")
    dirs_grp.add_argument("--mesh-dir", type=Path, default=Path("mesh_plate_hole"),
                          help="Mesh case directory (written by Stage 1, read by 2/3).")
    dirs_grp.add_argument("--checkpoint-dir", type=Path, default=Path("plate_checkpoints"),
                          help="Directory for training checkpoints.")
    dirs_grp.add_argument("--results-dir", type=Path, default=Path("plate_results"),
                          help="Directory for training result plots.")
    dirs_grp.add_argument("--comparison-dir", type=Path, default=Path("plate_comparison"),
                          help="Directory for Stage 3 outputs.")
    dirs_grp.add_argument("--checkpoint", type=Path, default=None,
                          help="Explicit checkpoint (.pt) for Stage 3. Defaults to "
                               "model_best.pt in --checkpoint-dir.")

    mesh_grp = parser.add_argument_group("Meshing parameters (Stage 1)")
    mesh_grp.add_argument("--plate-size", type=float, default=None,
                          help="Square plate side length in mm. Auto-inferred "
                               "from the hole config when omitted.")
    mesh_grp.add_argument("--matrix-near-field", type=float, default=2.0,
                          help="Matrix near-field factor (default: 2.0).")
    mesh_grp.add_argument("--inclusion-refinement", type=float, default=1.5,
                          help="Hole-interior refinement factor (default: 1.5). "
                               "Those points are seeded so the triangulation is "
                               "well graded up to the rim, then removed with the "
                               "void.")
    mesh_grp.add_argument("--min-density", type=float, default=0.02,
                          help="Min spacing factor at the hole rim (default: 0.02).")
    mesh_grp.add_argument("--max-density", type=float, default=0.03,
                          help="Max spacing factor in the far field (default: 0.03).")
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
                          help="Stress unit that --youngs-matrix and --traction are "
                               "given in (default: MPa).")
    phys_grp.add_argument("--youngs-matrix", type=float, default=None,
                          help="Matrix Young's modulus (default: 210e3 MPa).")
    phys_grp.add_argument("--nu-matrix", type=float, default=None,
                          help="Matrix Poisson ratio (default: 0.3).")
    phys_grp.add_argument("--traction", type=float, default=None,
                          help="Uniform right-edge traction, in <output-unit> "
                               "(default: 1.0 MPa, a nominal strain of T/E = 4.8e-6 "
                               "against the default steel).")
    phys_grp.add_argument("--output-scale", type=float, default=None,
                          help="OUTPUT_DISPLACEMENT_SCALE, the fixed multiplier on "
                               "the network output. Default: auto = (T/E) x "
                               "plate_size. Pass 1.0 to disable.")

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
    # Carries the plate size that OUTPUT_DISPLACEMENT_SCALE is derived from;
    # without it the trainer cannot size its output conditioning.
    "mesh_summary.json",
]


# ============================================================
# STAGE 1 — MESHING
# ============================================================

def run_meshing(args: argparse.Namespace) -> None:
    """Run Stage 1: build the hole-plate mesh and its graph arrays.

    Args:
        args: Parsed CLI options; supplies the hole config path, the plate size
            and the meshing density factors.

    Raises:
        FileNotFoundError: If the hole config JSON does not exist.
    """
    _banner("STAGE 1: MESH GENERATION")

    if not args.config.exists():
        raise FileNotFoundError(
            f"Config file not found: {args.config}\n"
            "Provide --config pointing to a valid JSON hole config."
        )

    from meshing_pipeline import run_meshing_pipeline

    run_meshing_pipeline(
        config_file=str(args.config),
        plate_size=args.plate_size,
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

    MUST run before Stage 3 imports evaluate_gnn_vs_fem: that module snapshots
    UNIT_TO_PA, FEM_E_MATRIX, FEM_NU_MATRIX and FEM_SIGMA into module-level
    constants **at import time**, reading them off TrainConfig. Patching the
    config after that import would leave the FEM reference on the old material
    properties while the GNN trained on the new ones — a silent, total mismatch.
    Calling this from main() also makes the physics flags work under
    --skip-training. Do not move that call.
    """
    import train_pignn as _trainer_mod
    cfg = _trainer_mod.Config

    cfg.MATERIAL_MODEL  = args.material_model.upper()
    cfg.DATA_DIR        = str(args.mesh_dir)
    cfg.CHECKPOINT_DIR  = str(args.checkpoint_dir)
    cfg.RESULTS_DIR     = str(args.results_dir)

    if args.epochs     is not None:  cfg.NUM_EPOCHS    = args.epochs
    if args.lr         is not None:  cfg.LEARNING_RATE = args.lr
    if args.hidden_dim is not None:  cfg.HIDDEN_DIM    = args.hidden_dim
    if args.num_layers is not None:  cfg.NUM_LAYERS    = args.num_layers
    if args.device     is not None:  cfg.DEVICE        = args.device

    # Material / loading. The unit must be set first: the modulus and traction
    # are expressed in it.
    if args.output_unit   is not None: cfg.POSTPROC_OUTPUT_UNIT  = args.output_unit
    if args.youngs_matrix is not None: cfg.YOUNGS_MODULUS_MATRIX = args.youngs_matrix
    if args.nu_matrix     is not None: cfg.POISSONS_RATIO_MATRIX = args.nu_matrix
    if args.traction      is not None: cfg.TRACTION_MAGNITUDE    = args.traction
    if args.output_scale  is not None: cfg.OUTPUT_DISPLACEMENT_SCALE = args.output_scale

    return cfg


# ============================================================
# STAGE 2 — TRAINING
# ============================================================

def run_training(args: argparse.Namespace) -> None:
    """Run Stage 2: train the PI-GNN by minimising the total potential energy.

    The consistent nodal load of the right-edge traction is assembled here and
    handed to the trainer, and the plate size is written onto the ``Config``
    before the model is built because ``OUTPUT_DISPLACEMENT_SCALE`` is sized
    from it.

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
    print(f"  E_matrix        : {cfg.YOUNGS_MODULUS_MATRIX} {unit}")
    print(f"  nu_matrix       : {cfg.POISSONS_RATIO_MATRIX}")
    print(f"  Traction        : {cfg.TRACTION_MAGNITUDE} {unit}")
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

    import numpy as np
    L = mesh_info.get('plate_size')
    if not L:
        L = float(max(np.ptp(nodes[:, 0]), np.ptp(nodes[:, 1])))
    # Recorded on Config so effective_output_displacement_scale sizes the
    # characteristic displacement from the actual plate, not a guess.
    cfg.PLATE_SIZE = L
    print(f"  Plate geometry  : L={L}, holes={len(mesh_info.get('holes', []) or [])}")

    element_areas = _trainer_mod.MeshGeometry.compute_element_areas(nodes, elements)
    _ = _trainer_mod.MeshGeometry.compute_edge_lengths(nodes, loading_facets)
    nodal_load, F_x = _trainer_mod.MeshGeometry.compute_nodal_load(
        nodes, loading_facets, cfg.TRACTION_MAGNITUDE)
    bc_handler = _trainer_mod.BoundaryConditions(nodes, features)

    ref_coords           = torch.tensor(nodes,         dtype=torch.float64, device=cfg.DEVICE)
    elements_tensor      = torch.tensor(elements,      dtype=torch.long,    device=cfg.DEVICE)
    element_areas_tensor = torch.tensor(element_areas, dtype=torch.float64, device=cfg.DEVICE)
    nodal_load_t         = torch.tensor(nodal_load,    dtype=torch.float64, device=cfg.DEVICE)

    graph_data = _trainer_mod.build_graph_data(
        nodes, elements, features, topology).to(cfg.DEVICE)

    energy_calc = _trainer_mod.CompositeEnergyCalculator(
        cfg.YOUNGS_MODULUS_MATRIX,
        cfg.POISSONS_RATIO_MATRIX,
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
            cfg, char_length=L),
    ).double().to(cfg.DEVICE)

    trainer = _trainer_mod.Trainer(
        model, energy_calc, bc_handler, graph_data,
        ref_coords, elements_tensor, element_areas_tensor, nodal_load_t,
        nodes, elements, element_material_ids, cfg, mesh_info=mesh_info,
    )

    trainer.train()

    _trainer_mod.visualize_results(
        model, graph_data, bc_handler, nodes, elements,
        element_material_ids, trainer.history, cfg, mesh_info,
    )
    _trainer_mod.visualize_stress_field(
        model, graph_data, bc_handler, nodes, elements, element_material_ids,
        energy_calc, ref_coords, elements_tensor, element_areas_tensor, cfg,
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
        material_model: ``'LE'`` or ``'NH'``, selecting the constitutive model
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

    _banner("HOLE-PLATE PI-GNN PIPELINE (2-D PLANE STRESS)", width=70)
    print(f"\n  Material model  : {mat_model}")
    print(f"  Problem type    : 2-D plate with circular hole(s), single matrix phase")
    print(f"  Mesh dir        : {args.mesh_dir}")
    print(f"  Checkpoint dir  : {args.checkpoint_dir}")
    print(f"  Results dir     : {args.results_dir}")
    print(f"  Comparison dir  : {args.comparison_dir}")
    print(f"  Skip meshing    : {args.skip_meshing}")
    print(f"  Skip training   : {args.skip_training}")
    print(f"  Skip comparison : {args.skip_comparison}")

    if mat_model == "NH":
        print("\n  NOTE: --material-model NH selected. At the default load the "
              "nominal\n        strain is ~5e-4, where NH and LE agree to within "
              "O(strain^2).\n        NH is exercised here as a code path, not "
              "because the physics\n        demands it.")

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
