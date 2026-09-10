# 2-D Plate with a Central Hole — PI-GNN + FEM

A physics-informed Graph Neural Network that predicts the displacement and
stress field of a **2-D plate with a circular hole** under a uniform edge
traction, by minimising the total potential energy, validated against a FEniCSx
FEM reference solving the identical problem.

The plate is held by a **roller** on its left edge and pulled by a **uniform
traction** on its right edge. The material is linear-elastic steel under **plane
stress**, and the load is sized so the nominal strain stays inside the
linear-elastic range.

---

## The problem

```
        uy = 0 (pin)
          |
          v
        +-----------------------------+
        |  o                          | -->
        |          (  )               | -->  uniform traction T = 1 MPa in +x
        |       free hole rim         | -->
        |  o                          | -->
        +-----------------------------+
         ^
         ux = 0 on the whole left edge (roller; uy free)
```

| | |
|---|---|
| Geometry | square plate, side `L = 20 mm` |
| Hole | circular void, radius `r = 1.0 mm`, centred at `(10, 10) mm` |
| `x = x_min` | roller, `ux = 0` (`uy` free) |
| top-left corner | pin, `uy = 0` |
| `x = x_max` | **uniform traction `T = 1 MPa`** in `+x` |
| hole rim, `y` edges | free (traction-free) |
| Material | steel, `E = 210 GPa`, `ν = 0.30`, **plane stress** |
| Discretisation | linear triangles, graded towards the hole rim |

### The holes are voids, not inclusions

The shapes in the JSON config are treated as **voids**: the interior triangles
are removed at mesh time (orphaned nodes dropped and the connectivity
renumbered) and each hole boundary becomes an exterior traction-free **free
surface**. Everything that remains is a single matrix phase, so there is no
material interface anywhere in the domain and no inclusion contrast to set.

Only **circular** holes are supported here (irregular voids are out of scope);
the `inclusion_shapes.py` abstraction is kept so more shapes could plug in via
one `signed_distance()` primitive.

### The BCs remove exactly three rigid-body modes

A 2-D body has three of them. The roller edge kills the `x` translation and the
rotation; the single top-left pin kills the remaining `y` translation. Together
that is exactly enough — a **fully fixed** left edge would add a spurious
transverse restraint at the held end and change the problem.

The pin is a **vertex** (a dim-0 entity), not a facet. A facet search for it
finds nothing, which leaves `uy` with an unconstrained rigid-body mode: singular
for the LE solve, non-convergent for the NH Newton solve. Both FEM entry points
locate it with `locate_entities_boundary(domain, 0, ...)` for that reason.

### The load is a dead traction

`t = (T, 0)` is fixed in the reference frame and does not follow the material.
It is therefore **conservative**: the external work `W_ext = ∫t·u dS` is a
genuine potential, which is what makes `Π = E_int − W_ext` a real energy the GNN
can minimise, and what makes the `2·E_int/W_ext → 1` health check meaningful.

At `T = 1 MPa` against 210 GPa steel the nominal strain is `T/E = 4.8×10⁻⁶`, so
linear elasticity is squarely valid and the dead-load direction error is
negligible.

### Achieved response

| | |
|---|---|
| nominal strain `T/E` | 4.8×10⁻⁶ |
| far-field `σ_xx` | ≈ `T` = 1 MPa |
| **stress concentration `Kt = max σ_xx / T`** | **→ 3.0** (Kirsch, small hole in a wide plate) |
| area ratio `J = det F` | ≈ 1.000 (essentially area-preserving at this strain) |

`Kt` is the **headline output**: it is dimensionless, set purely by geometry,
and has a closed-form answer, so it checks that the predicted field has the
right *shape* at the rim rather than merely the right magnitude far from it.
Both solvers report it, and the training loop prints it every `PRINT_EVERY`
epochs. On a coarse mesh it under-reports (the rim is not resolved); refine
`--min-density` until it converges.

---

## Environment

Stages 1 and 2 need only the PyTorch stack; Stage 3 additionally needs FEniCSx.
Because FEniCSx is not reliably installable from PyPI, conda is the supported
path:

```bash
conda env create -f environment.yml
conda activate femml
```

`requirements.txt` lists the same pip-installable packages for an environment
that already provides FEniCSx (dolfinx, ufl, petsc4py, mpi4py):

```bash
pip install -r requirements.txt
```

Verify the environment before a long run:

```bash
python -c "import torch, torch_geometric, numpy, scipy, meshio; print('GNN stack OK')"
python -c "import dolfinx, ufl, petsc4py, mpi4py; print('FEniCSx OK')"
```

| dependency | used by |
|---|---|
| `numpy`, `scipy` | all stages (meshing, Delaunay, KD-trees) |
| `torch`, `torch_geometric` | Stage 2 training, Stage 3 GNN inference |
| `tensorboard` | training curves under `<checkpoint-dir>/runs/` |
| `dolfinx`, `ufl`, `petsc4py`, `mpi4py` | Stage 3 FEM reference, `fem_reference_solver.py` |
| `meshio` | mesh I/O (`composite_mesh.vtu`) |
| `pandas` | error CSV export |
| `matplotlib` | all figures |

> Set `COMPOSITE_TORCH_COMPILE=0` to disable `torch.compile` if your PyTorch
> build has no working inductor backend. Any compile failure already falls back
> to eager automatically, so this is only needed to silence the warning.

---

## Running

### Option A — orchestrator (recommended)

`run_pipeline.py` runs **Stage 1 mesh → Stage 2 train → Stage 3 FEM-compare**.

```bash
conda activate femml

# Full run from the hole config
python run_pipeline.py \
    --config hole_config.json --plate-size 20 \
    --min-density 0.02 --max-density 0.03 \
    --mesh-dir mesh_plate_hole \
    --checkpoint-dir plate_checkpoints \
    --results-dir plate_results \
    --comparison-dir plate_comparison \
    --material-model LE --traction 1.0 \
    --epochs 20000 --seed 0

# Reuse an existing mesh
python run_pipeline.py --skip-meshing --mesh-dir mesh_plate_hole --epochs 20000

# Compare only, from an existing checkpoint
python run_pipeline.py --skip-training --mesh-dir mesh_plate_hole \
    --checkpoint plate_checkpoints/model_best.pt

# Mesh only — inspect the mesh before committing the epoch budget
python run_pipeline.py --config hole_config.json \
    --mesh-dir mesh_plate_hole --skip-training --skip-comparison
```

`--skip-training` implies `--skip-meshing`. Run `python run_pipeline.py --help`
for all flags.

### Option B — a single stage directly

```bash
# Stage 1 — meshing only
python meshing_pipeline.py hole_config.json --plate-size 20 --output mesh_plate_hole

# Stage 2 — train (knobs read from train_pignn.py → class Config)
python train_pignn.py

# Stage 3 — GNN vs FEM comparison for a mesh dir
python evaluate_gnn_vs_fem.py \
    --input-dir mesh_plate_hole --material-model LE \
    --output-dir plate_comparison

# Standalone FEniCSx FEM only (no GNN) — writes CSV/VTU/summary
python fem_reference_solver.py --mesh-dir mesh_plate_hole \
    --output-dir plate_fem_reference --traction 1e6     # Pa
```

---

## Configuring a case

### Material and loading

Set in `train_pignn.py → class Config`, or overridden from the orchestrator CLI:

| flag | `Config` field | default |
|---|---|---|
| `--output-unit Pa\|kPa\|MPa\|GPa` | `POSTPROC_OUTPUT_UNIT` — the unit `--youngs-matrix` and `--traction` are given in | `MPa` |
| `--youngs-matrix` | `YOUNGS_MODULUS_MATRIX` | `210e3` |
| `--nu-matrix` | `POISSONS_RATIO_MATRIX` | `0.30` |
| `--traction` | `TRACTION_MAGNITUDE` — uniform right-edge traction | `1.0` |
| `--output-scale` | `OUTPUT_DISPLACEMENT_SCALE` | `None` (auto) |
| `--material-model LE\|NH` | `MATERIAL_MODEL` | `LE` |

Mesh coordinates are in **mm**; the FEM converts to SI internally
(`coordinates × 1e-3`, `stress × UNIT_TO_PA`).

> **Ordering constraint.** `evaluate_gnn_vs_fem` snapshots `UNIT_TO_PA`,
> `FEM_E_MATRIX`, `FEM_NU_MATRIX` and `FEM_SIGMA` into module-level constants
> **at import time**. All overrides are therefore applied by
> `apply_config_overrides()` called from `main()` *before* any stage imports that
> module. Patching afterwards would leave the FEM reference on the old properties
> while the GNN trained on the new ones — a silent, total mismatch. Do not move
> that call.

### Geometry

The hole geometry comes from `hole_config.json`:

```json
[ { "center_x": 10, "center_y": 10, "radius": 1.0 } ]
```

A JSON *list* holds several holes; a bare dict holds one. `--plate-size` sets
`L`; omit it to infer it from the hole bounding boxes (`max extent × 1.1`,
rounded up).

### `OUTPUT_DISPLACEMENT_SCALE` — loss conditioning

**Why it exists.** The GNN's raw output at initialisation is `O(1e-2)`, set by
weight init and *independent of the physics*. Here the true displacement is
`O(1e-4) mm` — two orders away. When the two disagree like that, the two terms of
`Π = E_int − W_ext` are wildly unbalanced at init: `E_int` dominates, the only
meaningful gradient is *"shrink u"*, and `W_ext` — the term that creates the
deformation **shape** — is numerically negligible. Training then collapses to a
near-trivial field **no matter how many epochs you run**.

**What it does.** Multiplies the network output by a fixed (non-trainable)
characteristic displacement, so the network learns `O(1)` values while the
**physical strain is unchanged**.

| value | meaning |
|-------|---------|
| `None` *(default)* | auto = `(T / E) × L`, a nominal strain times the plate size. Here `1.0/210e3 × 20 = 9.52e-5 mm`, within a factor of a few of the truth. |
| `1.0` | disabled |
| float | explicit |

Results are insensitive to the exact value; only the **order of magnitude**
matters. The scale is *not* a parameter, so it is absent from
`model_state_dict` — it is recorded in the checkpoint
(`output_displacement_scale`) and restored by `make_model` at inference.
Rebuilding a model without it would silently rescale the entire field.

---

## The mesh

```
hole_config.json
   ↓  adaptive Poisson-disk collocation, graded towards the rim
   ↓  Delaunay triangulation
   ↓  remove triangles whose centroid lies inside a hole
   ↓  drop orphaned nodes, renumber the connectivity
perforated triangular mesh — one matrix phase, free hole rims
```

Points are seeded **inside** the hole too, so the triangulation is well graded
right up to the rim before the void is carved out; those points are removed with
it. A triangle is in a hole when its **centroid** is strictly inside, which gives
a clean, boundary-conforming faceted free surface.

The mesh arrays are handed from the generator to the processor **directly, in
memory**. There is no DOLFIN-XML intermediate: it gave the mesh two on-disk
representations that could drift apart. `composite_mesh.vtu` — which the FEM
reads — is written from the same arrays the trainer loads, so the two solvers
cannot see different meshes.

Cross-section density is set by `--min-density` (spacing at the rim, × the hole
`char_length`) and `--max-density` (far-field spacing, × `plate_size`).

> **Resolve the rim.** `Kt` is the quantity most sensitive to mesh density, and a
> coarse rim under-reports it in *both* solvers, so the GNN-vs-FEM error can look
> excellent while both are wrong against Kirsch. Refine `--min-density` until the
> FEM's `Kt` stops moving before reading anything into the comparison.

The collocation sampler is **not seeded**, so regenerating a mesh moves the
nodes. Generate once and reuse it with `--skip-meshing` if you need runs to be
comparable.

---

## Outputs

### Training → `<checkpoint-dir>/`, `<results-dir>/`

* `model_best.pt` — the **lowest-Π** weights over the whole trajectory. The loss
  *is* the total potential energy, and the minimum principle makes the lowest `Π`
  the best admissible approximation, whereas the final epoch is an arbitrary
  sample from Adam's oscillation about the minimum. Stage 3 prefers this file.
* `model_epoch_*.pt`, `training_time.json`, TensorBoard logs in `runs/`
* `displacement_field_epoch_*.png` — `ux`, `uy`, reference and deformed mesh
* `stress_field_epoch_*.png` — `σ_xx`, `σ_yy`, `σ_xy`, von Mises, titled with `Kt`
* `displacement_field.png`, `stress_field.png`, `training_history.png`
  (loss, `E_int`, `W_ext`, `Kt` against the Kirsch line)

`training_time.json` separates **optimisation compute** from checkpoint and plot
I/O, so the reported training time is not inflated by figure writing:

```json
{"total_training_time_s": ..., "wall_time_s": ..., "checkpoint_io_time_s": ..., "epochs": ...}
```

### Comparison → `<comparison-dir>/<case>/`

* `combined_gnn_fem.vtu` — the mesh with every GNN/FEM/error field. Both a
  *smoothed nodal* copy (for display) and the *raw element* value (what the
  metrics are computed from) are stored for every stress component.
* `error_data.csv` — per node: coordinates, both displacement components and
  every stress field for GNN and FEM with abs/rel errors
* `summary.txt` — material/geometry/BCs, **plate response** (applied force, `Kt`,
  peak stresses, `J` range), ML params, timings, L2 errors (global +
  hole-rim band), R² (global / near-field / far-field), energy comparison,
  signed errors
* `aggregate_summary.txt` — median & mean L2 across the processed cases

### Viewing it in ParaView

Open `combined_gnn_fem.vtu` and apply **Warp By Vector** — it works with no
setup, because the file declares `GNN_displacement_mm` as the *active* point
vector (`<PointData Vectors="...">`). Without that attribute VTK registers the
displacement arrays but leaves `active_vectors = None`, and the Warp By Vector
filter has nothing to default to; that is a real trap, since the array is plainly
there in the array list and the failure looks like a data problem rather than a
metadata one.

* **Warp By Vector** → deformed plate. The displacement is `O(1e-4) mm`, so a
  *Scale Factor* of ~1e4 is needed to see anything.
* To warp by the reference solution instead, pick `FEM_displacement_mm` in the
  filter's *Vectors* dropdown.
* `GNN_von_mises_MPa` is the active scalar, so the plate is coloured sensibly
  straight away.

`fem_reference_solver.py`'s `simulation_*.vtu` and the meshing stage's
`composite_mesh.vtu` declare their active arrays the same way.

---

## Verification

Four checks, each of which fails loudly if the physics is wired up wrong.

**1. The two energy functionals are identical.** `compute_potential_energies`
takes the FEM's converged displacement field and evaluates the *GNN's* energy
functional on it. Because both discretise the same plane-stress functional on
the same mesh, comparing "GNN Π" against "FEM Π" in `summary.txt` is genuinely
the same functional evaluated on two different fields — which is what makes the
minimum principle (`Π_GNN ≥ Π_FEM`) a meaningful check rather than a comparison
of two different quantities.

**2. The minimum-principle identity holds exactly.** On the FEM's LE solution,
`2·E_int/W_ext = 1.0000`, which is the identity the training-loop health check
watches for. A converged LE *training* run reproduces it too (measured: 1.0090,
0.9966 over successive reports — Adam oscillating about the exact minimum).

**3. The requested load is the delivered load.** The consistent nodal force
vector assembles a resultant `F_x` equal to `T × L` to `0.0e+00` relative error,
with the `y` component at exactly `0` — printed at the start of every training
run. The FEM assembles the same resultant over the same discretised edge and
`summary.txt` prints both side by side.

**4. The two FEM entry points agree.** `fem_reference_solver.py` and the
Stage-3 comparison share `fem_kernel.py`, so they cannot drift apart. On the
same mesh both return `max|u| = 9.981935e-05 mm` and `Kt = 2.2777` — identical to
every printed digit.

---

## Code structure

```
.
├── run_pipeline.py              Orchestrator: mesh → train → compare (CLI-driven)
│
├── Stage 1 — Meshing
│   ├── meshing_pipeline.py      drives meshing
│   ├── collocation_generator.py adaptive points, graded towards the rim
│   ├── mesh_generator.py        Delaunay + void removal + renumbering
│   ├── mesh_processor.py        graph, facets, features, VTU, summary
│   └── inclusion_shapes.py      hole geometry (circle)
│
├── Stage 2 — Training
│   └── train_pignn.py           PI-GNN (energy minimisation); class Config
│
├── Stage 3 — Inference + FEM comparison
│   ├── evaluate_gnn_vs_fem.py   GNN vs FEM → VTU/CSV/summary
│   └── fem_reference_solver.py  standalone FEniCSx FEM → CSV/VTU
│
├── fem_kernel.py                Shared 2-D plane-stress FEM kernel (LE + NH,
│                                roller/pin/traction BCs, adaptive load stepping).
│                                Single source of truth — imported by both FEM
│                                entry points, so the standalone solver and the
│                                comparison cannot diverge.
│
├── hole_config.json             Hole geometry for the shipped case
├── environment.yml              Conda environment (includes FEniCSx)
└── requirements.txt             Pip dependencies (FEniCSx installed separately)
```

Module dependencies run strictly one way, so each stage can be read on its own:

```
inclusion_shapes ──┬─→ collocation_generator ─┐
                   ├─→ mesh_generator ────────┼─→ meshing_pipeline ──┐
                   └─→ mesh_processor ────────┘                      │
                                                                     │
fem_kernel ────────┬─→ fem_reference_solver  (standalone CLI)        ├─→ run_pipeline
                   └─→ evaluate_gnn_vs_fem ──────────────────────────┤
                                                                     │
train_pignn ─────────────────────────────────────────────────────────┘
```

`fem_reference_solver.py` is a leaf: nothing imports it. It is the standalone FEM
cross-check, run directly from the command line, and shares `fem_kernel.py` with
the Stage-3 comparison so the two solvers cannot drift apart.

The GNN is `encoder → L × GNNLayer → decoder`, predicting 2 components per node.
Each layer's message is built from the receiver state `x_i`, the **difference**
`x_j − x_i`, and the geometric edge attribute `[dx, dy, |d|]` — a directional
derivative needs both the state difference and the physical separation it is
taken over, which is what lets a layer represent a finite-difference stencil
rather than just mean-smoothing its neighbourhood. That matters most at the hole
rim, where the whole point is to resolve a steep gradient.

Boundary conditions are imposed **hard**, by masking the network output to zero
on the constrained DOFs, so they hold exactly at every epoch and no penalty term
competes with the energy.

### Plane stress — where the two models differ

| | LE | NH |
|---|---|---|
| strain measure | small strain `ε = sym(∇u)` | deformation gradient `F` |
| `σ₃₃ = 0` imposed by | the **reduced** Lamé `λ_ps = 2μν/(1−ν)` baked into the constitutive law | an out-of-plane stretch `F₃₃` carried as an extra unknown |
| Lamé `λ` used | `λ_ps` | the **full 3-D** `λ = Eν/((1+ν)(1−2ν))` |

Using `λ_ps` inside the NH model would **double-count** the plane-stress
correction, because that reduction is produced *by* the `F₃₃` condensation. The
GNN energy (`train_pignn.py`) and both FEM paths share this exact formulation
through `fem_kernel.py`.

The trainer's LE branch evaluates the energy on the **small strain**
`sym(∇u)`, matching the FEM's weak form exactly, so an LE comparison is a true
code-to-code check rather than a constitutive mismatch. (NH is unaffected: its
energy is written directly in terms of `F`.)

### Where the units live

| quantity | mesh / GNN | FEM internal | reported |
|---|---|---|---|
| length | mm | m (`×1e-3`) | mm |
| modulus, stress | `POSTPROC_OUTPUT_UNIT` (MPa) | Pa (`× UNIT_TO_PA`) | MPa |
| displacement | mm (physical) | m | mm |
| energy `Π` | MPa·mm² (≡ mJ per mm of thickness) | — | MPa·mm² |
| force `F_x` | MPa·mm (per unit thickness) | N/m | MPa·mm |

The GNN trains on the **actual physical material**, so its displacement output is
already physical mm and its stress already in the config unit. There is no
reduced-modulus training trick and no post-hoc rescaling.

### Performance note

Training cost scales with mesh size, and the FEM reference solves in
milliseconds at this problem size. Use `--threads N` to pin both the GNN and the
FEM to the same core budget when the timing comparison in `summary.txt` matters
— without it the GNN may use every core while MUMPS uses one, and the reported
speed-up is not a like-for-like number. The summary flags a GNN-on-GPU run as
cross-device for the same reason.

---

## Changelog — `veri_1_hole_R1` -> this version

Refactored against the 3-D composite-rod pipeline (`app_petal_inc_3D_outscale`)
so the two read alike. **The geometry, loading and physics are unchanged**: same
square plate, same circular void, same roller/pin/traction BCs, same plane-stress
LE and NH formulations.

| area | change |
|---|---|
| **Module names** | `R3_pignn_composites_post_proc.py` -> `train_pignn.py`; `R3_predict_composite_with_fem_comparison.py` -> `evaluate_gnn_vs_fem.py`; `R1_fenicsx_LE_composite.py` -> `fem_reference_solver.py`; `fem_core_composite.py` -> `fem_kernel.py`; `run_composite_pipeline.py` -> `run_pipeline.py`; `pipeline_composite.py` -> `meshing_pipeline.py`; the `*_composite` mesh modules lose the suffix. `run_composite_pipeline` -> `run_meshing_pipeline`. |
| **Loss conditioning** | The reduced-modulus LE trick (`POSTPROC_SHARED_SCALE`) is gone, and with it a whole class of unit-conversion bugs. `OUTPUT_DISPLACEMENT_SCALE` replaces it: the network trains on the ACTUAL 210 GPa steel and its output is multiplied by a fixed characteristic displacement instead. Displacements are physical mm everywhere, with no post-hoc rescaling in the trainer or the comparison. |
| **Learning rate** | `1e-4` -> `1e-3`, matched to the now-`O(1)` scaled output. The old value was tuned against the raw-magnitude output. |
| **GNN layer** | The message is now built from `[x_i, x_j − x_i, edge_attr]` with `edge_attr = [dx, dy, |d|]` in normalised coordinates, instead of `[x_i, x_j]`. A layer can represent a finite-difference stencil rather than mean-smoothing its neighbourhood — which matters most at the hole rim. **Old checkpoints do not load.** |
| **Best-model save** | Training tracks the lowest-`Π` weights over the whole trajectory and writes `model_best.pt`; Stage 3 prefers it over the last epoch. The minimum principle makes argmin `Π` the principled selection rule; the final epoch is an arbitrary sample from Adam's oscillation. |
| **Checkpoint metadata** | Now carries the architecture (`input_dim`, `hidden_dim`, `num_layers`), `output_displacement_scale`, the material and load, and a `selection` tag. `make_model` rebuilds from these rather than from live `Config` defaults. |
| **Timing** | `training_time.json` separates optimisation compute from checkpoint/plot I/O. The comparison times the GNN forward pass and the FEM solve, reports both with their device, and computes the speed-up — flagging a GNN-on-GPU run as cross-device. `--threads N` pins both to the same core budget. |
| **Health check** | The training loop prints `2·E_int/W_ext` (-> 1.0 at the LE minimum) every `PRINT_EVERY` epochs, and `summary.txt` reports it for both the GNN and FEM fields. |
| **Headline output** | `Kt = max σ_xx / T` is computed and reported throughout (training loop, field-plot titles, training-history panel against the Kirsch line, both FEM entry points, `summary.txt`), as the dimensionless geometry-driven check the twist angle plays in the 3-D pipeline. |
| **External work** | Edge-length midpoint rule -> the exact consistent nodal force vector `f_i = L/6·(2t_i + t_j)`, assembled once and reused. For the uniform traction here the two are algebraically identical, so results are unchanged; the general form stays exact if the traction is ever made position-dependent. The assembled resultant is printed as a startup check. |
| **`LE` strain measure** | The trainer's LE branch now evaluates the energy on the small strain `sym(∇u)` instead of the Green–Lagrange strain, matching the FEM's weak form exactly. Previously the GNN minimised the St. Venant–Kirchhoff energy while the reference stayed small-strain — a systematic bias, though negligible at this load (they differ by `O(strain²)` ≈ 1e-11). The LE option is now a genuine code-to-code check. NH is unaffected: its energy is written directly in terms of `F`. |
| **Mesh I/O** | The DOLFIN-XML round-trip is gone; `CompositeMeshProcessor` takes the `(nodes, elements)` arrays directly and `composite_mesh.vtu` is the single mesh file the FEM reads. `mesh_summary.json` now carries `plate_size`, the domain bounds, the loaded-edge length and the BC provenance. |
| **Metrics** | Added the hole-rim near-field L2 band (both solvers restricted to elements touching the rim shell), a `uy`-only L2 row, and the applied-force / `J`-range rows. The `summary.txt` layout, section rules and `aggregate_summary.txt` now match the 3-D pipeline. |
| **BC sourcing** | The roller and loaded node sets are read from the `node_features` flags the mesh processor set, not re-derived from a coordinate query, so the trainer and the mesh stage cannot disagree on a tolerance. The pin is selected as the top-most node OF the roller set, guaranteeing membership. |
| **VTU metadata** | Every VTU writer declares its active point vector and scalar (`<PointData Vectors=... Scalars=...>`), so ParaView's Warp By Vector works in one click. |
| **NH load stepping** | Fixed `linspace` increments -> adaptive: a failed Newton step rolls back to the last converged state and halves the increment, growing back after recovery. |
| **`fem_reference_solver.py`** | Gains a CLI (`--mesh-dir`, `--output-dir`, `--material-model`, `--traction`) instead of module-level constants, reads the plate geometry from `mesh_summary.json`, and reports `Kt`, the assembled resultant force and solve timings. Its `E_MATRIX` was `70e9 Pa` (aluminium) while the trainer used 210 GPa steel; it is now `210e9 Pa`, so the standalone solve reproduces the SAME problem the pipeline trains on. |
| **Console style** | ASCII throughout (`[ok]`, `->`, `mu`/`lambda`/`sigma`), matching the 3-D pipeline. |

---