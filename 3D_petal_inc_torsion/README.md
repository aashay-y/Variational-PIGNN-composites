# 3-D Composite Rod in Torsion — PI-GNN + FEM

A physics-informed Graph Neural Network that predicts the displacement and
stress field of a **3-D composite rod** (matrix + prismatic petal inclusion)
under an applied **torque**, by minimising the total potential energy, validated
against a FEniCSx FEM reference solving the identical problem.

The rod is **clamped at one end and twisted at the other by a prescribed
torque**. The material is a compressible **Neo-Hookean hyperelastic** solid, and
the torque is calibrated so the rod reaches a rim shear strain of ~0.21, 
several times beyond the linear-elastic range.

The correspondence between the paper and the code is:

| paper | code |
|---|---|
| Total potential energy `Π(u) = E_int − W_ext` (training objective) | `CompositeEnergyCalculator.compute_total_potential_energy` |
| Neo-Hookean energy density `W` | `CompositeEnergyCalculator.compute_strain_energy_density_NH` |
| Message-passing operator | `GNNLayer` in `train_pignn.py` |
| Hard-imposed clamped BC | `BoundaryConditions.apply` |
| Consistent nodal load `f_i = A/12·(2t_i + t_j + t_k)` | `MeshGeometry.compute_nodal_load` |
| Torque-to-traction inversion | `MeshGeometry.traction_for_torque` |
| FEM reference weak form | `solve_composite` in `fem_kernel.py` |

## The problem

| Geometry | circular rod, radius `R = 1.0 mm`, length `H = 2.0 mm` (`H/2R = 1.0`) |
| Inclusion | petal `r(θ) = 0.5 + 0.2·sin(5θ)` mm, tip radius 0.7 mm = 0.7·R, concentric, straight through the full depth |
| `z = 0` | clamped, all three components zero |
| `z = H` | **applied torque `M_z = 0.25 µN·m`** (= 0.25 kPa·mm³ = 2.5×10⁻⁷ N·m) |
| lateral | free (traction-free cylindrical surface) |
| Material | compressible Neo-Hookean, finite strain (`LE` available as a reference option) |
| Discretisation | linear tetrahedra, **fully 3-D** — no plane-stress / plane-strain assumption |

**Everything is three-dimensional.** Three displacement components per node, six
independent stress components per element. `σ_zz`, `σ_xz`, `σ_yz` are genuine
unknowns, not quantities recovered after the fact from a 2-D solve.

### The load is a torque; the twist is an *output*

**You specify the torque** (`--torque`, `Config.APPLIED_TORQUE`).

```
t(X) = (τ/R) · ( −(y − c_y),  (x − c_x),  0 )
```

which is tangential, grows linearly with radius, and vanishes on the axis (so it
needs no special case at `r = 0`, unlike a normalised `ê_θ` field). Three
properties matter:

* Its **resultant force is exactly zero** by symmetry, so it is a **pure
  couple**, with moment `M_z = (τ/R)·∫r²dA` about the axis. The run prints the
  assembled resultant force (`~1e-16`) as a startup check.
* **τ is derived, not specified.** It is solved for so the torque *assembled over
  the discretised loaded face* equals your requested `M_z` **exactly** (verified
  to 0.0e+00 relative error). Scaling off the analytic `τ·πR³/2` instead would
  leave the realised torque ~0.65 % low on this mesh's polygonal rim — and that
  deficit shrinks with refinement, so two meshes would be loaded with two
  different torques. A load parameter must not do that.
* It is **conservative**: the traction vector is fixed in the reference frame,
  so the external work `W_ext = ∫t·u dS` is a genuine potential. That is what
  makes `Π = E_int − W_ext` a real energy the GNN can minimise, and it preserves
  the `2·E_int/W_ext → 1` health check from the 2-D pipeline.

The consequence is that **the twist angle Φ is a result, not a boundary
condition.** Both solvers report it.

**Torque units.** `APPLIED_TORQUE` is in `POSTPROC_OUTPUT_UNIT · mm³`. With the
default `kPa` that is `kPa·mm³`, which is *exactly* one micro-newton-metre:

```
1 kPa·mm³ = 10³ Pa × 10⁻⁹ m³ = 10⁻⁶ N·m = 1 µN·m
```

so the default `0.25` reads directly as **0.25 µN·m**. (`fem_reference_solver.py`
takes `--torque` in SI N·m instead, since it has no GNN config to inherit units
from.)

## Running

### Reproducing the reported results

The shipped artefacts were produced by the command below. Its meshing
parameters are exactly those recorded in
`mesh_rod_petal_fine/mesh_summary.json → mesh_generation_parameters`, and its
epoch count matches `rod_fine_checkpoints/training_time.json`.

```bash
conda activate femml

python run_pipeline.py \
    --config petal_config.json \
    --plate-radius 1.0 --height 2.0 --n-layers 20 \
    --min-density 0.04 --max-density 0.04 \
    --matrix-near-field 2.0 --inclusion-refinement 1.5 \
    --mesh-dir mesh_rod_petal_fine \
    --checkpoint-dir rod_fine_checkpoints \
    --results-dir rod_fine_results \
    --comparison-dir rod_fine_comparison \
    --material-model NH --torque 0.25 \
    --epochs 30000 --seed 0
```

This writes the mesh, trains for 30,000 epochs (~10.4 h on CPU for this mesh;
see *Performance note*) and runs the FEM comparison. To re-run only the
evaluation against the shipped checkpoint, without retraining:

```bash
python run_pipeline.py --skip-training \
    --mesh-dir mesh_rod_petal_fine \
    --checkpoint rod_fine_checkpoints/model_best.pt \
    --comparison-dir rod_fine_comparison \
    --material-model NH
```

Expected aggregate errors are in `rod_fine_comparison/aggregate_summary.txt`
(relative L2: displacement 0.83 %, `σ_θz` 5.33 %, von Mises 5.00 %).

### Option A — orchestrator (recommended)

`run_pipeline.py` runs **Stage 1 mesh → Stage 2 train → Stage 3
FEM-compare**.

```bash
# Full run from the petal config
python run_pipeline.py \
    --config petal_config.json \
    --plate-radius 1.0 --height 2.0 --n-layers 10 \
    --mesh-dir mesh_rod_petal --checkpoint-dir rod_checkpoints \
    --results-dir rod_results --comparison-dir rod_comparison \
    --epochs 30000 --seed 0

# Reuse an existing mesh
python run_pipeline.py --skip-meshing --mesh-dir mesh_rod_petal \
    --epochs 30000

# Compare only, from an existing checkpoint
python run_pipeline.py --skip-training --mesh-dir mesh_rod_petal \
    --checkpoint rod_checkpoints/model_best.pt

# Mesh only — inspect the mesh before committing the epoch budget
python run_pipeline.py --config petal_config.json \
    --mesh-dir mesh_rod_petal --skip-training --skip-comparison
```

`--skip-training` implies `--skip-meshing`. The directory defaults are
`mesh_rod_petal`, `rod_checkpoints`, `rod_results` and `rod_comparison`; the
shipped artefacts use the `*_fine` names above because they were produced with
explicit overrides. Run `python run_pipeline.py --help` for all flags.

### Option B — a single stage directly

```bash
# Stage 1 — meshing only
python meshing_pipeline.py petal_config.json \
    --plate-radius 1.0 --height 2.0 --n-layers 10 --output mesh_rod_petal

# Stage 2 — train (knobs read from train_pignn.py → class Config)
python train_pignn.py

# Stage 3 — GNN vs FEM comparison for a mesh dir
python evaluate_gnn_vs_fem.py \
    --input-dir mesh_rod_petal --material-model NH \
    --output-dir rod_comparison

# Standalone FEniCSx FEM only (no GNN) — writes CSV/VTU/summary
python fem_reference_solver.py --mesh-dir mesh_rod_petal \
    --output-dir rod_fem_reference --torque 2.5e-7      # N*m
```

---

## Configuring a case

### Material and loading

Set in `train_pignn.py → class Config`, or overridden from the
orchestrator CLI:

| flag | `Config` field | default |
|---|---|---|
| `--output-unit Pa\|kPa\|MPa\|GPa` | `POSTPROC_OUTPUT_UNIT` — the unit `--youngs-matrix` and `--torque` are given in | `kPa` |
| `--youngs-matrix` | `YOUNGS_MODULUS_MATRIX` | `1.5` |
| `--nu-matrix` / `--nu-inclusion` | `POISSONS_RATIO_MATRIX` / `_INCLUSION` | `0.40` / `0.35` |
| `--inclusion-ratio` | `INCLUSION_RATIO` (`E_inc/E_mat`) | `5.0/1.5 = 3.3333` |
| `--torque` | `APPLIED_TORQUE` — torque about the rod axis, in `<unit>·mm³` (= µN·m for kPa). The rim traction `τ` is **derived** from it. | `0.25` |
| `--material-model NH\|LE` | `MATERIAL_MODEL` | `NH` |

Mesh coordinates are in **mm**; the FEM converts to SI internally
(`coordinates × 1e-3`, `stress × UNIT_TO_PA`).

> **Ordering constraint.** `evaluate_gnn_vs_fem` snapshots
> `UNIT_TO_PA`, `FEM_E_MATRIX`, `FEM_E_INCLUSION`, `APPLIED_TORQUE` and
> `FIXED_INCLUSION_RATIO` into module-level constants **at import time**. (The
> rim traction `τ` is deliberately *not* among them: it depends on the mesh, so
> it is derived per case.) All
> overrides are therefore applied by `apply_config_overrides()` called from
> `main()` *before* any stage imports that module. Patching afterwards would
> leave the FEM reference on the old properties while the GNN trained on the new
> ones — a silent, total mismatch. Do not move that call.

### Geometry

The inclusion cross-section comes from `petal_config.json`:

```json
{ "type": "petal", "center_x": 1.0447213595, "center_y": 1.0447213595,
  "base_radius": 0.5, "amplitude": 0.2, "num_petals": 5, "phase": 0.0 }
```

The rod's **torsion axis defaults to the inclusion's own centre**, so a centred
inclusion gives a concentric rod without editing the JSON. That centre is
`shape.center()`, *not* the bounding-box centre: a petal's bbox is not centred on
the petal, because `r(θ) = r₀ + a·sin(kθ)` is asymmetric in `θ`, and using the
bbox would put the axis off by ~`a/2`.

`--plate-radius` sets `R`; pass a negative value to infer it (1.45 × the
inclusion's maximum reach from the axis, measured on the true boundary curve —
≈ 1.01 for the shipped petal). `--height` sets `H`, `--n-layers` the number of
element layers through the depth.

New inclusion shapes plug in via one `signed_distance()` primitive in
`inclusion_shapes.py`; circle and petal ship.

### `OUTPUT_DISPLACEMENT_SCALE` — loss conditioning

**Why it exists.** The GNN's raw output at initialisation is `O(1e-2)`, set by
weight init and *independent of the physics*. If the true displacement is orders
of magnitude away, the two terms of `Π = E_int − W_ext` are wildly unbalanced at
init. When `E_int` dominates, the only meaningful gradient is *"shrink u"*, while
`W_ext` — the term that creates the deformation **shape** — is numerically
negligible, and training collapses to a near-trivial field **no matter how many
epochs you run**.

**What it does.** Multiplies the network output by a fixed (non-trainable)
characteristic displacement, so the network learns `O(1)` values while the
**physical strain is unchanged**.

| value | meaning |
|-------|---------|
| `None` *(default)* | auto = `(τ / E) × H`, a nominal strain times the rod length, with `τ` the torque-derived rim traction. In torsion the rim displacement is ~`γ·H`, so this lands within a factor of a few of the truth (0.214 mm auto vs 0.437 mm actual here). |
| `1.0` | disabled |
| float | explicit |

Results are insensitive to the exact value; only the **order of magnitude**
matters. The scale is *not* a parameter, so it is absent from `model_state_dict`
— it is recorded in the checkpoint (`output_displacement_scale`) and restored by
`make_model` at inference. Rebuilding a model without it would silently rescale
the entire field.

**Diagnosing a bad run.** For linear elasticity, at the exact minimum of `Π`:

```
2 · E_int / W_ext  →  1.0
```

This is dimensionless and config-independent, so it is the single best health
check. For Neo-Hookean it is not identically 1 but stays `O(1)` (≈ 0.76 at the
converged solution here). Values far from `O(1)`, a *positive* `Π`, or a
*negative* `W_ext` mean the optimisation has not converged — inspect
conditioning before touching the architecture. The training loop prints it every
`PRINT_EVERY` epochs alongside the current twist angle.

---

## The mesh — extruded cross-section

The 3-D mesh is built by **extruding a 2-D cross-section**, not by meshing a
solid directly. The rod is a straight prismatic body, so this is exact, and it
reuses the whole existing geometry stack:

```
petal_config.json
   ↓  adaptive Poisson-disk collocation on the DISK cross-section
   ↓  Delaunay triangulation  (the disk is convex → the hull IS the domain)
   ↓  classify triangles matrix / inclusion / interface
   ↓  extrude N layers in z  →  triangular prisms
   ↓  split each prism into 3 tetrahedra
tetrahedral rod mesh
```

Node ids are **layer-major** (`node = layer × N_section + node_2d`), so `z = 0`
and `z = H` are exact node layers — the clamped and loaded sets are node sets,
not geometric queries — and any field can be sliced at a given height.

### Prism → tetrahedron conformity

Three tets per prism is the minimum, but a naive split leaves the quadrilateral
side faces with **mismatched diagonals** between neighbouring prisms — a
non-conforming mesh that FEniCSx rejects and whose energy integral is wrong. The
fix is to make the diagonal a function of the **global vertex indices only**, so
two prisms sharing a face independently choose the same one. Sorting each
triangle's node ids ascending as `v0 < v1 < v2` and emitting

```
T1 = (b0, b1, b2, t0)      diagonal on face (b_i,b_j), i<j  is always  b_j — t_i
T2 = (b1, b2, t0, t1)      → depends only on the ordering of the two shared
T3 = (b2, t0, t1, t2)        global ids, so neighbours always agree
```

does exactly that (the standard Dompierre et al. subdivision). The generator
verifies positive volume on every tet and flips any that come out negative.

Verified on the shipped mesh: every interior face is shared by exactly 2 tets,
every boundary face by 1, the two end caps carry exactly `N_section_triangles`
faces each, and the total volume matches `πR²H` to 0.3 % (the deficit is the
polygonal rim, not an error).

### Sizing the mesh

`--n-layers` should give a layer thickness `dz = H/n_layers` comparable to the
in-plane element size; the generator prints the ratio and warns about slivers.

| `--n-layers` | `dz / h_xy` | nodes | tets | min tet quality |
|---|---|---|---|---|
| 8 | 1.93 | 2,331 | 11,304 | 0.15 |
| 10 *(default)* | 1.57 | 2,838 | 14,070 | 0.21 |
| 16 | ~0.98 | ~4,400 | ~22,500 | — |

(Node counts vary by a few percent between runs — the Poisson-disk sampler is
unseeded, so each mesh is a fresh draw.)

Cross-section density is set by `--min-density` (spacing at the interface, ×
inclusion `char_length`) and `--max-density` (far-field spacing, × `2R`).

> **Sampler ceiling.** `poisson_disk_sampling` stops after a fixed
> `max_attempts`, and filling a region with `N` points costs ~`2N` of them — so
> past a certain density it **returns a partly-filled region silently**. Such a
> mesh trains and compares perfectly happily and reports a meaningless error.
> The generator therefore checks the realised count against the `~0.8·A/h²` a
> uniform fill would need and prints a loud warning if it falls below half that.
> Heed it: raise `max_attempts` or back off to a coarser spacing.

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
* `displacement_face_epoch_*.png` — `u_x`, `u_y`, **`u_z` (warping)** and the
  twist angle on the loaded face
* `stress_midheight_epoch_*.png` — `σ_θz`, von Mises, `σ_zz`, `σ_rz` on the
  mid-height slice
* `deformed_rod.png`, `training_history.png` (loss, `E_int`, `W_ext`, twist)

### Comparison → `<comparison-dir>/<case>/`

* `combined_gnn_fem.vtu` — the **full 3-D tetrahedral mesh** with every
  GNN/FEM/error field: 35 point arrays and 36 cell arrays. The matrix/inclusion
  interface is **split** (duplicated nodes) so the nodal stress fields carry a
  true discontinuity there instead of ramping across straddling cells.
  Both a *smoothed nodal* copy (for display) and the *raw element* value (what
  the metrics are computed from) are stored for every stress component.
* `error_data.csv` — per node: coordinates, `r`, all three displacement
  components, twist, and every stress field for GNN and FEM with abs/rel errors
* `summary.txt` — material/geometry/BCs, **torsion response** (applied torque,
  twist, warping, `J` range), ML params, timings, L2 errors (global + interface
  band), R² (global / near-field / far-field), energy comparison, signed errors
* `aggregate_summary.txt` — median & mean L2 across the processed cases

### Viewing it in ParaView

Open `combined_gnn_fem.vtu` and apply **Warp By Vector** — it works with no
setup, because the file declares `GNN_displacement_mm` as the *active* point
vector (`<PointData Vectors="...">`). Without that attribute VTK registers the
displacement arrays but leaves `active_vectors = None`, and the Warp By Vector
filter has nothing to default to; that is a real trap, since the array is
plainly there in the array list and the failure looks like a data problem rather
than a metadata one.

* **Warp By Vector** → deformed rod. Set *Scale Factor* to 1 for the true
  deformed shape (the twist is ~24°, so little or no exaggeration is needed).
* To warp by the reference solution instead, pick `FEM_displacement_mm` in the
  filter's *Vectors* dropdown.
* `GNN_von_mises_kPa` is the active scalar, so the deformed rod is coloured
  sensibly straight away.
* The rod comes out slightly **taller** when warped (z: 0 → 2.018 mm). That is
  the Poynting effect — a twisted hyperelastic rod elongates — not a bug.
* Use **Clip** or **Slice** through the axis to see the interior fields; the
  file is a genuine 3-D volume mesh (14,070 tetrahedra), not a surface.

`fem_reference_solver.py`'s `simulation_torque_*.vtu` and the meshing stage's
`composite_mesh.vtu` declare their active arrays the same way.

### Fields the 3-D problem adds

| field | why it matters |
|---|---|
| `u_z` (warping) | out-of-plane displacement of the cross-section. Zero for a homogeneous circular shaft; here the 5-fold petal drives a 5-fold warping pattern. This is the hardest component to get right and gets its own L2 row. |
| `σ_θz` | **the** torsion stress. For a homogeneous shaft it is the only non-zero component and grows linearly with `r`; here it also picks up the petal's 5-fold signature. |
| `σ_rz`, `σ_rr`, `σ_θθ` | remaining cylindrical components, referred to the recorded torsion axis |
| `σ_zz`, `σ_xz`, `σ_yz` | genuine unknowns now — in torsion the out-of-plane shears carry most of the load |
| `J = det F` | volume ratio; `1.0` would be incompressible. Shows how far the compressible Neo-Hookean model is being pushed. |
| twist angle | per node, and the loaded-face mean/max — the headline output |
| applied torque `M_z` | integrated over the discretised face by both solvers |

**von Mises** is formed from the full 3-D deviator on both sides:

```
σ_vm = √( ½[(σxx−σyy)² + (σyy−σzz)² + (σzz−σxx)²] + 3(σxy² + σyz² + σxz²) )
```

All three shear terms are required. The 2-D form that kept only `σxy` would
discard essentially the whole torsional contribution.

---

## Verification

Five checks, each of which fails loudly if the physics is wired up wrong.

**1. The two energy functionals are identical.** Take the FEM's converged
displacement field and evaluate the *GNN's* energy functional on it, then
compare against FEniCSx's own assembled `∫ψ dx`. This tests the strain-energy
density, the Lamé parameters, the per-element material assignment and the volume
integration across two completely separate implementations (PyTorch vs UFL):

| model | FEniCSx assembled | GNN functional | rel. diff |
|---|---|---|---|
| NH | 4.5272038288e-07 J | 4.5272038288e-07 J | `2.8e-15` |
| LE | 1.1160610498e-06 J | 1.1160610498e-06 J | `7.6e-16` |

Machine precision. So when `summary.txt` compares "GNN Π" against "FEM Π", it is
genuinely the same functional evaluated on two different fields — which is what
makes the minimum principle (`Π_GNN ≥ Π_FEM`) a meaningful check rather than a
comparison of two different quantities.

**2. The external work matches too** — `W_ext` from the consistent nodal load
vector agrees with the FEM's surface integral `∫t·u dS` to `8.7e-16` (NH) and
`1.5e-15` (LE), confirming that `f_i = A/12·(2t_i + t_j + t_k)` really is exact
for a linearly-varying traction rather than a quadrature approximation.

**3. The minimum-principle identity holds exactly.** On the FEM's LE solution,
`2·E_int/W_ext = 2 × 1.1160610498 / 2.2321220996 = 1.0000000`, which is the
identity the training-loop health check watches for. A converged LE *training*
run reproduces it too (measured: 0.9964, 1.0071, 0.9987 over successive
reports — Adam oscillating about the exact minimum).

**4. The requested torque is the delivered torque.** τ is solved against the
assembled torque of the discretised face, so the realised `M_z` equals the
request to `0.0e+00` relative error at every magnitude tested (0.55, 1.1,
2.2 µN·m), while the resultant force stays at `~1e-16` — a pure couple.

**5. The load agrees with independent analysis.** The dead-load torsion model
(`(GJ/H)Φ = M_z cos Φ`, table above) predicts 60.3° where the FEM gives 62.7°.
Geometry, material assignment, traction field and solver all have to be right
for those to agree at all.

Checks (1)–(3) are what `compute_potential_energies` reports in every Stage-3
run: it evaluates `CompositeEnergyCalculator.compute_total_potential_energy` on
both the GNN and the FEM displacement fields, on the same mesh, and writes
`fem_tpe`, `gnn_tpe` and their split into `E_int` / `W_ext` to `summary.txt`.
Check (4) is printed by `MeshGeometry.compute_nodal_load` at the start of every
training run (`Resultant force` and `Applied torque Mz`). Check (5) is the
`Mean twist of loaded face` reported against the table in
*How far a dead traction may be pushed*.

---

## Code structure

```
.
├── run_pipeline.py              Orchestrator: mesh → train → compare (CLI-driven)
│
├── Stage 1 — Meshing
│   ├── meshing_pipeline.py      drives meshing
│   ├── collocation_generator.py adaptive points on the DISK cross-section
│   ├── mesh_generator.py        Delaunay + classification + EXTRUSION to tets
│   ├── mesh_processor.py        3-D graph, facets, features, VTU, summary
│   └── inclusion_shapes.py      inclusion cross-section geometry (circle, petal)
│
├── Stage 2 — Training
│   └── train_pignn.py           PI-GNN (energy minimisation); class Config
│
├── Stage 3 — Inference + FEM comparison
│   ├── evaluate_gnn_vs_fem.py   GNN vs FEM → VTU/CSV/summary
│   └── fem_reference_solver.py  standalone FEniCSx FEM → CSV/VTU
│
├── fem_kernel.py                Shared 3-D FEM kernel (LE + NH, torsion BCs,
│                                adaptive load stepping). Single source of truth —
│                                imported by both FEM entry points, so the
│                                standalone solver and the comparison cannot
│                                diverge.
│
├── petal_config.json            Inclusion geometry for the shipped case
├── environment.yml              Conda environment (includes FEniCSx)
├── requirements.txt             Pip dependencies (FEniCSx installed separately)
│
└── Shipped artefacts of the reported run
    ├── mesh_rod_petal_fine/     Mesh case: 38,430 nodes / 214,260 tets, 20 layers
    ├── rod_fine_checkpoints/    Checkpoints (incl. model_best.pt) + TensorBoard runs/
    ├── rod_fine_results/        Training-history and deformed-rod figures
    └── rod_fine_comparison/     Stage-3 GNN vs FEM outputs + aggregate_summary.txt
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

The GNN is `encoder → L × GNNLayer → decoder`, predicting 3 components per node.
Each layer's message is built from the receiver state `x_i`, the **difference**
`x_j − x_i`, and the geometric edge attribute `[dx, dy, dz, |d|]` — a directional
derivative needs both the state difference and the physical separation it is
taken over, which is what lets a layer represent a finite-difference stencil
rather than just mean-smoothing its neighbourhood.

Boundary conditions are imposed **hard**, by masking the network output to zero
on the clamped face, so they hold exactly at every epoch and no penalty term
competes with the energy.

### Where the units live

| quantity | mesh / GNN | FEM internal | reported |
|---|---|---|---|
| length | mm | m (`×1e-3`) | mm |
| modulus, stress | `POSTPROC_OUTPUT_UNIT` (kPa) | Pa (`× UNIT_TO_PA`) | kPa |
| displacement | mm (physical) | m | mm |
| energy `Π` | kPa·mm³ | — | kPa·mm³ |
| torque `M_z` | kPa·mm³ | N·m | kPa·mm³ |

The GNN trains on the **actual physical material**, so its displacement output is
already physical mm and its stress already in the config unit. There is no
reduced-modulus training trick and no post-hoc rescaling.

### Performance note

Training cost scales with mesh size. The shipped run
(`mesh_rod_petal_fine`, 38,430 nodes / 214,260 tets) took **37,274 s of
optimisation compute** for 30,000 epochs on CPU — about 1.24 s/epoch, or ~10.4
hours — as recorded in `rod_fine_checkpoints/training_time.json`; a GPU is
substantially faster. The reported time excludes checkpoint and plot I/O, which
the trainer measures separately. The FEM reference solves in seconds
(Neo-Hookean, adaptive load stepping, MUMPS direct LU). Use
`--threads N` to pin both the GNN and the FEM to the same core budget when the
timing comparison in `summary.txt` matters.

> **Transient start-up crash.** A `SIGSEGV` inside `libucs` (UCX) during
> interpreter start-up is occasionally observed. It is unrelated to this code and
> clears on retry.

---

## Changelog — 2-D plate → 3-D rod

| area | change |
|---|---|
| **Problem** | 2-D plate under right-edge traction (plane strain) → 3-D rod clamped at `z=0` and twisted at `z=H` by a tangential dead traction. Twist is an output of the applied torque. |
| **Kinematics** | All plane-stress / plane-strain assumptions removed. `F`, strain and Cauchy stress are full 3×3; `σ_zz`, `σ_xz`, `σ_yz` are unknowns rather than recovered quantities. |
| **Elements** | Triangles → tetrahedra, via extrusion of the classified cross-section with a globally-consistent prism split. |
| **Domain** | Square plate → circular cross-section. `plate_size` → `plate_radius` + `plate_center` (the torsion axis, recorded in `mesh_summary.json`). |
| **Material** | Default is now compressible Neo-Hookean at `E_mat = 1.5 kPa (ν 0.40)` / `E_inc = 5.0 kPa (ν 0.35)`. `LE` retained as an option but flagged as invalid at this load. |
| **Load input** | The user specifies a **torque** (`--torque`), not a traction. The rim traction that delivers it is solved for against the *assembled* torque of the discretised face, so the requested torque is delivered exactly (0.0e+00 rel. error) and is mesh-independent. |
| **External work** | Edge-length midpoint rule → the **exact consistent nodal force vector** of a linearly-varying traction over triangular facets, `f_i = A/12·(2t_i + t_j + t_k)`. A one-point rule would lose the `r²` weighting that carries the torque. |
| **BCs** | Two rollers (`ux=0`, `uy=0`) → one fully clamped end face, which removes all six rigid-body modes. |
| **Node features** | `(N,6) [Dirichlet, Neumann, Material_ID, Interface, X, Y]` → `(N,7)` with `Z`. GNN input dim 4 → 5, edge attributes `[dx,dy,|d|]` → `[dx,dy,dz,|d|]`, output 2 → 3. |
| **Metrics** | Area-weighted → **volume-weighted** relative L2. Added `u_z`, `σ_zz`, `σ_yz`, `σ_xz`, `σ_θz`, `J`, twist and torque. |
| **`LE` strain measure** | The 2-D trainer evaluated the LE energy on the Green–Lagrange strain (St. Venant–Kirchhoff) while the FEM used small strain — a documented systematic bias. The trainer's LE branch now uses small strain `sym(∇u)`, matching the FEM exactly, so the LE option is a genuine code-to-code check. (NH is unaffected: its energy is written directly in terms of `F`.) |
| **Removed** | `POSTPROC_SHARED_SCALE` and the reduced-modulus LE training trick (the old README already recommended against it — `OUTPUT_DISPLACEMENT_SCALE` supersedes it, and removing it deleted a whole class of unit-conversion bugs). Also removed: the element-size sweep (`sweep_element_size.py`, `plot_element_sweep.py`), the edge-ablation control (`ABLATE_EDGES`), and the helper scripts `regenerate_meshes_rollerBC.py` / `build_pignn_mesh_from_cpinn.py`. |
| **Load magnitude** | The torque default was `1.1 µN·m` (~63° twist). At that twist the dead traction's `sin Φ` component — 89 % of it — pulls the free end radially outward, ballooning the rim by +21 %, and only 46 % of the moment still drives the twist. Reduced to **`0.25 µN·m`** (~24°, `γ = 0.21`), where the bulge is 2 % and 91 % of the moment is useful, while staying several times past the linear-elastic range. See *How far a dead traction may be pushed*. |
| **Mesh I/O** | The DOLFIN-XML round-trip is gone; the processor takes arrays directly and writes `composite_mesh.vtu`, which is also what the FEM reads. `mesh_summary.json` now carries the rod axis/radius/height that the traction field is defined from. |

---

## Citation

If you use this code or build on the method, please cite the accompanying
paper. Replace the placeholder fields below with the final publication details.

```bibtex
@article{PLACEHOLDER_CITATION_KEY,
  title   = {{TODO: paper title}},
  author  = {TODO: Author, First and Author, Second},
  journal = {TODO: journal or conference},
  year    = {TODO: year},
  volume  = {TODO},
  number  = {TODO},
  pages   = {TODO},
  doi     = {TODO: 10.xxxx/xxxxx},
  url     = {TODO}
}
```
---