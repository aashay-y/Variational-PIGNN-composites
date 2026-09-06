## Environment

Stages 1 and 2 need only the PyTorch stack; Stage 3 additionally needs
FEniCSx. Because FEniCSx is not reliably installable from PyPI, conda is the
supported path:

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