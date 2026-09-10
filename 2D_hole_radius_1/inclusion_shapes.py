"""
Hole / Void Shape Abstraction
=============================
Geometry layer for the holes (voids) cut out of the plate. The whole
mesh/graph pipeline (collocation -> Delaunay -> void removal -> matrix graph)
is derived from one primitive: "is this point inside the hole?". This module
provides that primitive behind a common interface, dispatched on the JSON
``type`` field.

This voids-only pipeline supports **circular holes only** (irregular voids are
out of scope); the abstraction is kept so the same machinery could host more
shapes later, but the registry intentionally exposes ``circle`` alone.

The unifying primitive is ``signed_distance(x, y)``:
    < 0  inside the hole
    > 0  outside (in the matrix)
    = 0  on the hole boundary
with magnitude approximately the distance to the boundary. Everything else
(point-in tests, boundary sampling, density grading, bbox) is derived from it.

Supported types:
    - "circle" (default): center_x, center_y, radius

Backward compatibility: a config dict with no "type" key and the classic
{center_x, center_y, radius} fields maps to ``Circle``, whose signed distance is
exactly ``hypot(dx, dy) - radius``.

Author: Claude
Date: 2026-06-11
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend: figures are saved, never displayed.
# Avoids pulling in the Qt/PySide6 backend, whose shiboken machinery injects a
# plain ``typing.Self`` that later breaks importing torch._dynamo / torch_geometric.
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class InclusionShape:
    """Common interface for a hole. Subclasses implement signed_distance,
    boundary_points, sample_interior_seed, bbox, char_length and draw."""

    # --- core primitive (subclass must implement) ---------------------------
    def signed_distance(self, x, y):
        """Signed distance: <0 inside, >0 outside, 0 on boundary. Accepts
        scalars or numpy arrays."""
        raise NotImplementedError

    # --- point classification (derived from signed_distance) ----------------
    def contains(self, x, y, tolerance: float = 1e-6) -> bool:
        """Inclusive inside test (boundary counts as inside).
        Matches the legacy node-classification test ``dist <= r + tol``."""
        return bool(self.signed_distance(x, y) <= tolerance)

    def is_strictly_inside(self, x, y, tolerance: float = 1e-6) -> bool:
        """Strict inside test. Matches legacy ``dist < r - tol``."""
        return bool(self.signed_distance(x, y) < -tolerance)

    def is_on_boundary(self, x, y, tolerance: float = 1e-3) -> bool:
        """Boundary test. Matches legacy ``abs(dist - r) < tol``."""
        return bool(abs(self.signed_distance(x, y)) < tolerance)

    # --- geometry helpers (subclass must implement) -------------------------
    def boundary_points(self, spacing: float) -> np.ndarray:
        """(M, 2) points spaced ~``spacing`` apart along the boundary curve."""
        raise NotImplementedError

    def sample_interior_seed(self, rng: np.random.Generator) -> Tuple[float, float]:
        """A point guaranteed to lie strictly inside (for Poisson-disk seeding)."""
        raise NotImplementedError

    def bbox(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box (xmin, xmax, ymin, ymax)."""
        raise NotImplementedError

    def char_length(self) -> float:
        """Representative radius used to scale local mesh density."""
        raise NotImplementedError

    def center(self) -> Tuple[float, float]:
        """Geometric center used for detail-view framing."""
        xmin, xmax, ymin, ymax = self.bbox()
        return (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))

    def draw(self, ax, **kwargs):
        """Draw the hole boundary on a matplotlib axis."""
        raise NotImplementedError


class Circle(InclusionShape):
    """Circular hole -- the only supported void shape."""

    def __init__(self, cfg: Dict):
        self.cx = float(cfg['center_x'])
        self.cy = float(cfg['center_y'])
        self.r = float(cfg['radius'])

    def signed_distance(self, x, y):
        # Fast scalar path (this is called in the Poisson-sampler hot loop);
        # numpy path only for array inputs.
        if np.isscalar(x) and np.isscalar(y):
            return math.hypot(x - self.cx, y - self.cy) - self.r
        return np.hypot(np.asarray(x) - self.cx, np.asarray(y) - self.cy) - self.r

    def boundary_points(self, spacing: float) -> np.ndarray:
        circumference = 2.0 * np.pi * self.r
        n_points = max(12, int(np.ceil(circumference / spacing)))
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        return np.column_stack([self.cx + self.r * np.cos(angles),
                                self.cy + self.r * np.sin(angles)])

    def sample_interior_seed(self, rng: np.random.Generator) -> Tuple[float, float]:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(0.0, self.r * 0.95)
        return (self.cx + radius * np.cos(angle), self.cy + radius * np.sin(angle))

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.r, self.cx + self.r, self.cy - self.r, self.cy + self.r)

    def char_length(self) -> float:
        return self.r

    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def draw(self, ax, **kwargs):
        kwargs.setdefault('fill', False)
        kwargs.setdefault('edgecolor', 'red')
        kwargs.setdefault('linewidth', 2)
        ax.add_patch(plt.Circle((self.cx, self.cy), self.r, **kwargs))


# ---------------------------------------------------------------------------
# Factory + config helpers
# ---------------------------------------------------------------------------

_SHAPE_REGISTRY = {
    'circle': Circle,
}


def make_inclusion(cfg: Dict) -> InclusionShape:
    """Build a hole shape from a config dict, dispatching on ``type``
    (defaults to 'circle'; circle is the only supported void type)."""
    shape_type = str(cfg.get('type', 'circle')).lower()
    if shape_type not in _SHAPE_REGISTRY:
        raise ValueError(
            f"Unknown hole type '{shape_type}'. "
            f"This voids pipeline supports circular holes only: {sorted(_SHAPE_REGISTRY)}")
    return _SHAPE_REGISTRY[shape_type](cfg)


def make_inclusions(configs: List[Dict]) -> List[InclusionShape]:
    """Vectorized factory over a list of config dicts."""
    return [make_inclusion(c) for c in configs]


def infer_plate_size(configs: List[Dict]) -> float:
    """Infer a square plate size from the union of hole bounding boxes.
    Mirrors the legacy heuristic (max positive extent x 1.1, rounded up) but is
    shape-agnostic instead of assuming center+radius."""
    max_extent = 0.0
    for cfg in configs:
        _, xmax, _, ymax = make_inclusion(cfg).bbox()
        max_extent = max(max_extent, xmax, ymax)
    return float(np.ceil(max_extent * 1.1))


def load_inclusion_config(json_path: str) -> Tuple[List[Dict], float]:
    """Load hole config (single dict or list of dicts) and infer plate size.
    Returns the raw dicts (the serializable source of truth) plus plate_size."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        inclusions = [data]
    elif isinstance(data, list):
        inclusions = data
    else:
        raise ValueError("Invalid JSON format. Expected dict or list of dicts.")

    return inclusions, infer_plate_size(inclusions)


def describe(cfg: Dict) -> str:
    """Short human-readable one-liner for a hole config."""
    shape = make_inclusion(cfg)
    cx, cy = shape.center()
    return f"circle: center=({cx:.4f}, {cy:.4f}), radius={shape.char_length():.4f}"
