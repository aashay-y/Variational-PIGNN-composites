"""
Inclusion Shape Abstraction
===========================
Geometry layer for the inclusion CROSS-SECTION of the composite rod. The whole
mesh/graph pipeline (collocation -> Delaunay -> extrusion -> material/interface
graph) is derived from one primitive: "is this point inside the inclusion?".

The rod is a straight prismatic body, so the inclusion is this 2-D shape swept
along z: every z-layer of the 3-D mesh classifies against the SAME planar
signed-distance field, and these shapes stay purely two-dimensional.

The primitive is provided for any inclusion shape behind a common interface,
dispatched on the JSON ``type`` field.

The unifying primitive is ``signed_distance(x, y)``:
    < 0  inside the inclusion
    > 0  outside (in the matrix)
    = 0  on the interface
with magnitude approximately the distance to the boundary. Everything else
(point-in tests, boundary sampling, density grading, bbox) is derived from it.

Supported types:
    - "circle" (default): center_x, center_y, radius        -- existing behavior
    - "petal":  center_x, center_y, base_radius, amplitude,
                num_petals, phase                            -- analytic flower

Backward compatibility: a config dict with no "type" key and the classic
{center_x, center_y, radius} fields maps to ``Circle``, whose signed distance is
exactly ``hypot(dx, dy) - radius`` -- reproducing prior results bit-for-bit.
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
    """Common interface for an inclusion. Subclasses implement signed_distance,
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
        """Draw the inclusion boundary on a matplotlib axis."""
        raise NotImplementedError


class Circle(InclusionShape):
    """Circular inclusion -- the original (and default) shape."""

    def __init__(self, cfg: Dict):
        """Build a circular inclusion from its config dict.

        Args:
            cfg: Mapping with keys ``center_x``, ``center_y`` and ``radius``,
                all in the same length unit as the rest of the mesh.
        """
        self.center_x = float(cfg['center_x'])
        self.center_y = float(cfg['center_y'])
        self.radius = float(cfg['radius'])

    def signed_distance(self, x, y):
        """Exact signed distance to the circle, negative inside.

        Args:
            x: Scalar or array of x coordinates.
            y: Scalar or array of y coordinates, broadcastable against ``x``.

        Returns:
            float or ndarray: ``hypot(dx, dy) - radius``, matching the shape of
            the broadcast inputs.
        """
        # Fast scalar path (this is called in the Poisson-sampler hot loop);
        # numpy path only for array inputs.
        if np.isscalar(x) and np.isscalar(y):
            return math.hypot(x - self.center_x, y - self.center_y) - self.radius
        return np.hypot(np.asarray(x) - self.center_x,
                        np.asarray(y) - self.center_y) - self.radius

    def boundary_points(self, spacing: float) -> np.ndarray:
        """Sample the circle boundary at roughly uniform arc-length spacing.

        Args:
            spacing: Target arc length between consecutive samples.

        Returns:
            ndarray: Boundary coordinates of shape (M, 2), with at least 12
            points regardless of ``spacing``.
        """
        circumference = 2.0 * np.pi * self.radius
        n_points = max(12, int(np.ceil(circumference / spacing)))
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        return np.column_stack([self.center_x + self.radius * np.cos(angles),
                                self.center_y + self.radius * np.sin(angles)])

    def sample_interior_seed(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Draw a random point strictly inside the circle.

        Args:
            rng: NumPy generator supplying the random draws.

        Returns:
            tuple: An ``(x, y)`` point inside 95 % of the radius, so it stays
            clear of the interface.
        """
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(0.0, self.radius * 0.95)
        return (self.center_x + radius * np.cos(angle),
                self.center_y + radius * np.sin(angle))

    def bbox(self) -> Tuple[float, float, float, float]:
        """Return the axis-aligned bounding box ``(xmin, xmax, ymin, ymax)``."""
        return (self.center_x - self.radius, self.center_x + self.radius,
                self.center_y - self.radius, self.center_y + self.radius)

    def char_length(self) -> float:
        """Return the circle radius as the characteristic mesh-density length."""
        return self.radius

    def center(self) -> Tuple[float, float]:
        """Return the exact circle centre ``(x, y)``."""
        return (self.center_x, self.center_y)

    def draw(self, ax, **kwargs):
        """Draw the circle outline on a matplotlib axis.

        Args:
            ax: Target matplotlib axis.
            **kwargs: Patch styling overrides forwarded to ``plt.Circle``;
                fill, edge colour and line width default to an unfilled red
                outline.
        """
        kwargs.setdefault('fill', False)
        kwargs.setdefault('edgecolor', 'red')
        kwargs.setdefault('linewidth', 2)
        ax.add_patch(plt.Circle((self.center_x, self.center_y),
                                self.radius, **kwargs))


class Petal(InclusionShape):
    """Petal/flower inclusion: r(theta) = base_radius + amplitude*sin(num_petals*theta + phase).

    Star-convex about its center whenever base_radius > |amplitude| (the radius
    stays positive and single-valued in theta), so the radial inside/outside
    test ``r_point - r_boundary(theta)`` is EXACT for classification.
    """

    # Number of samples in the cached boundary polyline. 4000 points keeps the
    # arc-length resampling error well below the finest mesh spacing used here.
    _N_BOUNDARY_SAMPLES = 4000

    def __init__(self, cfg: Dict):
        """Build a petal inclusion from its config dict.

        Also caches a closed high-resolution polyline of the true boundary,
        which ``boundary_points`` and ``bbox`` are derived from.

        Args:
            cfg: Mapping with keys ``center_x``, ``center_y``, ``base_radius``
                and ``num_petals``; ``amplitude`` and ``phase`` are optional
                and default to 0.

        Raises:
            ValueError: If ``base_radius <= |amplitude|``, which would make the
                shape non star-convex and the radial inside test ambiguous.
        """
        self.center_x = float(cfg['center_x'])
        self.center_y = float(cfg['center_y'])
        self.base_radius = float(cfg['base_radius'])
        self.amplitude = float(cfg.get('amplitude', 0.0))
        self.num_petals = float(cfg['num_petals'])
        self.phase = float(cfg.get('phase', 0.0))

        if self.base_radius <= abs(self.amplitude):
            # Not star-convex -> radial test would be ambiguous. Guard loudly.
            raise ValueError(
                f"Petal requires base_radius > |amplitude| for a valid radial "
                f"inside test (got base_radius={self.base_radius}, "
                f"amplitude={self.amplitude}).")

        # Cache a closed high-res polyline of the true curve (first point == last).
        theta = np.linspace(-np.pi, np.pi, self._N_BOUNDARY_SAMPLES, endpoint=True)
        boundary_radius = self._radius_at(theta)
        self._boundary_x = self.center_x + boundary_radius * np.cos(theta)
        self._boundary_y = self.center_y + boundary_radius * np.sin(theta)

    def _radius_at(self, theta):
        """Evaluate the polar boundary radius r(theta) of the petal curve.

        Args:
            theta: Scalar or array of polar angles in radians, measured about
                the petal centre.

        Returns:
            float or ndarray: ``base_radius + amplitude * sin(num_petals *
            theta + phase)``, matching the shape of ``theta``.
        """
        return self.base_radius + self.amplitude * np.sin(
            self.num_petals * theta + self.phase)

    def signed_distance(self, x, y):
        """Radial signed distance to the petal boundary, negative inside.

        Exact for classification because the shape is star-convex about its
        centre: a point is inside precisely when its polar radius is smaller
        than the boundary radius at the same angle. The magnitude is a radial
        offset rather than a true Euclidean distance, which is what the density
        grading and interface tests are calibrated against.

        Args:
            x: Scalar or array of x coordinates.
            y: Scalar or array of y coordinates, broadcastable against ``x``.

        Returns:
            float or ndarray: ``r_point - r_boundary(theta)``, matching the
            shape of the broadcast inputs.
        """
        # Fast scalar path (hot loop); numpy path only for array inputs.
        if np.isscalar(x) and np.isscalar(y):
            dx = x - self.center_x
            dy = y - self.center_y
            theta = math.atan2(dy, dx)
            r_boundary = self.base_radius + self.amplitude * math.sin(
                self.num_petals * theta + self.phase)
            return math.hypot(dx, dy) - r_boundary
        dx = np.asarray(x) - self.center_x
        dy = np.asarray(y) - self.center_y
        theta = np.arctan2(dy, dx)
        r_point = np.hypot(dx, dy)
        return r_point - self._radius_at(theta)

    def boundary_points(self, spacing: float) -> np.ndarray:
        """Sample the petal boundary at uniform arc-length spacing.

        Args:
            spacing: Target arc length between consecutive samples.

        Returns:
            ndarray: Boundary coordinates of shape (M, 2), with at least 12
            points regardless of ``spacing``.
        """
        # Arc-length resampling of the cached closed curve -> even spacing along
        # the true boundary (naturally denser near the sharp petal tips).
        segment_lengths = np.hypot(np.diff(self._boundary_x),
                                   np.diff(self._boundary_y))
        arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = arc_length[-1]
        n_points = max(12, int(round(total_length / spacing)))
        targets = np.linspace(0.0, total_length, n_points, endpoint=False)
        px = np.interp(targets, arc_length, self._boundary_x)
        py = np.interp(targets, arc_length, self._boundary_y)
        return np.column_stack([px, py])

    def sample_interior_seed(self, rng: np.random.Generator) -> Tuple[float, float]:
        """Draw a random point strictly inside the petal.

        Args:
            rng: NumPy generator supplying the random draws.

        Returns:
            tuple: An ``(x, y)`` point inside 90 % of the local boundary
            radius, so it stays clear of the interface.
        """
        theta = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(0.0, 0.9) * self._radius_at(theta)
        return (self.center_x + radius * np.cos(theta),
                self.center_y + radius * np.sin(theta))

    def bbox(self) -> Tuple[float, float, float, float]:
        """Return the bounding box ``(xmin, xmax, ymin, ymax)`` of the curve.

        Computed from the cached boundary polyline, so it tracks the petal tips
        rather than an analytic over-estimate.
        """
        return (float(self._boundary_x.min()), float(self._boundary_x.max()),
                float(self._boundary_y.min()), float(self._boundary_y.max()))

    def char_length(self) -> float:
        """Return the base radius as the characteristic mesh-density length."""
        return self.base_radius

    def center(self) -> Tuple[float, float]:
        """Return the petal centre ``(x, y)``.

        Overrides the bounding-box midpoint of the base class, which is offset
        from the true centre because r(theta) is asymmetric in theta.
        """
        return (self.center_x, self.center_y)

    def draw(self, ax, **kwargs):
        """Draw the petal outline on a matplotlib axis.

        Args:
            ax: Target matplotlib axis.
            **kwargs: Patch styling overrides forwarded to ``plt.Polygon``;
                fill, edge colour and line width default to an unfilled red
                outline.
        """
        kwargs.setdefault('fill', False)
        kwargs.setdefault('edgecolor', 'red')
        kwargs.setdefault('linewidth', 2)
        ax.add_patch(plt.Polygon(
            np.column_stack([self._boundary_x, self._boundary_y]),
            closed=True, **kwargs))


# ---------------------------------------------------------------------------
# Factory + config helpers
# ---------------------------------------------------------------------------

_SHAPE_REGISTRY = {
    'circle': Circle,
    'petal': Petal,
}


def make_inclusion(cfg: Dict) -> InclusionShape:
    """Build an InclusionShape from a config dict, dispatching on ``type``
    (defaults to 'circle' for backward compatibility)."""
    shape_type = str(cfg.get('type', 'circle')).lower()
    if shape_type not in _SHAPE_REGISTRY:
        raise ValueError(
            f"Unknown inclusion type '{shape_type}'. "
            f"Supported: {sorted(_SHAPE_REGISTRY)}")
    return _SHAPE_REGISTRY[shape_type](cfg)


def make_inclusions(configs: List[Dict]) -> List[InclusionShape]:
    """Vectorized factory over a list of config dicts."""
    return [make_inclusion(c) for c in configs]


def infer_plate_center(configs: List[Dict]) -> Tuple[float, float]:
    """Centre of the rod cross-section = the torsion axis.

    Taken as the mean of the inclusions' own centres (``shape.center()``), not
    of their bounding boxes: a petal's bbox is not centred on the petal, because
    r(theta) = r0 + a*sin(k*theta) is asymmetric in theta. Using the bbox centre
    would put the torsion axis off the inclusion by ~a/2 and silently break the
    concentric geometry. Pass ``plate_center`` explicitly for an eccentric rod.
    """
    if not configs:
        return (0.0, 0.0)
    centers = [make_inclusion(cfg).center() for cfg in configs]
    return (float(np.mean([c[0] for c in centers])),
            float(np.mean([c[1] for c in centers])))


def inclusion_max_reach(configs: List[Dict],
                        center: Tuple[float, float] = None) -> float:
    """Farthest distance from ``center`` to any point on any inclusion boundary.

    Measured on the true boundary curve (``boundary_points``), which is exact for
    both Circle and Petal — a bounding box would over-report a petal by up to
    ~40 % and inflate the rod radius with it.
    """
    if center is None:
        center = infer_plate_center(configs)
    center_x, center_y = center
    reach = 0.0
    for cfg in configs:
        shape = make_inclusion(cfg)
        # Spacing well below the feature size -> a dense, faithful sample.
        pts = shape.boundary_points(shape.char_length() * 0.02)
        reach = max(reach, float(np.hypot(pts[:, 0] - center_x,
                                          pts[:, 1] - center_y).max()))
    return reach


def infer_plate_radius(configs: List[Dict]) -> float:
    """Infer the outer radius R of the circular cross-section.

    The disk must contain every inclusion with matrix material around it, so R is
    the inclusion's maximum reach from the axis times a 1.45 margin. For the
    shipped petal (base_radius 0.5, amplitude 0.2 -> tip radius 0.7) this gives
    R ~ 1.0, i.e. the petal tips reach 70 % of the outer radius — close enough to
    the rim that the inclusion genuinely stiffens the section in torsion (where
    stiffness is weighted by r^2), and far enough that a matrix shell remains.
    Override with ``--plate-radius``.
    """
    reach = inclusion_max_reach(configs)
    if reach <= 0.0:
        return 1.0
    return float(np.round(reach * 1.45, 4))


def load_inclusion_config(json_path: str) -> Tuple[List[Dict], float]:
    """Load inclusion config (single dict or list of dicts) and infer the
    cross-section radius. Returns the raw dicts (the serializable source of
    truth) plus plate_radius."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        inclusions = [data]
    elif isinstance(data, list):
        inclusions = data
    else:
        raise ValueError("Invalid JSON format. Expected dict or list of dicts.")

    return inclusions, infer_plate_radius(inclusions)


def describe(cfg: Dict) -> str:
    """Short human-readable one-liner for an inclusion config (type-aware)."""
    shape = make_inclusion(cfg)
    center_x, center_y = shape.center()
    shape_type = str(cfg.get('type', 'circle')).lower()
    if shape_type == 'circle':
        return (f"circle: center=({center_x:.4f}, {center_y:.4f}), "
                f"radius={shape.char_length():.4f}")
    if shape_type == 'petal':
        return (f"petal: center=({center_x:.4f}, {center_y:.4f}), "
                f"base_radius={cfg['base_radius']}, "
                f"amplitude={cfg.get('amplitude', 0.0)}, num_petals={cfg['num_petals']}")
    xmin, xmax, ymin, ymax = shape.bbox()
    return (f"{shape_type}: center=({center_x:.4f}, {center_y:.4f}), "
            f"bbox=[{xmin:.3f},{xmax:.3f}]x[{ymin:.3f},{ymax:.3f}]")
