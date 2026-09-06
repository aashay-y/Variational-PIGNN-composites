"""
Adaptive Collocation Point Generator for the Composite Rod Cross-Section
=======================================================================
Generates spatially-varying density collocation points on the **circular
cross-section** of the composite rod: a disk of radius ``plate_radius`` centred
on the torsion axis, containing filled inclusions (circle, petal).

The 3-D mesh is built by extruding this 2-D cross-section along z
(``mesh_generator.CompositeMeshGenerator.extrude_to_3d``), so the
inclusion is a straight prismatic body running the full depth of the rod and
every z-layer shares this identical point set.

Refinement is driven entirely by the inclusion's signed distance field, exactly
as in the original square-plate version — only the outer domain changed from a
square to a disk.

"""

import numpy as np
import json
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt

from inclusion_shapes import (make_inclusion, infer_plate_radius,
                              infer_plate_center, Circle)


class CompositeCollocationGenerator:
    """Adaptive collocation points on the circular cross-section of the rod."""

    def __init__(self,
                 plate_radius: float,
                 inclusions: List[Dict],
                 plate_center: Tuple[float, float] = None,
                 matrix_near_field_factor: float = 3.0,
                 inclusion_refinement_factor: float = 0.5,
                 min_density_factor: float = 0.05,
                 max_density_factor: float = 0.3,
                 growth_rate: float = None):
        """
        Parameters
        ----------
        plate_radius : float
            Outer radius R of the circular cross-section.
        inclusions : List[Dict]
            Inclusion configurations (``type``: 'circle' | 'petal', ...).
        plate_center : (float, float), optional
            Centre of the disk = the torsion axis. Defaults to the centre of the
            union of the inclusion bounding boxes, so a config that describes a
            centred inclusion produces a concentric rod without editing the JSON.
        matrix_near_field_factor : float
            Matrix near-field extends to (char_length x this) outside the inclusion.
        inclusion_refinement_factor : float
            Inclusion refinement zone = char_length x this (inward).
        min_density_factor : float
            Minimum spacing at the inclusion boundary = char_length x this.
        max_density_factor : float
            Maximum spacing in the far field = (2 x plate_radius) x this. The
            factor multiplies the DIAMETER, so it scales with the overall size of
            the cross-section rather than its radius.
        growth_rate : float, optional
            Element growth rate for the fine->coarse transition (~fractional size
            increase per ring of elements). ``None`` spans the near-field band.
        """
        self.plate_radius = float(plate_radius)
        self.inclusions = inclusions
        # Geometry layer: one shape object per inclusion config (dispatched on the JSON "type"; defaults to circle for backward compatibility).
        self.shapes = [make_inclusion(inc) for inc in inclusions]

        if plate_center is None:
            plate_center = infer_plate_center(inclusions)
        self.axis_x, self.axis_y = float(plate_center[0]), float(plate_center[1])

        self.rng = np.random.default_rng()
        self.matrix_near_field_factor = matrix_near_field_factor
        self.inclusion_refinement_factor = inclusion_refinement_factor
        self.min_density_factor = min_density_factor
        self.max_density_factor = max_density_factor
        self.growth_rate = growth_rate

        # Derived parameters
        self.max_spacing = 2.0 * self.plate_radius * max_density_factor

        # Storage for generated points
        self.boundary_points = []
        self.inclusion_boundary_points = []
        self.matrix_interior_points = []
        self.inclusion_interior_points = []
        self.all_points = []

        print("="*60)
        print("COMPOSITE CROSS-SECTION COLLOCATION POINT GENERATOR")
        print("="*60)
        print(f"Cross-section: disk R={self.plate_radius:.4f} "
              f"centred at ({self.axis_x:.4f}, {self.axis_y:.4f})")
        print(f"Number of inclusions: {len(inclusions)}")
        print(f"Matrix near-field factor: {matrix_near_field_factor}")
        print(f"Inclusion refinement factor: {inclusion_refinement_factor}")
        print(f"Min spacing factor: {min_density_factor}")
        print(f"Max spacing factor: {max_density_factor}")
        print(f"Max spacing (far-field): {self.max_spacing:.4f}")
        print()

    # ── domain membership ────────────────────────────────────────────────────

    def is_inside_plate(self, x: float, y: float, tolerance: float = 0.0) -> bool:
        """Inside the circular cross-section?"""
        return (x - self.axis_x) ** 2 + (y - self.axis_y) ** 2 <= (self.plate_radius + tolerance) ** 2

    def density_function(self, x: float, y: float) -> float:
        """
        Local point spacing at (x, y).

        - Inside an inclusion: refined near the boundary, coarser toward the centre
        - In the matrix: refined near the inclusion boundary, coarser in the far field
        - At the interface: finest spacing
        """
        if not self.shapes:
            return self.max_spacing

        radii = []

        for shape in self.shapes:
            # Signed distance to this inclusion's boundary (<0 inside, >0 outside).
            # The refinement bands are scaled by the shape's characteristic length,
            # reproducing the original circle behavior (where char_length == radius).
            cl = shape.char_length()
            sd = float(shape.signed_distance(x, y))

            # Minimum spacing at boundary
            min_spacing = cl * self.min_density_factor

            if sd < 0.0:
                # INSIDE INCLUSION: refinement zone from boundary inward
                dist_from_boundary = -sd
                band = cl * self.inclusion_refinement_factor
            else:
                # IN MATRIX: refinement zone from boundary outward.
                dist_from_boundary = sd
                band = cl * (self.matrix_near_field_factor - 1.0)

            # Growth-rate-limited (geometric) size field: spacing(d) = min_spacing + growth * d   (capped at max_spacing)
            # A spacing growing linearly with distance from the boundary is exactly a geometric growth per element, which is what produces a smooth fine->coarse transition.
            if self.growth_rate is not None and self.growth_rate > 0.0:
                growth = self.growth_rate
            elif band > 0.0:
                growth = (self.max_spacing - min_spacing) / band
            else:
                growth = 0.0

            spacing = min(min_spacing + growth * dist_from_boundary, self.max_spacing)
            radii.append(spacing)

        # Use minimum (most restrictive) spacing
        return min(radii)

    def is_inside_inclusion(self, x: float, y: float, tolerance: float = 1e-6) -> Tuple[bool, int]:
        """Is the point strictly inside any inclusion? Returns (bool, index)."""
        for i, shape in enumerate(self.shapes):
            if shape.is_strictly_inside(x, y, tolerance):
                return True, i
        return False, -1

    def is_on_inclusion_boundary(self, x: float, y: float, tolerance: float = 1e-3) -> bool:
        """Check if point is on any inclusion boundary"""
        return any(shape.is_on_boundary(x, y, tolerance) for shape in self.shapes)

    def generate_plate_boundary_points(self) -> List[Tuple[float, float]]:
        """
        Points on the circular rim, spaced by the local density function.

        Walking the circumference by the local spacing gives an adaptive rim; the
        count is then rounded to an integer so the last point does not land on
        top of the first, which would produce a degenerate Delaunay triangle.
        """
        print("Generating cross-section rim points...")
        R = self.plate_radius
        circumference = 2.0 * np.pi * R

        # Average spacing around the rim (the density field is smooth there, so a
        # small probe set is enough to size the point count).
        probe = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        spacings = [self.density_function(self.axis_x + R * np.cos(t),
                                          self.axis_y + R * np.sin(t)) for t in probe]
        mean_spacing = float(np.mean(spacings))
        n_points = max(24, int(np.ceil(circumference / mean_spacing)))

        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        boundary_points = [(float(self.axis_x + R * np.cos(a)),
                            float(self.axis_y + R * np.sin(a))) for a in angles]

        print(f"  Generated {len(boundary_points)} rim points "
              f"(mean spacing {mean_spacing:.4f}, arc spacing {circumference/n_points:.4f})")
        return boundary_points

    def generate_inclusion_boundary_points(self) -> List[Tuple[float, float]]:
        """Generate points on inclusion boundaries"""
        print("Generating inclusion boundary points...")
        boundary_points = []

        for i, shape in enumerate(self.shapes):
            # Spacing on boundary (scaled by characteristic length, as before)
            local_spacing = shape.char_length() * self.min_density_factor

            # Points sampled along the true boundary curve (arc-length spaced)
            pts = shape.boundary_points(local_spacing)
            boundary_points.extend((float(px), float(py)) for px, py in pts)

            print(f"  Inclusion {i+1}: {len(pts)} points "
                  f"(char_length={shape.char_length():.2f})")

        print(f"  Total inclusion boundary points: {len(boundary_points)}")
        return boundary_points

    def poisson_disk_sampling(self,
                              region_type: str = 'matrix',
                              num_candidates: int = 30,
                              max_attempts: int = 400000) -> List[Tuple[float, float]]:
        """
        Generate interior points using Poisson disk sampling.

        ``region_type`` is 'matrix' (inside the disk, outside every inclusion) or
        'inclusion' (inside an inclusion).

        NOTE ON ``max_attempts``: filling a region with N points costs roughly 2N
        iterations, so a region needing more than ~max_attempts/2 points comes
        back SILENTLY PARTLY FILLED. ``generate`` checks the realised count
        against the area estimate and warns loudly if it falls short.
        """
        region_name = region_type.capitalize()
        print(f"Generating {region_name} interior points (Poisson disk sampling)...")

        # Grid for spatial hashing
        cell_size = self.max_spacing
        grid = {}

        def grid_coords(x, y):
            """Map a point to its spatial-hash cell index ``(gx, gy)``."""
            return (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))

        def get_neighbors(gx, gy):
            """Collect points in the 5x5 cell block centred on ``(gx, gy)``.

            The block spans two cells in each direction because the rejection
            radius can reach ``max_spacing``, so a conflicting point may sit up
            to two cells away.
            """
            neighbors = []
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    key = (gx + dx, gy + dy)
                    if key in grid:
                        neighbors.extend(grid[key])
            return neighbors

        # Initialize with boundary points
        points = []
        active = []

        # Add existing boundaries to grid
        for pt in self.boundary_points + self.inclusion_boundary_points:
            gx, gy = grid_coords(pt[0], pt[1])
            if (gx, gy) not in grid:
                grid[(gx, gy)] = []
            grid[(gx, gy)].append(pt)

        # Find initial seed point(s)
        # For inclusion mode: seed each inclusion independently so all get populated
        if region_type == 'inclusion':
            if not self.shapes:
                return []
            for shape in self.shapes:
                seeded = False
                for _ in range(1000):
                    # Shape-aware interior seed (guaranteed strictly inside)
                    x, y = shape.sample_interior_seed(self.rng)
                    inside, _ = self.is_inside_inclusion(x, y)
                    if inside:
                        gx, gy = grid_coords(x, y)
                        neighbors = get_neighbors(gx, gy)
                        valid = all(
                            np.sqrt((x - nx)**2 + (y - ny)**2) >= self.density_function(x, y)
                            for nx, ny in neighbors
                        )
                        if valid:
                            pt = (x, y)
                            points.append(pt)
                            active.append(pt)
                            if (gx, gy) not in grid:
                                grid[(gx, gy)] = []
                            grid[(gx, gy)].append(pt)
                            seeded = True
                            break
                if not seeded:
                    inc_x, inc_y = shape.center()
                    print(f"  Warning: Could not seed inclusion at "
                          f"({inc_x:.3f}, {inc_y:.3f})")
        else:
            # Matrix: single seed anywhere inside the disk but outside the inclusions
            seeded = False
            for _ in range(2000):
                # Uniform in the disk (sqrt keeps the density uniform in area).
                a = np.random.uniform(0.0, 2.0 * np.pi)
                rr = self.plate_radius * np.sqrt(np.random.uniform(0.0, 1.0))
                x = self.axis_x + rr * np.cos(a)
                y = self.axis_y + rr * np.sin(a)
                inside, _ = self.is_inside_inclusion(x, y)
                if not inside:
                    gx, gy = grid_coords(x, y)
                    neighbors = get_neighbors(gx, gy)
                    valid = all(
                        np.sqrt((x - nx)**2 + (y - ny)**2) >= self.density_function(x, y)
                        for nx, ny in neighbors
                    )
                    if valid:
                        pt = (x, y)
                        points.append(pt)
                        active.append(pt)
                        if (gx, gy) not in grid:
                            grid[(gx, gy)] = []
                        grid[(gx, gy)].append(pt)
                        seeded = True
                        break
            if not seeded:
                print(f"  Warning: Could not find initial {region_type} point")
                return []

        # Main Poisson disk sampling loop
        iteration = 0
        last_report = 0

        while active and iteration < max_attempts:
            idx = np.random.randint(len(active))
            px, py = active[idx]

            local_r = self.density_function(px, py)

            found = False
            for _ in range(num_candidates):
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(local_r, 2*local_r)

                x = px + radius * np.cos(angle)
                y = py + radius * np.sin(angle)

                # Check if in valid region
                valid_region = False
                if region_type == 'matrix':
                    if self.is_inside_plate(x, y):
                        inside, _ = self.is_inside_inclusion(x, y)
                        if not inside:
                            valid_region = True
                else:  # inclusion
                    inside, _ = self.is_inside_inclusion(x, y)
                    if inside:
                        valid_region = True

                if valid_region:
                    req_dist = self.density_function(x, y)
                    gx, gy = grid_coords(x, y)
                    neighbors = get_neighbors(gx, gy)

                    valid = True
                    for nx, ny in neighbors:
                        dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                        min_req = min(req_dist, self.density_function(nx, ny))
                        if dist < min_req * 0.9:
                            valid = False
                            break

                    if valid:
                        pt = (x, y)
                        points.append(pt)
                        active.append(pt)
                        if (gx, gy) not in grid:
                            grid[(gx, gy)] = []
                        grid[(gx, gy)].append(pt)
                        found = True
                        break

            if not found:
                active.pop(idx)

            iteration += 1

            if len(points) - last_report >= 250:
                print(f"  Generated {len(points)} points (active: {len(active)})...")
                last_report = len(points)

        if iteration >= max_attempts:
            print(f"  WARNING: hit max_attempts={max_attempts} with {len(active)} "
                  f"points still active — this region may be PARTLY FILLED.")

        print(f"  Generated {len(points)} {region_type} interior points")
        return points

    def generate(self) -> np.ndarray:
        """Generate all cross-section collocation points -> (N, 2)."""
        print("\n" + "="*60)
        print("GENERATING CROSS-SECTION COLLOCATION POINTS")
        print("="*60)

        # 1. Outer rim of the cross-section
        self.boundary_points = self.generate_plate_boundary_points()

        # 2. Inclusion boundaries
        self.inclusion_boundary_points = self.generate_inclusion_boundary_points()

        # 3. Matrix interior
        self.matrix_interior_points = self.poisson_disk_sampling(region_type='matrix')

        # 4. Inclusion interior
        self.inclusion_interior_points = self.poisson_disk_sampling(region_type='inclusion')

        # Combine all points
        self.all_points = (self.boundary_points +
                          self.inclusion_boundary_points +
                          self.matrix_interior_points +
                          self.inclusion_interior_points)

        n_total = len(self.all_points)

        print("\n" + "="*60)
        print("COLLOCATION POINT GENERATION COMPLETE")
        print("="*60)
        print(f"Rim points:                 {len(self.boundary_points)}")
        print(f"Inclusion boundary points:  {len(self.inclusion_boundary_points)}")
        print(f"Matrix interior points:     {len(self.matrix_interior_points)}")
        print(f"Inclusion interior points:  {len(self.inclusion_interior_points)}")
        print(f"Total points:               {n_total}")

        # Sanity check against the area a uniform fill would need. The sampler
        # can stop early (see poisson_disk_sampling) and returns a partly-filled
        # region with no exception, which would train and compare happily while
        # reporting a meaningless answer.
        area = np.pi * self.plate_radius ** 2
        h_mean = self.max_spacing
        expected = 0.8 * area / (h_mean ** 2)
        if n_total < 0.5 * expected:
            print(f"WARNING: {n_total} points is well below the ~{expected:.0f} a "
                  f"uniform fill at h={h_mean:.4f} would need. The Poisson sampler "
                  f"may have stopped early — check the point plot before training.")
        print("="*60)

        return np.array(self.all_points, dtype=np.float64)

    # ── visualisation ────────────────────────────────────────────────────────

    def visualize_density_field(self, output_path='density_field_composite.png'):
        """Visualize the density function as a heatmap over the cross-section."""
        print("\nGenerating density field visualization...")

        n_grid = 200
        R = self.plate_radius
        x = np.linspace(self.axis_x - R, self.axis_x + R, n_grid)
        y = np.linspace(self.axis_y - R, self.axis_y + R, n_grid)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(n_grid):
            for j in range(n_grid):
                Z[i, j] = self.density_function(X[i, j], Y[i, j])

        # Mask everything outside the disk so the plot shows the real domain.
        outside = (X - self.axis_x) ** 2 + (Y - self.axis_y) ** 2 > R ** 2
        Z = np.ma.array(Z, mask=outside)

        fig, ax = plt.subplots(figsize=(10, 9))

        im = ax.contourf(X, Y, Z, levels=20, cmap='viridis_r')
        plt.colorbar(im, ax=ax, label='Local Spacing (smaller = finer mesh)')

        # Cross-section rim
        ax.add_patch(plt.Circle((self.axis_x, self.axis_y), R, fill=False,
                                edgecolor='black', linewidth=2))

        # Draw inclusions
        for shape in self.shapes:
            shape.draw(ax, edgecolor='red', linewidth=2)

            # Near-field / refinement rings are circle-specific visual aids
            if isinstance(shape, Circle):
                inc_x, inc_y, inc_r = (shape.center_x, shape.center_y,
                                       shape.radius)
                ax.add_patch(plt.Circle((inc_x, inc_y),
                                        inc_r * self.matrix_near_field_factor,
                                        fill=False, edgecolor='orange', linewidth=1,
                                        linestyle='--', alpha=0.5))
                inc_ref = inc_r * (1.0 - self.inclusion_refinement_factor)
                if inc_ref > 0.1:
                    ax.add_patch(plt.Circle((inc_x, inc_y), inc_ref, fill=False,
                                            edgecolor='cyan', linewidth=1,
                                            linestyle=':', alpha=0.5))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Adaptive Density Field — rod cross-section\n'
                     '(Black=rim, Red=inclusion boundary)')
        ax.set_aspect('equal')
        ax.set_xlim(self.axis_x - 1.05 * R, self.axis_x + 1.05 * R)
        ax.set_ylim(self.axis_y - 1.05 * R, self.axis_y + 1.05 * R)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    def visualize_points(self, points: np.ndarray, output_path='collocation_points_composite.png'):
        """Visualize generated collocation points"""
        print("\nGenerating collocation points visualization...")

        fig, ax = plt.subplots(figsize=(12, 12))

        # Separate point types
        n_boundary = len(self.boundary_points)
        n_inc_boundary = len(self.inclusion_boundary_points)
        n_matrix = len(self.matrix_interior_points)

        boundary_pts = points[:n_boundary]
        inc_boundary_pts = points[n_boundary:n_boundary + n_inc_boundary]
        matrix_pts = points[n_boundary + n_inc_boundary:n_boundary + n_inc_boundary + n_matrix]
        inclusion_pts = points[n_boundary + n_inc_boundary + n_matrix:]

        # Plot points
        if len(matrix_pts) > 0:
            ax.plot(matrix_pts[:, 0], matrix_pts[:, 1], 'b.',
                   markersize=2, label=f'Matrix interior ({len(matrix_pts)})', alpha=0.6)

        if len(inclusion_pts) > 0:
            ax.plot(inclusion_pts[:, 0], inclusion_pts[:, 1], 'c.',
                   markersize=2, label=f'Inclusion interior ({len(inclusion_pts)})', alpha=0.6)

        if len(boundary_pts) > 0:
            ax.plot(boundary_pts[:, 0], boundary_pts[:, 1], 'ro',
                   markersize=4, label=f'Cross-section rim ({len(boundary_pts)})')

        if len(inc_boundary_pts) > 0:
            ax.plot(inc_boundary_pts[:, 0], inc_boundary_pts[:, 1], 'go',
                   markersize=3, label=f'Inclusion boundary ({len(inc_boundary_pts)})', alpha=0.7)

        # Draw inclusions
        for shape in self.shapes:
            shape.draw(ax, edgecolor='black', linewidth=2)

        R = self.plate_radius
        ax.plot(self.axis_x, self.axis_y, 'k+', markersize=14, label='torsion axis')

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Cross-Section Collocation Points (Total: {len(points)})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(self.axis_x - 1.1 * R, self.axis_x + 1.1 * R)
        ax.set_ylim(self.axis_y - 1.1 * R, self.axis_y + 1.1 * R)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def load_inclusion_config(json_path: str) -> Tuple[List[Dict], float]:
    """
    Load inclusion configuration from a JSON file (single dict or list of dicts).

    Returns (inclusions, plate_radius) where plate_radius is inferred from the
    inclusion extent (see inclusion_shapes.infer_plate_radius).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict):
        inclusions = [data]
    elif isinstance(data, list):
        inclusions = data
    else:
        raise ValueError("Invalid JSON format. Expected dict or list.")

    return inclusions, infer_plate_radius(inclusions)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python collocation_generator.py <config.json> [plate_radius]")
        print("Example: python collocation_generator.py petal_config.json 1.0")
        sys.exit(1)

    json_file = sys.argv[1]

    inclusions, inferred_radius = load_inclusion_config(json_file)
    plate_radius = float(sys.argv[2]) if len(sys.argv) > 2 else inferred_radius

    print(f"Loaded {len(inclusions)} inclusions")
    print(f"Cross-section radius: {plate_radius}")

    generator = CompositeCollocationGenerator(
        plate_radius=plate_radius,
        inclusions=inclusions,
        matrix_near_field_factor=2.0,
        inclusion_refinement_factor=0.5,
        min_density_factor=0.05,
        max_density_factor=0.1
    )

    points = generator.generate()

    generator.visualize_density_field()
    generator.visualize_points(points)

    np.save('collocation_points_composite.npy', points)
    print(f"\n[ok] Saved collocation_points_composite.npy: {points.shape}")
