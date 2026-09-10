"""
Adaptive Collocation Point Generator for Plates with Holes (Voids)
==================================================================
Generates spatially-varying density collocation points for a plate with
circular holes. The mesh is refined near each hole boundary (the stress
concentrates there) and coarsens into the far field.

Points are seeded everywhere — including inside the hole regions — so the
Delaunay triangulation is well graded right up to the rim; the interior (void)
triangles and their orphaned nodes are removed later, at the mesh-generation
stage. Everything that survives is matrix.

Author: Claude
Date: 2026-04-24
"""

import numpy as np
import json
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use("Agg")   # headless backend (see inclusion_shapes.py for why).
import matplotlib.pyplot as plt

from inclusion_shapes import make_inclusion, infer_plate_size, Circle


class CompositeCollocationGenerator:
    """Generate adaptive density collocation points for a plate with holes"""
    
    def __init__(self, 
                 plate_size: float,
                 inclusions: List[Dict],
                 matrix_near_field_factor: float = 3.0,
                 inclusion_refinement_factor: float = 0.5,
                 min_density_factor: float = 0.05,
                 max_density_factor: float = 0.3,
                 growth_rate: float = None):
        """
        Initialize composite collocation point generator

        Parameters:
        -----------
        plate_size : float
            Size of square plate
        inclusions : List[Dict]
            List of hole configurations with 'center_x', 'center_y', 'radius'
        matrix_near_field_factor : float
            Matrix near-field extends to (hole_radius x this) from center (outside)
        inclusion_refinement_factor : float
            Hole-interior refinement zone = hole_radius x this (inside; these
            points are seeded then removed with the void at mesh time)
        min_density_factor : float
            Minimum spacing at the hole boundary = hole_radius x this
        max_density_factor : float
            Maximum spacing in far-field = plate_size x this
        growth_rate : float, optional
            Element growth rate for the fine->coarse transition. The local spacing
            grows by roughly this fraction per element away from the boundary
            (e.g. 0.15 -> each ring of elements is ~15% larger than the previous
            one), capped at the far-field ``max_spacing``. Small values give a
            wider, smoother transition. When ``None`` (default), the growth rate
            is derived so the ramp spans exactly the near-field band, reproducing
            the legacy behaviour but with a guaranteed-smooth linear grade.
        """
        self.plate_size = plate_size
        self.inclusions = inclusions
        # Geometry layer: one shape object per inclusion config (dispatched on
        # the JSON "type"; defaults to circle for backward compatibility).
        self.shapes = [make_inclusion(inc) for inc in inclusions]
        self.rng = np.random.default_rng()
        self.matrix_near_field_factor = matrix_near_field_factor
        self.inclusion_refinement_factor = inclusion_refinement_factor
        self.min_density_factor = min_density_factor
        self.max_density_factor = max_density_factor
        self.growth_rate = growth_rate
        
        # Derived parameters
        self.max_spacing = plate_size * max_density_factor
        
        # Storage for generated points
        self.boundary_points = []
        self.inclusion_boundary_points = []
        self.matrix_interior_points = []
        self.inclusion_interior_points = []
        self.all_points = []
        
        print("="*60)
        print("HOLE-PLATE COLLOCATION POINT GENERATOR")
        print("="*60)
        print(f"Plate size: {plate_size}x{plate_size}")
        print(f"Number of holes: {len(inclusions)}")
        print(f"Matrix near-field factor: {matrix_near_field_factor}")
        print(f"Hole-interior refinement factor: {inclusion_refinement_factor}")
        print(f"Min spacing factor: {min_density_factor}")
        print(f"Max spacing factor: {max_density_factor}")
        print(f"Max spacing (far-field): {self.max_spacing:.4f}")
        print()
        
    def density_function(self, x: float, y: float) -> float:
        """
        Compute local density radius at point (x, y)
        
        For a plate with holes:
        - Inside hole: refined near boundary, coarser toward center (these
          interior points are removed with the void at mesh time)
        - In matrix: refined near the hole boundary, coarser in far-field
        - At the hole boundary: finest mesh (minimum spacing)
        
        Returns:
        --------
        radius : float
            Minimum spacing required at this location
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
                # band == (radius*factor - radius) == char_length*(factor-1)
                dist_from_boundary = sd
                band = cl * (self.matrix_near_field_factor - 1.0)

            # Growth-rate-limited (geometric) size field.
            #
            # spacing(d) = min_spacing + growth * d  (capped at max_spacing)
            #
            # A spacing that grows linearly with distance from the boundary is
            # exactly a *geometric* growth per element: each successive ring of
            # points is a factor (1 + growth) larger than the one before it. That
            # bounded per-element ratio is what produces a smooth, gradual
            # fine->coarse transition (instead of the old fixed-width cosine ramp,
            # which forced the whole min->max jump into the near-field band and
            # looked abrupt whenever min<<max). The transition width is no longer
            # hard-capped: the mesh keeps coarsening until it reaches max_spacing.
            if self.growth_rate is not None and self.growth_rate > 0.0:
                growth = self.growth_rate
            elif band > 0.0:
                # Legacy-compatible default: span the near-field band exactly,
                # but as a smooth linear (geometric) grade rather than a cosine.
                growth = (self.max_spacing - min_spacing) / band
            else:
                growth = 0.0

            spacing = min(min_spacing + growth * dist_from_boundary, self.max_spacing)
            radii.append(spacing)

        # Use minimum (most restrictive) spacing
        return min(radii)
    
    def is_inside_inclusion(self, x: float, y: float, tolerance: float = 1e-6) -> Tuple[bool, int]:
        """
        Check if point is inside any inclusion
        
        Returns:
        --------
        inside : bool
            True if inside an inclusion
        inclusion_id : int
            Index of inclusion (-1 if not inside any)
        """
        for i, shape in enumerate(self.shapes):
            if shape.is_strictly_inside(x, y, tolerance):
                return True, i
        return False, -1

    def is_on_inclusion_boundary(self, x: float, y: float, tolerance: float = 1e-3) -> bool:
        """Check if point is on any inclusion boundary"""
        return any(shape.is_on_boundary(x, y, tolerance) for shape in self.shapes)
    
    def generate_plate_boundary_points(self) -> List[Tuple[float, float]]:
        """Generate points on the plate boundary with adaptive spacing"""
        print("Generating plate boundary points...")
        boundary_points = []
        
        edges = [
            ('bottom', [(0, 0), (self.plate_size, 0)]),
            ('right',  [(self.plate_size, 0), (self.plate_size, self.plate_size)]),
            ('top',    [(self.plate_size, self.plate_size), (0, self.plate_size)]),
            ('left',   [(0, self.plate_size), (0, 0)])
        ]
        
        for edge_name, (start, end) in edges:
            edge_pts = [start]
            current_pos = 0.0
            edge_length = self.plate_size
            
            while current_pos < edge_length - 1e-6:
                t = current_pos / edge_length
                if edge_name in ['bottom', 'top']:
                    x = start[0] + t * (end[0] - start[0])
                    y = start[1]
                else:
                    x = start[0]
                    y = start[1] + t * (end[1] - start[1])
                
                local_spacing = self.density_function(x, y)
                current_pos += local_spacing
                
                if current_pos < edge_length - 1e-6:
                    t = current_pos / edge_length
                    if edge_name in ['bottom', 'top']:
                        next_x = start[0] + t * (end[0] - start[0])
                        next_y = start[1]
                    else:
                        next_x = start[0]
                        next_y = start[1] + t * (end[1] - start[1])
                    edge_pts.append((next_x, next_y))
            
            if edge_name != 'left':
                edge_pts.append(end)
            
            boundary_points.extend(edge_pts)
        
        print(f"  Generated {len(boundary_points)} boundary points")
        return boundary_points
    
    def generate_inclusion_boundary_points(self) -> List[Tuple[float, float]]:
        """Generate points on the hole boundaries"""
        print("Generating hole boundary points...")
        boundary_points = []

        for i, shape in enumerate(self.shapes):
            # Spacing on boundary (scaled by characteristic length, as before)
            local_spacing = shape.char_length() * self.min_density_factor

            # Points sampled along the true boundary curve (arc-length spaced)
            pts = shape.boundary_points(local_spacing)
            boundary_points.extend((float(px), float(py)) for px, py in pts)

            print(f"  Hole {i+1}: {len(pts)} points "
                  f"(char_length={shape.char_length():.2f})")

        print(f"  Total hole boundary points: {len(boundary_points)}")
        return boundary_points
    
    def poisson_disk_sampling(self,
                              region_type: str = 'matrix',
                              num_candidates: int = 30,
                              max_attempts: int = 100000) -> List[Tuple[float, float]]:
        """
        Generate interior points using Poisson disk sampling
        
        Parameters:
        -----------
        region_type : str
            'matrix' or 'inclusion' - which region to fill
        num_candidates : int
            Candidates per active point
        max_attempts : int
            Maximum iterations
            
        Returns:
        --------
        points : List[Tuple[float, float]]
            Generated interior points
        """
        # The 'inclusion' region label denotes the hole interior; those points
        # are seeded here and removed with the void at the mesh stage.
        region_name = 'Hole' if region_type == 'inclusion' else region_type.capitalize()
        print(f"Generating {region_name} interior points (Poisson disk sampling)...")
        
        # Grid for spatial hashing
        cell_size = self.max_spacing
        grid = {}
        
        def grid_coords(x, y):
            return (int(x / cell_size), int(y / cell_size))
        
        def get_neighbors(gx, gy):
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
                    cx, cy = shape.center()
                    print(f"  Warning: Could not seed hole at ({cx:.3f}, {cy:.3f})")
        else:
            # Matrix: single seed anywhere in the matrix
            seeded = False
            for _ in range(1000):
                x = np.random.uniform(0, self.plate_size)
                y = np.random.uniform(0, self.plate_size)
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
                    if 0 <= x <= self.plate_size and 0 <= y <= self.plate_size:
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
            
            if len(points) - last_report >= 100:
                print(f"  Generated {len(points)} points (active: {len(active)})...")
                last_report = len(points)
        
        print(f"  Generated {len(points)} {region_type} interior points")
        return points
    
    def generate(self) -> np.ndarray:
        """
        Generate all collocation points for the hole-plate mesh

        Returns:
        --------
        points : np.ndarray
            (N, 2) array of point coordinates
        """
        print("\n" + "="*60)
        print("GENERATING COLLOCATION POINTS FOR HOLE PLATE")
        print("="*60)

        # 1. Plate boundary
        self.boundary_points = self.generate_plate_boundary_points()

        # 2. Hole boundaries
        self.inclusion_boundary_points = self.generate_inclusion_boundary_points()

        # 3. Matrix interior
        self.matrix_interior_points = self.poisson_disk_sampling(region_type='matrix')

        # 4. Hole interior (seeded for a well-graded Delaunay; removed at mesh time)
        self.inclusion_interior_points = self.poisson_disk_sampling(region_type='inclusion')
        
        # Combine all points
        self.all_points = (self.boundary_points +
                          self.inclusion_boundary_points +
                          self.matrix_interior_points +
                          self.inclusion_interior_points)
        
        print("\n" + "="*60)
        print("COLLOCATION POINT GENERATION COMPLETE")
        print("="*60)
        print(f"Plate boundary points:      {len(self.boundary_points)}")
        print(f"Hole boundary points:       {len(self.inclusion_boundary_points)}")
        print(f"Matrix interior points:     {len(self.matrix_interior_points)}")
        print(f"Hole interior points:       {len(self.inclusion_interior_points)} (removed at mesh time)")
        print(f"Total points:               {len(self.all_points)}")
        print("="*60)
        
        return np.array(self.all_points, dtype=np.float64)
    
    def visualize_density_field(self, output_path='density_field_composite.png'):
        """Visualize the density function as a heatmap"""
        print("\nGenerating density field visualization...")
        
        n_grid = 200
        x = np.linspace(0, self.plate_size, n_grid)
        y = np.linspace(0, self.plate_size, n_grid)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(n_grid):
            for j in range(n_grid):
                Z[i, j] = self.density_function(X[i, j], Y[i, j])
        
        fig, ax = plt.subplots(figsize=(10, 9))
        
        im = ax.contourf(X, Y, Z, levels=20, cmap='viridis_r')
        plt.colorbar(im, ax=ax, label='Local Spacing (smaller = finer mesh)')
        
        # Draw holes
        for shape in self.shapes:
            # Hole boundary
            shape.draw(ax, edgecolor='red', linewidth=2)

            # Near-field / refinement rings are circle visual aids
            if isinstance(shape, Circle):
                cx, cy, r = shape.cx, shape.cy, shape.r
                ax.add_patch(plt.Circle((cx, cy), r * self.matrix_near_field_factor,
                                        fill=False, edgecolor='orange', linewidth=1,
                                        linestyle='--', alpha=0.5))
                inc_ref = r * (1.0 - self.inclusion_refinement_factor)
                if inc_ref > 0.1:
                    ax.add_patch(plt.Circle((cx, cy), inc_ref, fill=False,
                                            edgecolor='cyan', linewidth=1,
                                            linestyle=':', alpha=0.5))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Adaptive Density Field - Hole Plate\n' +
                    '(Red=hole boundary, Orange=matrix near-field, Cyan=hole-interior refinement)')
        ax.set_aspect('equal')
        ax.set_xlim(0, self.plate_size)
        ax.set_ylim(0, self.plate_size)
        
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
                   markersize=2, label=f'Hole interior ({len(inclusion_pts)})', alpha=0.6)
        
        if len(boundary_pts) > 0:
            ax.plot(boundary_pts[:, 0], boundary_pts[:, 1], 'ro', 
                   markersize=4, label=f'Plate boundary ({len(boundary_pts)})')
        
        if len(inc_boundary_pts) > 0:
            ax.plot(inc_boundary_pts[:, 0], inc_boundary_pts[:, 1], 'go',
                   markersize=3, label=f'Hole boundary ({len(inc_boundary_pts)})', alpha=0.7)

        # Draw holes
        for shape in self.shapes:
            shape.draw(ax, edgecolor='black', linewidth=2)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Generated Collocation Points - Hole Plate (Total: {len(points)})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, self.plate_size + 0.5)
        ax.set_ylim(-0.5, self.plate_size + 0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()


def load_inclusion_config(json_path: str) -> Tuple[List[Dict], float]:
    """
    Load inclusion configuration from JSON file
    
    Supports both single inclusion and multiple inclusions formats.
    
    Returns:
    --------
    inclusions : List[Dict]
        List of inclusion configurations
    plate_size : float
        Inferred plate size
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if it's a single inclusion or list
    if isinstance(data, dict):
        # Single inclusion format
        inclusions = [data]
    elif isinstance(data, list):
        # Multiple inclusions format
        inclusions = data
    else:
        raise ValueError("Invalid JSON format. Expected dict or list.")

    # Infer plate size from the union of inclusion bounding boxes
    # (shape-agnostic: works for circle, petal, ... via inclusion_shapes).
    plate_size = infer_plate_size(inclusions)

    return inclusions, plate_size


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python collocation_generator.py <config.json> [plate_size]")
        print("Example: python collocation_generator.py hole_config.json 20")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Load configuration
    inclusions, inferred_size = load_inclusion_config(json_file)
    plate_size = float(sys.argv[2]) if len(sys.argv) > 2 else inferred_size
    
    print(f"Loaded {len(inclusions)} holes")
    print(f"Plate size: {plate_size}x{plate_size}")

    # Create generator
    generator = CompositeCollocationGenerator(
        plate_size=plate_size,
        inclusions=inclusions,
        matrix_near_field_factor=2.0,
        inclusion_refinement_factor=0.5,
        min_density_factor=0.05,
        max_density_factor=0.1
    )
    
    # Generate points
    points = generator.generate()
    
    # Visualizations
    generator.visualize_density_field()
    generator.visualize_points(points)
    
    # Save points
    np.save('collocation_points_composite.npy', points)
    print(f"\n[ok] Saved collocation_points_composite.npy: {points.shape}")
