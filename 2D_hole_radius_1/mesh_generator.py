"""
Hole-Plate Mesh Generator
=========================
Generates the triangular mesh for a plate with circular holes (voids).

The circular shapes are voids: triangles whose centroid falls inside a hole are
*removed*, orphaned nodes are dropped and the mesh is renumbered.  Each hole
boundary then becomes an exterior traction-free free surface.  Everything that
remains is matrix (a single homogeneous phase).

The geometry is driven entirely by the ``inclusion_shapes.signed_distance``
primitive (circular holes only in this pipeline).

``generate_mesh`` returns the (nodes, elements) arrays directly; there is no
intermediate DOLFIN-XML file. ``CompositeMeshProcessor`` consumes those arrays
and writes ``composite_mesh.vtu``, which is also what the FEM reads, so the mesh
the trainer sees and the mesh the FEM solves on cannot drift apart.

Author: Claude
Date: 2026-04-24
"""

import numpy as np
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use("Agg")   # headless backend (see inclusion_shapes.py for why).
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from inclusion_shapes import make_inclusion


class CompositeMeshGenerator:
    """Generate the Delaunay triangulation for a plate with circular holes."""

    def __init__(self, points: np.ndarray, inclusions: List[Dict], plate_size: float):
        """
        Initialize the hole-plate mesh generator

        Parameters:
        -----------
        points : np.ndarray
            (N, 2) array of collocation points
        inclusions : List[Dict]
            List of hole configurations (circular voids).
        plate_size : float
            Size of square plate
        """
        self.points = points
        self.inclusions = inclusions
        # Geometry layer: one shape object per hole (circular voids).
        self.shapes = [make_inclusion(inc) for inc in inclusions]
        self.plate_size = plate_size

        self.triangles = None
        self.num_nodes = len(points)
        self.num_elements = 0

        # Element/node sets. After the voids are removed everything is matrix;
        # the inclusion/interface sets stay empty (kept so the shared quality /
        # visualization helpers can iterate over them uniformly).
        self.matrix_elements = set()
        self.inclusion_elements = set()
        self.interface_elements = set()

        self.matrix_nodes = set()
        self.inclusion_nodes = set()

        print("\n" + "="*60)
        print("HOLE-PLATE MESH GENERATOR")
        print("="*60)
        print(f"Input points: {self.num_nodes}")
        print(f"Holes: {len(inclusions)}")
        print()
    
    def compute_triangle_quality(self, tri_indices: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for a triangle"""
        vertices = self.points[tri_indices]
        
        # Edge lengths
        edges = np.array([
            np.linalg.norm(vertices[1] - vertices[0]),
            np.linalg.norm(vertices[2] - vertices[1]),
            np.linalg.norm(vertices[0] - vertices[2])
        ])
        
        # Area
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        area = 0.5 * abs(np.cross(v1, v2))
        
        # Angles
        a, b, c = edges
        angles = np.array([
            np.arccos(np.clip((b**2 + c**2 - a**2) / (2*b*c), -1, 1)),
            np.arccos(np.clip((a**2 + c**2 - b**2) / (2*a*c), -1, 1)),
            np.arccos(np.clip((a**2 + b**2 - c**2) / (2*a*b), -1, 1))
        ]) * 180 / np.pi
        
        aspect_ratio = edges.max() / edges.min() if edges.min() > 0 else np.inf
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'min_angle': angles.min(),
            'max_angle': angles.max(),
            'edge_lengths': edges
        }
    
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the Delaunay triangulation and carve out the holes.

        Returns:
        --------
        nodes : np.ndarray
            (N, 2) array of node coordinates
        elements : np.ndarray
            (E, 3) array of element connectivity
        """
        print("Generating Delaunay triangulation...")

        # Perform Delaunay triangulation
        tri = Delaunay(self.points)

        print(f"  Total triangles: {len(tri.simplices)}")

        self.triangles = tri.simplices.copy()
        self.num_elements = len(self.triangles)

        # Remove the void interior triangles + orphaned nodes and renumber.
        self.remove_hole_elements()

        print(f"  Final mesh: {self.num_elements} elements, {self.num_nodes} nodes")

        return self.points, self.triangles

    def remove_hole_elements(self, tolerance: float = 1e-9):
        """
        Remove void (hole) elements and orphaned nodes, then renumber.

        A triangle is considered inside a hole when its centroid lies strictly
        inside any shape (signed_distance < 0).  Using the centroid gives a clean,
        boundary-conforming faceted free surface: triangles straddling the
        boundary are kept if their centroid is on the matrix side and dropped
        otherwise.  After removal, nodes referenced by no remaining triangle are
        deleted and the connectivity is renumbered to a contiguous range.
        """
        print("\nRemoving void (hole) elements...")

        centroids = self.points[self.triangles].mean(axis=1)   # (E, 2)
        # An element is in a hole if its centroid is inside ANY shape.
        in_hole = np.zeros(self.num_elements, dtype=bool)
        for shape in self.shapes:
            sd = shape.signed_distance(centroids[:, 0], centroids[:, 1])
            in_hole |= (np.asarray(sd) < -tolerance)

        keep_elem = ~in_hole
        n_removed = int(np.count_nonzero(in_hole))
        kept_triangles = self.triangles[keep_elem]

        # Identify surviving nodes and build an old->new index remap.
        used_nodes = np.unique(kept_triangles)
        remap = -np.ones(self.num_nodes, dtype=np.int64)
        remap[used_nodes] = np.arange(len(used_nodes), dtype=np.int64)
        n_orphaned = self.num_nodes - len(used_nodes)

        # Apply the remap.
        self.points = self.points[used_nodes]
        self.triangles = remap[kept_triangles]
        self.num_nodes = len(self.points)
        self.num_elements = len(self.triangles)

        # Everything that remains is matrix; re-derive the (now trivial) sets.
        self.inclusion_nodes = set()
        self.matrix_nodes = set(range(self.num_nodes))
        self.inclusion_elements = set()
        self.interface_elements = set()
        self.matrix_elements = set(range(self.num_elements))

        print(f"  Removed {n_removed} interior triangles, {n_orphaned} orphaned nodes")
        print(f"  Remaining: {self.num_elements} elements, {self.num_nodes} nodes (all matrix)")
    
    def analyze_mesh_quality(self):
        """Analyze mesh quality by material"""
        print("\nAnalyzing mesh quality...")
        
        if self.triangles is None:
            print("  No mesh generated yet!")
            return
        
        # Analyze by material type
        for mat_type, elem_set in [
            ('Matrix', self.matrix_elements),
            ('Inclusion', self.inclusion_elements),
            ('Interface', self.interface_elements)
        ]:
            if len(elem_set) == 0:
                continue
            
            aspect_ratios = []
            min_angles = []
            max_angles = []
            areas = []
            
            for elem_id in elem_set:
                metrics = self.compute_triangle_quality(self.triangles[elem_id])
                aspect_ratios.append(metrics['aspect_ratio'])
                min_angles.append(metrics['min_angle'])
                max_angles.append(metrics['max_angle'])
                areas.append(metrics['area'])
            
            aspect_ratios = np.array(aspect_ratios)
            min_angles = np.array(min_angles)
            
            print(f"\n  {mat_type} elements ({len(elem_set)}):")
            print(f"    Aspect ratio: mean={aspect_ratios.mean():.2f}, max={aspect_ratios.max():.2f}")
            print(f"    Min angle: mean={min_angles.mean():.1f} deg, min={min_angles.min():.1f} deg")
            
            bad_aspect = np.sum(aspect_ratios > 10)
            bad_angle = np.sum(min_angles < 10)
            
            if bad_aspect > 0:
                print(f"    ! {bad_aspect} elements with aspect ratio > 10")
            if bad_angle > 0:
                print(f"    ! {bad_angle} elements with min angle < 10 deg")
    
    def visualize_mesh(self, output_path='mesh_composite.png', show_materials=True):
        """Visualize the hole-plate mesh"""
        print("\nGenerating mesh visualization...")
        
        if self.triangles is None:
            print("  No mesh to visualize!")
            return
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        if show_materials:
            # Everything that survives the void removal is matrix.
            for elem_id in self.matrix_elements:
                tri = self.points[self.triangles[elem_id]]
                poly = plt.Polygon(tri, facecolor='lightblue',
                                 edgecolor='black', linewidth=0.3, alpha=0.5)
                ax.add_patch(poly)
        else:
            # Simple mesh
            for tri_indices in self.triangles:
                triangle = self.points[tri_indices]
                triangle = np.vstack([triangle, triangle[0]])
                ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=0.3, alpha=0.5)

        # Plot nodes
        ax.plot(self.points[:, 0], self.points[:, 1], 'k.', markersize=0.5, alpha=0.3)

        # Draw hole boundaries
        for shape in self.shapes:
            shape.draw(ax, edgecolor='red', linewidth=2)

        # Legend
        if show_materials:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', edgecolor='black', label=f'Matrix ({len(self.matrix_elements)})'),
                plt.Line2D([0], [0], color='red', linewidth=2, label='Hole boundary')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Hole-Plate Mesh\n{self.num_elements} elements, {self.num_nodes} nodes',
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, self.plate_size + 0.5)
        ax.set_ylim(-0.5, self.plate_size + 0.5)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [ok] Saved: {output_path}")
        plt.close()
    
    def visualize_mesh_detail(self,
                             center: Tuple[float, float],
                             size: float,
                             output_path='mesh_detail_composite.png'):
        """Visualize a detailed region around a hole (the free surface)"""
        print(f"\nGenerating detailed mesh view at ({center[0]:.1f}, {center[1]:.1f})...")

        if self.triangles is None:
            print("  No mesh to visualize!")
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot matrix triangles in view (everything is matrix after void removal)
        for elem_id in self.matrix_elements:
            tri = self.points[self.triangles[elem_id]]
            if (np.all(tri[:, 0] >= center[0] - size/2) and
                np.all(tri[:, 0] <= center[0] + size/2) and
                np.all(tri[:, 1] >= center[1] - size/2) and
                np.all(tri[:, 1] <= center[1] + size/2)):
                poly = plt.Polygon(tri, facecolor='lightblue',
                                 edgecolor='black', linewidth=0.5, alpha=0.5)
                ax.add_patch(poly)

        # Plot nodes in view
        mask = ((self.points[:, 0] >= center[0] - size/2) &
                (self.points[:, 0] <= center[0] + size/2) &
                (self.points[:, 1] >= center[1] - size/2) &
                (self.points[:, 1] <= center[1] + size/2))
        view_points = self.points[mask]
        ax.plot(view_points[:, 0], view_points[:, 1], 'ko', markersize=3)

        # Draw hole boundaries (matplotlib clips to the view limits below)
        for shape in self.shapes:
            shape.draw(ax, edgecolor='red', linewidth=2)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', label='Matrix'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Hole boundary')
        ]
        ax.legend(handles=legend_elements)

        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'Hole-Plate Mesh Detail View', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(center[0] - size/2, center[0] + size/2)
        ax.set_ylim(center[1] - size/2, center[1] + size/2)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [ok] Saved: {output_path}")
        plt.close()


if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python mesh_generator.py <points.npy> <config.json> [plate_size]")
        print("Example: python mesh_generator.py collocation_points_composite.npy config.json 20")
        sys.exit(1)

    # Load inputs
    points = np.load(sys.argv[1])

    with open(sys.argv[2], 'r') as f:
        config = json.load(f)

    # Handle single or multiple holes
    if isinstance(config, dict):
        inclusions = [config]
    else:
        inclusions = config

    plate_size = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

    print(f"Loaded {len(points)} collocation points")
    print(f"Loaded {len(inclusions)} holes")

    # Generate mesh
    generator = CompositeMeshGenerator(points, inclusions, plate_size)
    nodes, elements = generator.generate_mesh()
    
    # Analyze quality
    generator.analyze_mesh_quality()
    
    # Visualizations
    generator.visualize_mesh('mesh_composite_full.png', show_materials=True)
    generator.visualize_mesh('mesh_composite_simple.png', show_materials=False)
    
    # Detail view around first hole
    if inclusions:
        first_shape = generator.shapes[0]
        cx, cy = first_shape.center()
        generator.visualize_mesh_detail(
            center=(cx, cy),
            size=first_shape.char_length() * 8,
            output_path='mesh_composite_detail.png'
        )

    print("\n[ok] Hole-plate mesh generation complete!")
