"""
Composite Rod Mesh Generator (2-D cross-section -> extruded 3-D tetrahedra)
==========================================================================
Two stages:

1. ``generate_mesh()``  — Delaunay triangulation of the cross-section
   collocation points, with every triangle KEPT and classified
   matrix / inclusion / interface. The cross-section is a disk, which is convex,
   so the Delaunay convex hull IS the domain and no trimming is required.

2. ``extrude_to_3d(height, n_layers)`` — sweeps that cross-section along +z into
   ``n_layers`` layers of triangular prisms and splits every prism into three
   tetrahedra. The inclusion therefore runs straight through the full depth of
   the rod, and z = 0 / z = H are flat node layers, which makes the clamped and
   loaded end faces exact node sets rather than geometric queries.

Prism splitting and conformity
------------------------------
Three tetrahedra per prism is the minimum, but a naive split leaves the
quadrilateral side faces with mismatched diagonals between neighbouring prisms —
a non-conforming mesh that FEniCSx would reject and whose energy integral would
be wrong. The fix is to make the diagonal a function of the GLOBAL vertex
indices only, so two prisms sharing a side face independently choose the same
one. Sorting each triangle's three cross-section node ids ascending as
(v0 < v1 < v2) and emitting

    T1 = (b0, b1, b2, t0)
    T2 = (b1, b2, t0, t1)
    T3 = (b2, t0, t1, t2)

does exactly that: on the side face over bottom edge (b_i, b_j) with i < j the
diagonal is always b_j--t_i, which depends only on the ordering of the two
shared global ids. This is the standard Dompierre et al. subdivision.

"""

import numpy as np
from scipy.spatial import Delaunay
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use("Agg")   # headless backend (see inclusion_shapes.py for why).
import matplotlib.pyplot as plt

from inclusion_shapes import make_inclusion


class CompositeMeshGenerator:
    """Delaunay cross-section + material classification + extrusion to tets."""

    def __init__(self, points: np.ndarray, inclusions: List[Dict],
                 plate_radius: float, plate_center: Tuple[float, float]):
        """
        Parameters
        ----------
        points : (N, 2)
            Cross-section collocation points.
        inclusions : List[Dict]
            Inclusion configurations (filled regions, circle or petal).
        plate_radius : float
            Outer radius of the cross-section.
        plate_center : (float, float)
            Cross-section centre = the torsion axis.
        """
        self.points = points
        self.inclusions = inclusions
        # Geometry layer: one shape object per inclusion (dispatched on JSON "type")
        self.shapes = [make_inclusion(inc) for inc in inclusions]
        self.plate_radius = float(plate_radius)
        self.axis_x, self.axis_y = float(plate_center[0]), float(plate_center[1])

        self.triangles = None
        self.num_nodes = len(points)
        self.num_elements = 0

        # Material classification (2-D cross-section)
        self.matrix_elements = set()
        self.inclusion_elements = set()
        self.interface_elements = set()

        self.matrix_nodes = set()
        self.inclusion_nodes = set()

        # 3-D extrusion products (filled by extrude_to_3d)
        self.nodes_3d = None
        self.tets = None
        self.tet_material_ids = None
        self.height = None
        self.n_layers = None

        print("\n" + "="*60)
        print("COMPOSITE ROD MESH GENERATOR")
        print("="*60)
        print(f"Cross-section points: {self.num_nodes}")
        print(f"Cross-section: disk R={self.plate_radius:.4f} at "
              f"({self.axis_x:.4f}, {self.axis_y:.4f})")
        print(f"Inclusions: {len(inclusions)}")
        print()

    # ── cross-section classification ─────────────────────────────────────────

    def is_point_in_inclusion(self, point: np.ndarray, tolerance: float = 1e-6) -> Tuple[bool, int]:
        """Is the point inside any inclusion? Returns (bool, inclusion index)."""
        for i, shape in enumerate(self.shapes):
            if shape.contains(point[0], point[1], tolerance):
                return True, i
        return False, -1

    def classify_nodes(self, tolerance: float = 1e-6):
        """Classify all cross-section nodes as matrix or inclusion"""
        print("Classifying nodes by material...")

        for node_id in range(self.num_nodes):
            inside, inc_id = self.is_point_in_inclusion(self.points[node_id], tolerance)
            if inside:
                self.inclusion_nodes.add(node_id)
            else:
                self.matrix_nodes.add(node_id)

        print(f"  Matrix nodes: {len(self.matrix_nodes)}")
        print(f"  Inclusion nodes: {len(self.inclusion_nodes)}")

    def classify_elements(self):
        """Classify triangles as matrix, inclusion, or interface"""
        print("Classifying elements by material...")

        for elem_id, elem in enumerate(self.triangles):
            nodes_in_inclusion = sum(1 for n in elem if n in self.inclusion_nodes)

            if nodes_in_inclusion == 3:
                self.inclusion_elements.add(elem_id)
            elif nodes_in_inclusion == 0:
                self.matrix_elements.add(elem_id)
            else:
                self.interface_elements.add(elem_id)

        print(f"  Matrix elements: {len(self.matrix_elements)}")
        print(f"  Inclusion elements: {len(self.inclusion_elements)}")
        print(f"  Interface elements: {len(self.interface_elements)}")

    def compute_triangle_quality(self, tri_indices: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for a cross-section triangle"""
        vertices = self.points[tri_indices]

        edges = np.array([
            np.linalg.norm(vertices[1] - vertices[0]),
            np.linalg.norm(vertices[2] - vertices[1]),
            np.linalg.norm(vertices[0] - vertices[2])
        ])

        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        area = 0.5 * abs(np.cross(v1, v2))

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
        Delaunay-triangulate the cross-section.

        The disk is convex and the rim is sampled by the collocation generator,
        so the convex hull of the point set IS the cross-section: every triangle
        lies inside the domain and all of them are kept (a 2-phase composite has
        no holes). The only cleanup is dropping degenerate slivers whose area is
        numerically zero, which would otherwise give an infinite deformation
        gradient after extrusion.

        Returns (nodes (N,2), triangles (T,3)).
        """
        print("Generating Delaunay triangulation of the cross-section...")

        tri = Delaunay(self.points)
        simplices = tri.simplices.copy()
        print(f"  Total triangles: {len(simplices)}")

        # Drop numerically degenerate triangles (zero area). On a convex domain
        # these are the only invalid cells Delaunay can produce.
        v0 = self.points[simplices[:, 0]]
        v1 = self.points[simplices[:, 1]]
        v2 = self.points[simplices[:, 2]]
        cross = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                 - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
        areas = 0.5 * np.abs(cross)
        keep = areas > 1e-14
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"  Dropped {n_drop} degenerate (zero-area) triangles")
            simplices = simplices[keep]
            cross = cross[keep]

        # Orient every triangle counter-clockwise so the extruded prisms have a
        # consistent bottom->top sense and the tets come out positive-volume.
        flip = cross < 0.0
        if flip.any():
            simplices[flip] = simplices[flip][:, [0, 2, 1]]
            print(f"  Re-oriented {int(flip.sum())} clockwise triangles to CCW")

        self.triangles = simplices
        self.num_elements = len(self.triangles)

        self.classify_nodes()
        self.classify_elements()

        print(f"  Cross-section mesh: {self.num_elements} triangles, "
              f"{self.num_nodes} nodes")

        return self.points, self.triangles

    # ── extrusion to 3-D tetrahedra ──────────────────────────────────────────

    def extrude_to_3d(self, height: float, n_layers: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sweep the classified cross-section along +z into tetrahedra.

        Node numbering is layer-major:  ``node_3d = layer * N2 + node_2d``,
        with layer 0 at z = 0 (clamped face) and layer ``n_layers`` at z = H
        (loaded face). The mesh processor relies on this ordering to identify the
        end faces and to slice fields at a given height, so do not reorder.

        Each prism is split into three tets by the global-index rule documented
        in the module header, which keeps the side-face diagonals conforming
        between neighbouring prisms.

        Returns (nodes_3d (N3,3), tets (E,4), tet_material_ids (E,)).
        """
        if self.triangles is None:
            raise ValueError("Call generate_mesh() before extrude_to_3d().")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1 (got {n_layers})")

        print("\n" + "="*60)
        print("EXTRUDING CROSS-SECTION TO 3-D TETRAHEDRA")
        print("="*60)

        N2 = self.num_nodes
        T2 = self.num_elements
        L = int(n_layers)
        H = float(height)
        dz = H / L

        print(f"  Height H       : {H:.4f}")
        print(f"  Layers         : {L}  (dz = {dz:.4f})")
        print(f"  Aspect dz/h_xy : {dz / self._mean_edge_length():.3f} "
              f"(1.0 is ideal; keep within ~0.5-2)")

        # ── nodes: L+1 copies of the cross-section, stacked in z ──
        z_levels = np.linspace(0.0, H, L + 1)
        xy = np.repeat(self.points[None, :, :], L + 1, axis=0)          # (L+1, N2, 2)
        zz = np.repeat(z_levels[:, None], N2, axis=1)[:, :, None]        # (L+1, N2, 1)
        nodes_3d = np.concatenate([xy, zz], axis=2).reshape(-1, 3)       # (N3, 3)

        # ── element material ids of the cross-section (0/1/2) ──
        tri_mat = np.zeros(T2, dtype=np.int64)
        for e in range(T2):
            if e in self.inclusion_elements:
                tri_mat[e] = 1
            elif e in self.interface_elements:
                tri_mat[e] = 2

        # ── prism -> 3 tets, with globally-consistent diagonals ──
        # Sort each triangle's node ids ascending: the split rule must depend on
        # the GLOBAL ids alone so neighbouring prisms agree on the shared
        # side-face diagonal (see module docstring).
        tri_sorted = np.sort(self.triangles, axis=1)                     # (T2, 3)

        tets = np.empty((L * T2 * 3, 4), dtype=np.int64)
        tet_mat = np.empty(L * T2 * 3, dtype=np.int64)

        for k in range(L):
            b = tri_sorted + k * N2               # bottom-layer ids  (T2, 3)
            t = tri_sorted + (k + 1) * N2         # top-layer ids     (T2, 3)
            b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
            t0, t1, t2 = t[:, 0], t[:, 1], t[:, 2]

            block = np.empty((T2 * 3, 4), dtype=np.int64)
            block[0::3] = np.column_stack([b0, b1, b2, t0])
            block[1::3] = np.column_stack([b1, b2, t0, t1])
            block[2::3] = np.column_stack([b2, t0, t1, t2])

            tets[k * T2 * 3:(k + 1) * T2 * 3] = block
            tet_mat[k * T2 * 3:(k + 1) * T2 * 3] = np.repeat(tri_mat, 3)

        # ── enforce positive volume ──
        # The subdivision above is orientation-consistent for CCW triangles, but
        # a negative-volume tet gives det(F) < 0 and a NaN in log(J), so check
        # rather than assume.
        vols = self._tet_volumes(nodes_3d, tets)
        neg = vols < 0.0
        if neg.any():
            tets[neg] = tets[neg][:, [0, 1, 3, 2]]
            vols = self._tet_volumes(nodes_3d, tets)
            print(f"  Flipped {int(neg.sum())} negative-volume tets")
        if (vols <= 0.0).any():
            raise ValueError(
                f"{int((vols <= 0).sum())} tetrahedra still have non-positive "
                f"volume after re-orientation — the cross-section mesh is invalid.")

        self.nodes_3d = nodes_3d
        self.tets = tets
        self.tet_material_ids = tet_mat
        self.height = H
        self.n_layers = L

        print(f"  Nodes    : {len(nodes_3d):,}  ({N2:,} per layer x {L + 1} layers)")
        print(f"  Tetrahedra: {len(tets):,}  ({T2:,} prisms x 3 x {L} layers)")
        print(f"  Volume   : {vols.sum():.6f}  "
              f"(analytic disk pi*R^2*H = {np.pi * self.plate_radius**2 * H:.6f})")
        print(f"  Material : matrix {int((tet_mat == 0).sum()):,} / "
              f"inclusion {int((tet_mat == 1).sum()):,} / "
              f"interface {int((tet_mat == 2).sum()):,}")

        return nodes_3d, tets, tet_mat

    @staticmethod
    def _tet_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
        """Signed volumes  det[X1-X0, X2-X0, X3-X0] / 6."""
        p = nodes[tets]                                   # (E, 4, 3)
        d = p[:, 1:, :] - p[:, :1, :]                     # (E, 3, 3)
        return np.linalg.det(d) / 6.0

    def _mean_edge_length(self) -> float:
        """Mean cross-section edge length (for the extrusion aspect report)."""
        tri = self.triangles
        p = self.points
        e = np.concatenate([
            np.linalg.norm(p[tri[:, 1]] - p[tri[:, 0]], axis=1),
            np.linalg.norm(p[tri[:, 2]] - p[tri[:, 1]], axis=1),
            np.linalg.norm(p[tri[:, 0]] - p[tri[:, 2]], axis=1),
        ])
        return float(e.mean())

    # ── quality reporting ────────────────────────────────────────────────────

    def analyze_mesh_quality(self):
        """Analyze cross-section mesh quality by material"""
        print("\nAnalyzing cross-section mesh quality...")

        if self.triangles is None:
            print("  No mesh generated yet!")
            return

        for mat_type, elem_set in [
            ('Matrix', self.matrix_elements),
            ('Inclusion', self.inclusion_elements),
            ('Interface', self.interface_elements)
        ]:
            if len(elem_set) == 0:
                continue

            aspect_ratios = []
            min_angles = []

            for elem_id in elem_set:
                metrics = self.compute_triangle_quality(self.triangles[elem_id])
                aspect_ratios.append(metrics['aspect_ratio'])
                min_angles.append(metrics['min_angle'])

            aspect_ratios = np.array(aspect_ratios)
            min_angles = np.array(min_angles)

            print(f"\n  {mat_type} elements ({len(elem_set)}):")
            print(f"    Aspect ratio: mean={aspect_ratios.mean():.2f}, max={aspect_ratios.max():.2f}")
            print(f"    Min angle: mean={min_angles.mean():.1f} deg, min={min_angles.min():.1f} deg")

            bad_aspect = np.sum(aspect_ratios > 10)
            bad_angle = np.sum(min_angles < 10)

            if bad_aspect > 0:
                print(f"    WARNING: {bad_aspect} elements with aspect ratio > 10")
            if bad_angle > 0:
                print(f"    WARNING: {bad_angle} elements with min angle < 10 deg")

    def analyze_tet_quality(self):
        """Report tetrahedron volume and shape quality after extrusion."""
        if self.tets is None:
            print("  No 3-D mesh generated yet!")
            return
        print("\nAnalyzing tetrahedron quality...")
        vols = self._tet_volumes(self.nodes_3d, self.tets)

        # Radius-ratio-style quality: normalised volume / (RMS edge length)^3.
        # 1.0 is a regular tet; below ~0.05 the element is a sliver.
        p = self.nodes_3d[self.tets]
        idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        e2 = np.stack([np.sum((p[:, i] - p[:, j]) ** 2, axis=1) for i, j in idx], axis=1)
        rms = np.sqrt(e2.mean(axis=1))
        quality = (vols * 6.0 * np.sqrt(2.0)) / np.maximum(rms ** 3, 1e-30)

        print(f"  Volume   : min={vols.min():.4e}, max={vols.max():.4e}, "
              f"mean={vols.mean():.4e}")
        print(f"  Quality  : min={quality.min():.4f}, mean={quality.mean():.4f} "
              f"(1.0 = regular tet)")
        n_bad = int((quality < 0.05).sum())
        if n_bad:
            print(f"    WARNING: {n_bad} sliver tets (quality < 0.05). Bring dz "
                  f"closer to the in-plane element size (--n-layers).")

    # ── visualisation ────────────────────────────────────────────────────────

    def visualize_mesh(self, output_path='mesh_composite.png', show_materials=True):
        """Visualize the cross-section mesh (the 3-D mesh is inspected in ParaView)."""
        print("\nGenerating cross-section mesh visualization...")

        if self.triangles is None:
            print("  No mesh to visualize!")
            return

        fig, ax = plt.subplots(figsize=(11, 11))

        if show_materials:
            for elem_set, color, label in [
                (self.matrix_elements,    '#4C72B0', 'Matrix'),
                (self.inclusion_elements, '#C44E52', 'Inclusion'),
                (self.interface_elements, '#DD8452', 'Interface'),
            ]:
                if not elem_set:
                    continue
                tri = self.triangles[sorted(elem_set)]
                ax.triplot(self.points[:, 0], self.points[:, 1], tri,
                           color=color, linewidth=0.4, alpha=0.8)
                ax.plot([], [], color=color, linewidth=2, label=f'{label} ({len(elem_set)})')
            ax.legend(loc='upper right', fontsize=9)
        else:
            ax.triplot(self.points[:, 0], self.points[:, 1], self.triangles,
                       color='k', linewidth=0.3, alpha=0.7)

        # Rim + inclusion outlines
        ax.add_patch(plt.Circle((self.axis_x, self.axis_y), self.plate_radius, fill=False,
                                edgecolor='black', linewidth=2))
        for shape in self.shapes:
            shape.draw(ax, edgecolor='red', linewidth=2)

        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_title(f'Rod Cross-Section Mesh — {self.num_elements} triangles, '
                     f'{self.num_nodes} nodes', fontweight='bold')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)

    def visualize_mesh_detail(self, center, size, output_path='mesh_detail.png'):
        """Zoomed cross-section view around a point."""
        if self.triangles is None:
            return
        fig, ax = plt.subplots(figsize=(10, 10))
        for elem_set, color in [
            (self.matrix_elements,    '#4C72B0'),
            (self.inclusion_elements, '#C44E52'),
            (self.interface_elements, '#DD8452'),
        ]:
            if not elem_set:
                continue
            ax.triplot(self.points[:, 0], self.points[:, 1],
                       self.triangles[sorted(elem_set)],
                       color=color, linewidth=0.6, alpha=0.9)
        for shape in self.shapes:
            shape.draw(ax, edgecolor='red', linewidth=2)

        view_center_x, view_center_y = center
        ax.set_xlim(view_center_x - size / 2, view_center_x + size / 2)
        ax.set_ylim(view_center_y - size / 2, view_center_y + size / 2)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_title('Cross-section detail (interface refinement)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)
