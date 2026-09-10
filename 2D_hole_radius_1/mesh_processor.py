"""
Hole-Plate Mesh Processor for PI-GNN Training
=============================================
Turns the 2-D plate-with-holes triangulation into the .npy arrays consumed by
the PI-GNN trainer and the FEM comparison, plus the ``composite_mesh.vtu`` the
FEM reads.

The processor takes the ``(nodes, elements)`` arrays DIRECTLY from
``mesh_generator.CompositeMeshGenerator`` — there is no intermediate DOLFIN-XML
file. A round-trip through XML bought nothing and gave the mesh two independent
on-disk representations that could drift apart; now the trainer and the FEM read
the same arrays and the same VTU.

The incoming mesh already has the void interiors removed (see
``mesh_generator.remove_hole_elements``), so every surviving node and element is
matrix — a single homogeneous phase. This processor identifies the outer plate
boundaries, the loading (right) edge, and the hole-rim free-surface facets (each
void boundary is an exterior traction-free surface), and records the plate
geometry in ``mesh_summary.json`` so downstream stages do not have to re-derive
it from a bounding box.
"""

import numpy as np
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")   # headless backend (see inclusion_shapes.py for why).
import matplotlib.pyplot as plt
from pathlib import Path

from inclusion_shapes import make_inclusion, describe


class CompositeMeshProcessor:
    """Process a 2D hole-plate triangular mesh for FEM/GNN applications"""

    def __init__(self, nodes, elements, config_filepath=None,
                 plate_size=None, plate_center=None):
        """
        Parameters
        ----------
        nodes : (N, 2) array
            2-D node coordinates from ``CompositeMeshGenerator.generate_mesh``
            (voids already removed and the connectivity renumbered).
        elements : (E, 3) array
            Triangular connectivity.
        config_filepath : str, optional
            JSON config with the hole geometry (used for the summary and for the
            near-field metrics downstream; the free-surface detection itself is
            purely topological).
        plate_size : float, optional
            Side length of the square plate. Defaults to the larger bounding-box
            extent. Recorded in mesh_summary.json because the training stage
            sizes ``OUTPUT_DISPLACEMENT_SCALE`` from it.
        plate_center : (float, float), optional
            Plate centre; defaults to the bounding-box centre. Recorded for
            reporting only.
        """
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.num_nodes = len(self.nodes)
        self.num_elements = len(self.elements)

        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError(f"Expected (N,2) nodes, got {self.nodes.shape}")
        if self.elements.ndim != 2 or self.elements.shape[1] != 3:
            raise ValueError(f"Expected (E,3) triangles, got {self.elements.shape}")

        self.config_filepath = config_filepath

        x0, x1 = float(self.nodes[:, 0].min()), float(self.nodes[:, 0].max())
        y0, y1 = float(self.nodes[:, 1].min()), float(self.nodes[:, 1].max())
        self.plate_size = (float(plate_size) if plate_size is not None
                           else max(x1 - x0, y1 - y0))
        if plate_center is None:
            plate_center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        self.center_x, self.center_y = float(plate_center[0]), float(plate_center[1])

        # Hole configuration: raw dicts + derived geometry shapes
        self.inclusions = []
        self.shapes = []

        # Everything that survives the void removal is matrix.
        self.matrix_nodes = set()
        self.matrix_elements = set()

        # Boundary identification
        self.left_edge_nodes = set()
        self.right_edge_nodes = set()
        self.top_edge_nodes = set()
        self.bottom_edge_nodes = set()
        self.loading_facets = []
        self.hole_boundary_facets = []   # void-rim free-surface facets

        print(f"[ok] Hole-plate mesh:")
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Triangles: {self.num_elements}")
        print(f"  - X range: [{x0:.4f}, {x1:.4f}]")
        print(f"  - Y range: [{y0:.4f}, {y1:.4f}]")
        print(f"  - Plate size: {self.plate_size:.4f}")

        if config_filepath:
            self._load_config()

    def _load_config(self):
        """Load hole configuration from JSON file (single dict or list of dicts)"""
        print(f"\nLoading configuration from: {self.config_filepath}")

        with open(self.config_filepath, 'r') as f:
            config = json.load(f)

        if isinstance(config, dict):
            self.inclusions = [config]
        elif isinstance(config, list):
            self.inclusions = config
        else:
            raise ValueError("Config JSON must be a dict or list of dicts.")

        # Geometry layer: one shape object per hole (circular voids).
        self.shapes = [make_inclusion(inc) for inc in self.inclusions]

        print(f"[ok] Hole configuration: {len(self.inclusions)} hole(s)")
        for i, inc in enumerate(self.inclusions):
            print(f"  - Hole {i+1}: {describe(inc)}")

    def classify_materials(self):
        """Mark every surviving node/element as matrix.

        The void interiors were removed at mesh time, so the plate is a single
        homogeneous phase; the hole boundaries are exterior free surfaces, not
        material interfaces.
        """
        print("\nClassifying materials (plate with holes — all matrix)...")

        self.matrix_nodes = set(range(self.num_nodes))
        self.matrix_elements = set(range(self.num_elements))

        print(f"[ok] Material classification:")
        print(f"  - Matrix nodes: {len(self.matrix_nodes)}")
        print(f"  - Matrix elements: {len(self.matrix_elements)}")

    def identify_boundaries(self, tolerance=1e-6):
        """
        Identify boundary edges (left, right, top, bottom)

        Parameters:
        -----------
        tolerance : float
            Tolerance for floating point comparison
        """
        print("\nIdentifying boundaries...")

        x_coords = self.nodes[:, 0]
        y_coords = self.nodes[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Edge identification
        self.left_edge_nodes = set(np.where(np.abs(x_coords - x_min) < tolerance)[0])
        self.right_edge_nodes = set(np.where(np.abs(x_coords - x_max) < tolerance)[0])
        self.bottom_edge_nodes = set(np.where(np.abs(y_coords - y_min) < tolerance)[0])
        self.top_edge_nodes = set(np.where(np.abs(y_coords - y_max) < tolerance)[0])

        print(f"  - Left edge (x={x_min:.4f}): {len(self.left_edge_nodes)} nodes")
        print(f"  - Right edge (x={x_max:.4f}): {len(self.right_edge_nodes)} nodes")
        print(f"  - Bottom edge (y={y_min:.4f}): {len(self.bottom_edge_nodes)} nodes")
        print(f"  - Top edge (y={y_max:.4f}): {len(self.top_edge_nodes)} nodes")

    def find_loading_facets(self, tolerance=1e-6):
        """Find element edges on the right (loading) boundary"""
        print("\nFinding loading surface facets...")

        x_max = self.nodes[:, 0].max()
        self.loading_facets = []

        for elem_id, elem in enumerate(self.elements):
            edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]

            for n1, n2 in edges:
                x1, x2 = self.nodes[n1, 0], self.nodes[n2, 0]

                if (np.abs(x1 - x_max) < tolerance and
                    np.abs(x2 - x_max) < tolerance):
                    self.loading_facets.append([elem_id, n1, n2])

        self.loading_facets = np.array(self.loading_facets, dtype=np.int64)
        print(f"  - Loading facets found: {len(self.loading_facets)}")

    def _facet_lengths(self, facets):
        """Edge lengths of ``[elem_id, n1, n2]`` facet rows (the 2-D analogue of
        a facet area). Used to report the loaded-edge length, which is what the
        applied traction is integrated over."""
        facets = np.asarray(facets, dtype=np.int64).reshape(-1, 3)
        if len(facets) == 0:
            return np.zeros(0, dtype=np.float64)
        p1 = self.nodes[facets[:, 1]]
        p2 = self.nodes[facets[:, 2]]
        return np.linalg.norm(p2 - p1, axis=1).astype(np.float64)

    def find_hole_boundary_facets(self, tolerance=1e-6):
        """Find the free-surface facets created by removing the voids.

        A free surface is any edge that belongs to exactly one triangle and is
        NOT on the four outer plate edges (those carry the roller/traction/free
        plate boundaries). This isolates the void rims. Stored as
        [elem_id, n1, n2] rows, consistent with the loading-facet array.
        """
        print("\nFinding hole-boundary (free-surface) facets...")

        x_coords = self.nodes[:, 0]
        y_coords = self.nodes[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        def _on_outer_edge(n1, n2):
            for lo, hi, coord in ((x_min, x_max, 0), (y_min, y_max, 1)):
                a, b = self.nodes[n1, coord], self.nodes[n2, coord]
                if (abs(a - lo) < tolerance and abs(b - lo) < tolerance) or \
                   (abs(a - hi) < tolerance and abs(b - hi) < tolerance):
                    return True
            return False

        # Count how many triangles share each (sorted) edge, remembering one owner.
        edge_count = defaultdict(int)
        edge_owner = {}
        for elem_id, elem in enumerate(self.elements):
            for n1, n2 in ((elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])):
                key = (min(int(n1), int(n2)), max(int(n1), int(n2)))
                edge_count[key] += 1
                edge_owner.setdefault(key, (elem_id, int(n1), int(n2)))

        self.hole_boundary_facets = []
        for key, count in edge_count.items():
            if count == 1:  # boundary edge
                n1, n2 = key
                if not _on_outer_edge(n1, n2):
                    self.hole_boundary_facets.append(list(edge_owner[key]))

        self.hole_boundary_facets = np.array(self.hole_boundary_facets, dtype=np.int64)
        print(f"  - Hole-boundary facets found: {len(self.hole_boundary_facets)}")

    def compute_node_topology(self):
        """Compute node adjacency/neighborhood graph"""
        print("\nComputing node topology...")

        adjacency = defaultdict(set)

        for elem in self.elements:
            adjacency[elem[0]].update([elem[1], elem[2]])
            adjacency[elem[1]].update([elem[0], elem[2]])
            adjacency[elem[2]].update([elem[0], elem[1]])

        max_neighbors = max(len(neighbors) for neighbors in adjacency.values())
        print(f"  - Max neighbors: {max_neighbors}")

        topology = np.full((self.num_nodes, max_neighbors), -1, dtype=np.int64)

        for node_id in range(self.num_nodes):
            neighbors = sorted(adjacency[node_id])
            topology[node_id, :len(neighbors)] = neighbors

        return topology

    def generate_node_features(self):
        """
        Generate node feature matrix

        Returns:
        --------
        features : np.ndarray
            (NumNodes, 6) array with:
            [Dirichlet, Neumann, Material_ID, Interface, X, Y]
            For a plate with holes the plate is a single matrix phase, so
            Material_ID and Interface are uniformly 0 (the columns are retained
            so the GNN input layout is identical to the composite pipeline).
        """
        print("\nGenerating node features...")

        features = np.zeros((self.num_nodes, 6), dtype=np.float64)

        for i in range(self.num_nodes):
            # Column 0: Dirichlet flag (left edge = fixed)
            features[i, 0] = 1.0 if i in self.left_edge_nodes else 0.0

            # Column 1: Neumann flag (right edge = loaded)
            features[i, 1] = 1.0 if i in self.right_edge_nodes else 0.0

            # Column 2: Material ID — always 0 (single matrix phase)
            # Column 3: Interface flag — always 0 (no material interface)

            # Columns 4-5: Coordinates
            features[i, 4:6] = self.nodes[i]

        print(f"  - Feature shape: {features.shape}")
        return features

    def generate_element_material_ids(self):
        """
        Generate element material IDs.

        Returns:
        --------
        material_ids : np.ndarray
            (NumElements,) array — all 0 (single matrix phase).
        """
        return np.zeros(self.num_elements, dtype=np.int64)

    def generate_interior_mask(self):
        """Generate mask for interior (non-boundary) nodes"""
        boundary_nodes = (self.left_edge_nodes | self.right_edge_nodes |
                         self.top_edge_nodes | self.bottom_edge_nodes)

        mask = np.ones(self.num_nodes, dtype=np.float64)
        for node_id in boundary_nodes:
            mask[node_id] = 0.0

        return mask

    def visualize_composite(self, output_path=None):
        """Visualize the hole-plate mesh (matrix + hole rims + boundary conditions)"""
        print("\nGenerating visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

        # Left plot: mesh + hole rims
        ax1.set_title('Mesh (matrix, voids removed)', fontsize=14, fontweight='bold')

        for elem_id in self.matrix_elements:
            elem = self.elements[elem_id]
            triangle = self.nodes[elem]
            triangle = np.vstack([triangle, triangle[0]])
            ax1.fill(triangle[:, 0], triangle[:, 1], color='lightblue',
                    edgecolor='blue', linewidth=0.3, alpha=0.5)

        # Plot the hole-rim free-surface facets
        if len(self.hole_boundary_facets) > 0:
            for facet in self.hole_boundary_facets:
                elem_id, n1, n2 = facet
                edge = self.nodes[[n1, n2]]
                ax1.plot(edge[:, 0], edge[:, 1], 'r-', linewidth=2, alpha=0.8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='blue', label=f'Matrix ({len(self.matrix_elements)} elements)'),
            plt.Line2D([0], [0], color='r', linewidth=2, label=f'Hole rims ({len(self.hole_boundary_facets)} facets)')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Right plot: Boundary conditions
        ax2.set_title('Boundary Conditions', fontsize=14, fontweight='bold')

        # Plot all elements in light gray
        for elem in self.elements:
            triangle = self.nodes[elem]
            triangle = np.vstack([triangle, triangle[0]])
            ax2.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=0.3, alpha=0.3)

        # Plot boundary nodes
        if self.left_edge_nodes:
            left_nodes = self.nodes[list(self.left_edge_nodes)]
            ax2.plot(left_nodes[:, 0], left_nodes[:, 1], 'bs',
                    markersize=4, label=f'Fixed (left) - {len(self.left_edge_nodes)} nodes')

        if self.right_edge_nodes:
            right_nodes = self.nodes[list(self.right_edge_nodes)]
            ax2.plot(right_nodes[:, 0], right_nodes[:, 1], 'ro',
                    markersize=4, label=f'Loaded (right) - {len(self.right_edge_nodes)} nodes')

        # Plot loading facets
        if len(self.loading_facets) > 0:
            for facet in self.loading_facets:
                elem_id, n1, n2 = facet
                edge = self.nodes[[n1, n2]]
                ax2.plot(edge[:, 0], edge[:, 1], 'r-', linewidth=3, alpha=0.8)

        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  [ok] Visualization saved: {output_path}")
        else:
            plt.savefig('composite_mesh_visualization.png', dpi=300, bbox_inches='tight')
            print(f"  [ok] Visualization saved: composite_mesh_visualization.png")

        plt.close()

    def export_to_vtu(self, output_path):
        """Export mesh to VTU format for ParaView visualization"""
        print(f"\nExporting to VTU format: {output_path}")

        with open(output_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')
            f.write(f'    <Piece NumberOfPoints="{self.num_nodes}" NumberOfCells="{self.num_elements}">\n')

            # Points
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
            for node in self.nodes:
                f.write(f'          {node[0]} {node[1]} 0.0\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')

            # Cells
            f.write('      <Cells>\n')
            f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
            for elem in self.elements:
                f.write(f'          {elem[0]} {elem[1]} {elem[2]}\n')
            f.write('        </DataArray>\n')

            f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
            for i in range(1, self.num_elements + 1):
                f.write(f'          {i*3}\n')
            f.write('        </DataArray>\n')

            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            for _ in range(self.num_elements):
                f.write('          5\n')
            f.write('        </DataArray>\n')
            f.write('      </Cells>\n')

            # Point data. VTK picks the ACTIVE point scalar from the Scalars=
            # attribute; without it the reader registers the arrays but leaves
            # active_scalars = None and ParaView opens the mesh uncoloured.
            f.write('      <PointData Scalars="Material_ID">\n')

            # Material ID — uniformly 0 (single matrix phase)
            f.write('        <DataArray type="Float64" Name="Material_ID" format="ascii">\n')
            for i in range(self.num_nodes):
                f.write('          0.0\n')
            f.write('        </DataArray>\n')

            # Boundary conditions
            f.write('        <DataArray type="Float64" Name="Dirichlet_BC" format="ascii">\n')
            for i in range(self.num_nodes):
                val = 1.0 if i in self.left_edge_nodes else 0.0
                f.write(f'          {val}\n')
            f.write('        </DataArray>\n')

            f.write('        <DataArray type="Float64" Name="Neumann_BC" format="ascii">\n')
            for i in range(self.num_nodes):
                val = 1.0 if i in self.right_edge_nodes else 0.0
                f.write(f'          {val}\n')
            f.write('        </DataArray>\n')

            f.write('      </PointData>\n')

            # Cell data
            f.write('      <CellData Scalars="Element_Material">\n')

            # Element material ID — all 0 (matrix)
            material_ids = self.generate_element_material_ids()
            f.write('        <DataArray type="Int64" Name="Element_Material" format="ascii">\n')
            for mat_id in material_ids:
                f.write(f'          {mat_id}\n')
            f.write('        </DataArray>\n')

            f.write('      </CellData>\n')

            f.write('    </Piece>\n')
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')

        print(f"  [ok] VTU file saved")

    def save_all_outputs(self, output_dir='.'):
        """Save all required output files for PI-GNN"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print("\n" + "="*70)
        print("SAVING OUTPUT FILES FOR PI-GNN")
        print("="*70)

        # 1. All nodes and elements (original mesh)
        np.save(output_dir / 'nodes.npy', self.nodes)
        print(f"[ok] Saved: nodes.npy {self.nodes.shape}")

        np.save(output_dir / 'elements.npy', self.elements)
        print(f"[ok] Saved: elements.npy {self.elements.shape}")

        # 2. Material classification for nodes (all matrix)
        matrix_node_mask = np.ones(self.num_nodes, dtype=np.int64)
        np.save(output_dir / 'matrix_node_mask.npy', matrix_node_mask)
        print(f"[ok] Saved: matrix_node_mask.npy {matrix_node_mask.shape} (sum={matrix_node_mask.sum()})")

        # 3. Material classification for elements (all matrix -> 0)
        element_material_ids = self.generate_element_material_ids()
        np.save(output_dir / 'element_material_ids.npy', element_material_ids)
        print(f"[ok] Saved: element_material_ids.npy {element_material_ids.shape} (all matrix)")

        # 4. Matrix element array
        matrix_elements = np.array([self.elements[i] for i in sorted(self.matrix_elements)],
                                   dtype=np.int64)
        np.save(output_dir / 'matrix_elements.npy', matrix_elements)
        print(f"[ok] Saved: matrix_elements.npy {matrix_elements.shape}")

        # 5. Interior points mask
        interior_mask = self.generate_interior_mask()
        np.save(output_dir / 'interior_points.npy', interior_mask)
        print(f"[ok] Saved: interior_points.npy {interior_mask.shape}")

        # 6. Node features
        node_features = self.generate_node_features()
        np.save(output_dir / 'node_features.npy', node_features)
        print(f"[ok] Saved: node_features.npy {node_features.shape}")
        print(f"    Features: [Dirichlet, Neumann, Material_ID, Interface, X, Y]")

        # 7. Loading facets (right edge)
        np.save(output_dir / 'loading_surface_facets.npy', self.loading_facets)
        print(f"[ok] Saved: loading_surface_facets.npy {self.loading_facets.shape}")

        # 8. Hole-boundary (free-surface) facets
        hole_facets = np.asarray(self.hole_boundary_facets, dtype=np.int64)
        np.save(output_dir / 'hole_boundary_facets.npy', hole_facets)
        print(f"[ok] Saved: hole_boundary_facets.npy {hole_facets.shape}")

        # 9. Node topology
        node_topology = self.compute_node_topology()
        np.save(output_dir / 'node_topology.npy', node_topology)
        print(f"[ok] Saved: node_topology.npy {node_topology.shape}")

        # 10. Boundary node sets (useful for BC application)
        boundary_info = {
            'left_edge_nodes': sorted(list(self.left_edge_nodes)),
            'right_edge_nodes': sorted(list(self.right_edge_nodes)),
            'top_edge_nodes': sorted(list(self.top_edge_nodes)),
            'bottom_edge_nodes': sorted(list(self.bottom_edge_nodes))
        }
        np.save(output_dir / 'boundary_nodes.npy', boundary_info)
        print(f"[ok] Saved: boundary_nodes.npy (dictionary with boundary node sets)")

        # 11. Material property indices (all matrix -> 0)
        material_node_ids = np.zeros(self.num_nodes, dtype=np.int64)
        np.save(output_dir / 'material_node_ids.npy', material_node_ids)
        print(f"[ok] Saved: material_node_ids.npy {material_node_ids.shape}")

        # 12. VTU file for ParaView
        self.export_to_vtu(output_dir / 'composite_mesh.vtu')

        # 13. Visualization
        self.visualize_composite(output_dir / 'composite_mesh_visualization.png')

        # 14. Save configuration summary
        loaded_len = (float(self._facet_lengths(self.loading_facets).sum())
                      if len(self.loading_facets) else 0.0)
        x0, x1 = float(self.nodes[:, 0].min()), float(self.nodes[:, 0].max())
        y0, y1 = float(self.nodes[:, 1].min()), float(self.nodes[:, 1].max())

        summary = {
            'problem_type': 'hole_plate_tension',
            'dimension': 2,
            'cell_type': 'triangle',
            # ── Plate geometry. Recorded explicitly rather than re-derived
            #    downstream: the trainer sizes OUTPUT_DISPLACEMENT_SCALE from
            #    plate_size, and the FEM comparison reports the domain from
            #    these numbers, so both must read the mesh that was built.
            'plate_size': self.plate_size,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'x_min': x0, 'x_max': x1,
            'y_min': y0, 'y_max': y1,
            'loaded_edge_length': loaded_len,
            'num_nodes': int(self.num_nodes),
            'num_elements': int(self.num_elements),
            'matrix_nodes': int(len(self.matrix_nodes)),
            'matrix_elements': int(len(self.matrix_elements)),
            'hole_boundary_facets': int(len(self.hole_boundary_facets)),
            'loading_facets': int(len(self.loading_facets)),
            'left_edge_nodes': int(len(self.left_edge_nodes)),
            'right_edge_nodes': int(len(self.right_edge_nodes)),
            'holes': self.inclusions,
            # Provenance: the BC set this case is meant to be solved with. The
            # mesh geometry/topology does not depend on it — both the PI-GNN and
            # the FEM locate these edges themselves — but recording it keeps a
            # case directory self-describing.
            'boundary_conditions': {
                'x_min':     'roller, ux = 0 (uy free)',
                'top_left':  'pin, uy = 0 (removes the last rigid-body mode)',
                'x_max':     'Neumann, uniform traction t = (T, 0) in +x',
                'hole_rims': 'free, traction-free exterior surfaces',
                'note':      'plane stress; the traction is a dead load, so '
                             'W_ext is a genuine potential and Pi = E_int - W_ext',
            },
        }

        with open(output_dir / 'mesh_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[ok] Saved: mesh_summary.json")

        print("\n" + "="*70)
        print("ALL FILES GENERATED SUCCESSFULLY!")
        print("="*70)
        print("\nFiles for PI-GNN:")
        print("  Core mesh:")
        print("    - nodes.npy, elements.npy")
        print("  Material classification (single matrix phase):")
        print("    - matrix_node_mask.npy, matrix_elements.npy")
        print("    - element_material_ids.npy (all 0), material_node_ids.npy (all 0)")
        print("  Features and topology:")
        print("    - node_features.npy (6 features; Material_ID/Interface = 0)")
        print("    - node_topology.npy")
        print("  Boundaries and free surfaces:")
        print("    - hole_boundary_facets.npy (void rims, traction-free)")
        print("    - loading_surface_facets.npy (loading boundary)")
        print("    - boundary_nodes.npy (all boundary node sets)")
        print("  Visualization:")
        print("    - composite_mesh.vtu (ParaView)")
        print("    - composite_mesh_visualization.png")


def main():
    """Rebuild the .npy arrays from a saved collocation-point set.

    The processor no longer reads a mesh file: it takes arrays. This entry point
    therefore regenerates the triangulation from the points before processing.
    For the normal path use ``meshing_pipeline.py``, which runs both stages.
    """
    import sys

    print("="*70)
    print("HOLE-PLATE MESH PROCESSOR FOR PI-GNN MODELS")
    print("="*70)
    print()

    if len(sys.argv) < 3:
        print("Usage: python mesh_processor.py <collocation_points.npy> "
              "<config_json> [output_dir] [plate_size]")
        print()
        print("Example:")
        print("  python mesh_processor.py out/collocation_points_composite.npy "
              "hole_config.json ./out 20")
        return 0

    points_file = sys.argv[1]
    config_file = sys.argv[2]
    output_dir  = sys.argv[3] if len(sys.argv) > 3 else './output'
    plate_size  = float(sys.argv[4]) if len(sys.argv) > 4 else None

    try:
        from mesh_generator import CompositeMeshGenerator
        from inclusion_shapes import load_inclusion_config

        points = np.load(points_file)
        inclusions, inferred_size = load_inclusion_config(config_file)
        if plate_size is None:
            plate_size = inferred_size

        mesh_gen = CompositeMeshGenerator(points, inclusions, plate_size)
        nodes, elements = mesh_gen.generate_mesh()

        processor = CompositeMeshProcessor(
            nodes, elements, config_file, plate_size=plate_size)
        processor.classify_materials()
        processor.identify_boundaries()
        processor.find_loading_facets()
        processor.find_hole_boundary_facets()
        processor.save_all_outputs(output_dir)

        print()
        print("Next steps:")
        print("  1. Open 'composite_mesh.vtu' in ParaView to verify the mesh")
        print("  2. Check 'composite_mesh_visualization.png' for an overview")
        print("  3. Train:  python run_pipeline.py --skip-meshing --mesh-dir <dir>")
        print()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
