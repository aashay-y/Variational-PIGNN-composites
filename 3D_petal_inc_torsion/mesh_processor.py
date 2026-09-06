"""
Composite Rod Mesh Processor for PI-GNN Training
================================================
Turns the extruded 3-D tetrahedral mesh of the composite rod (matrix + prismatic
inclusion) into the .npy arrays the PI-GNN and the FEniCSx comparison consume.

The rod is clamped at z = 0 and twisted by a tangential surface traction on
z = H, so the processor identifies:

  - ``fixed_face_nodes``   z = 0        Dirichlet, u = 0 (all three components)
  - ``loaded_face_nodes``  z = H        Neumann, tangential traction (torque)
  - ``lateral_nodes``      r = R        free surface
  - ``loading_surface_facets``          the TRIANGLES of the z = H face, with the
                                        tetrahedron that owns each one

Because the mesh is a straight extrusion, node ids are layer-major
(``node = layer * N_section + node_2d``) and material classification is done on
the planar cross-section coordinates alone — the inclusion is the same shape at
every height.

"""

import numpy as np
import json
import itertools
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")   # headless backend (see inclusion_shapes.py for why).
import matplotlib.pyplot as plt
from pathlib import Path

from inclusion_shapes import make_inclusion, describe

# The four triangular faces of a tetrahedron, as local vertex triples.
_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
# The six edges of a tetrahedron, as local vertex pairs.
_TET_EDGES = tuple(itertools.combinations(range(4), 2))


class CompositeMeshProcessor:
    """Process the 3-D composite rod tetrahedral mesh for FEM / GNN."""

    def __init__(self, nodes, elements, config_filepath=None,
                 plate_radius=None, plate_center=None, height=None,
                 n_layers=None, n_section_nodes=None, n_section_elements=None):
        """
        Parameters
        ----------
        nodes : (N, 3) array
            3-D node coordinates from ``CompositeMeshGenerator.extrude_to_3d``.
        elements : (E, 4) array
            Tetrahedral connectivity.
        config_filepath : str, optional
            JSON config with the inclusion cross-section geometry.
        plate_radius, plate_center, height : float / (float, float) / float
            Rod geometry. ``plate_center`` is the torsion axis; the traction
            field and every cylindrical component are referred to it, so it is
            recorded in mesh_summary.json rather than re-derived downstream.
        n_layers, n_section_nodes, n_section_elements : int, optional
            Extrusion bookkeeping, recorded so post-processing can slice the rod
            at a given height without re-deriving the layer structure.
        """
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.num_nodes = len(self.nodes)
        self.num_elements = len(self.elements)

        if self.nodes.shape[1] != 3:
            raise ValueError(f"Expected (N,3) nodes, got {self.nodes.shape}")
        if self.elements.shape[1] != 4:
            raise ValueError(f"Expected (E,4) tetrahedra, got {self.elements.shape}")

        self.config_filepath = config_filepath

        self.plate_radius = float(plate_radius) if plate_radius is not None else None
        if plate_center is None:
            plate_center = (float(self.nodes[:, 0].mean()), float(self.nodes[:, 1].mean()))
        self.axis_x, self.axis_y = float(plate_center[0]), float(plate_center[1])
        self.height = float(height) if height is not None else float(self.nodes[:, 2].max())
        self.n_layers = int(n_layers) if n_layers is not None else None
        self.n_section_nodes = int(n_section_nodes) if n_section_nodes is not None else None
        self.n_section_elements = (int(n_section_elements)
                                   if n_section_elements is not None else None)

        # Inclusion configuration: raw dicts + derived geometry shapes
        self.inclusions = []
        self.shapes = []

        # Material classification
        self.inclusion_nodes = set()
        self.matrix_nodes = set()
        self.inclusion_elements = set()
        self.matrix_elements = set()
        self.interface_elements = set()

        # Boundary identification
        self.fixed_face_nodes = set()      # z = 0   (clamped)
        self.loaded_face_nodes = set()     # z = H   (twisted)
        self.lateral_nodes = set()         # r = R   (free)
        self.loading_facets = []
        self.interface_facets = []

        print(f"[ok] Composite rod mesh:")
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Tetrahedra: {self.num_elements}")
        print(f"  - X range: [{self.nodes[:, 0].min():.4f}, {self.nodes[:, 0].max():.4f}]")
        print(f"  - Y range: [{self.nodes[:, 1].min():.4f}, {self.nodes[:, 1].max():.4f}]")
        print(f"  - Z range: [{self.nodes[:, 2].min():.4f}, {self.nodes[:, 2].max():.4f}]")

        if config_filepath:
            self._load_config()

    def _load_config(self):
        """Load inclusion configuration from JSON file (single dict or list of dicts)"""
        print(f"\nLoading configuration from: {self.config_filepath}")

        with open(self.config_filepath, 'r') as f:
            config = json.load(f)

        if isinstance(config, dict):
            self.inclusions = [config]
        elif isinstance(config, list):
            self.inclusions = config
        else:
            raise ValueError("Config JSON must be a dict or list of dicts.")

        # Geometry layer: one shape object per inclusion (dispatched on JSON "type")
        self.shapes = [make_inclusion(inc) for inc in self.inclusions]

        print(f"[ok] Inclusion configuration: {len(self.inclusions)} inclusion(s)")
        for i, inc in enumerate(self.inclusions):
            print(f"  - Inclusion {i+1}: {describe(inc)}")

    # ── material classification ──────────────────────────────────────────────

    def classify_materials(self, tolerance=1e-6):
        """
        Classify nodes and tetrahedra as matrix or inclusion.

        The inclusion is a straight prism, so the test is the planar
        signed-distance field evaluated at (x, y) — z plays no part.
        """
        print("\nClassifying materials...")

        if not self.shapes:
            raise ValueError("Inclusion configuration not loaded. Provide config file.")

        # Vectorised: one signed-distance evaluation per shape over all nodes.
        inside = np.zeros(self.num_nodes, dtype=bool)
        for shape in self.shapes:
            sd = np.asarray(shape.signed_distance(self.nodes[:, 0], self.nodes[:, 1]))
            inside |= sd <= tolerance

        self.inclusion_nodes = set(np.where(inside)[0].tolist())
        self.matrix_nodes = set(np.where(~inside)[0].tolist())

        # Element classification: 4 inclusion vertices -> inclusion, 0 -> matrix,
        # anything between -> interface (solved with matrix properties, as in 2-D).
        n_in = inside[self.elements].sum(axis=1)
        self.inclusion_elements = set(np.where(n_in == 4)[0].tolist())
        self.matrix_elements = set(np.where(n_in == 0)[0].tolist())
        self.interface_elements = set(np.where((n_in > 0) & (n_in < 4))[0].tolist())

        print(f"[ok] Material classification:")
        print(f"  - Matrix nodes: {len(self.matrix_nodes)}")
        print(f"  - Inclusion nodes: {len(self.inclusion_nodes)}")
        print(f"  - Matrix tets: {len(self.matrix_elements)}")
        print(f"  - Inclusion tets: {len(self.inclusion_elements)}")
        print(f"  - Interface tets: {len(self.interface_elements)}")

    # ── boundaries ───────────────────────────────────────────────────────────

    def identify_boundaries(self, tolerance=1e-6):
        """
        Identify the clamped face (z=0), the loaded face (z=H) and the free
        lateral surface (r=R).
        """
        print("\nIdentifying boundaries...")

        z = self.nodes[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        z_tol = max(tolerance, 1e-8 * abs(z_max - z_min))

        self.fixed_face_nodes = set(np.where(np.abs(z - z_min) < z_tol)[0].tolist())
        self.loaded_face_nodes = set(np.where(np.abs(z - z_max) < z_tol)[0].tolist())

        r = np.hypot(self.nodes[:, 0] - self.axis_x, self.nodes[:, 1] - self.axis_y)
        R = self.plate_radius if self.plate_radius is not None else float(r.max())
        # The rim is a polygon through the sampled boundary points, so its nodes
        # sit at exactly r = R while the chords between them dip slightly inside.
        # A relative band catches the rim nodes without sweeping in the interior.
        self.lateral_nodes = set(np.where(r > R * (1.0 - 1e-6))[0].tolist())

        print(f"  - Fixed face  (z={z_min:.4f}): {len(self.fixed_face_nodes)} nodes  [u = 0]")
        print(f"  - Loaded face (z={z_max:.4f}): {len(self.loaded_face_nodes)} nodes  [torque]")
        print(f"  - Lateral surface (r={R:.4f}): {len(self.lateral_nodes)} nodes  [free]")

    def _boundary_face_owner(self):
        """Map each boundary triangular face -> the single tet that owns it.

        A face shared by two tets is interior; a face appearing once is on the
        surface. Built once and reused for the loading facets.
        """
        face_count = defaultdict(list)
        for eid, tet in enumerate(self.elements):
            for f in _TET_FACES:
                key = tuple(sorted((int(tet[f[0]]), int(tet[f[1]]), int(tet[f[2]]))))
                face_count[key].append(eid)
        return {k: v[0] for k, v in face_count.items() if len(v) == 1}

    def find_loading_facets(self, tolerance=1e-6):
        """
        Find the triangular faces on the loaded end face z = H.

        Stored as ``[tet_id, n1, n2, n3]`` per row, mirroring the 2-D format
        ``[elem_id, n1, n2]``. The tet id is recorded so the facet can be traced
        back to the element that carries it.
        """
        print("\nFinding loading surface facets...")

        loaded = self.loaded_face_nodes
        owners = self._boundary_face_owner()

        facets = []
        for face, eid in owners.items():
            if all(n in loaded for n in face):
                facets.append([eid, face[0], face[1], face[2]])

        self.loading_facets = np.array(facets, dtype=np.int64).reshape(-1, 4)
        area = self._facet_areas(self.loading_facets).sum() if len(facets) else 0.0
        expected = np.pi * self.plate_radius ** 2 if self.plate_radius else float('nan')
        print(f"  - Loading facets found: {len(self.loading_facets)}")
        print(f"  - Loaded face area: {area:.6f}  (analytic pi*R^2 = {expected:.6f})")

    def _facet_areas(self, facets):
        """Triangle areas of ``[tet_id, n1, n2, n3]`` facet rows."""
        p = self.nodes[facets[:, 1:]]                     # (F, 3, 3)
        return 0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)

    def find_interface_facets(self):
        """Find the tetrahedron faces straddling the matrix-inclusion interface."""
        print("\nFinding interface facets...")

        inclusion = self.inclusion_nodes
        facets = []
        for eid in self.interface_elements:
            tet = self.elements[eid]
            for f in _TET_FACES:
                n = [int(tet[i]) for i in f]
                flags = [nid in inclusion for nid in n]
                # A face on the interface has vertices in both phases.
                if any(flags) and not all(flags):
                    facets.append([eid, n[0], n[1], n[2]])

        self.interface_facets = np.array(facets, dtype=np.int64).reshape(-1, 4)
        print(f"  - Interface facets found: {len(self.interface_facets)}")

    # ── graph / features ─────────────────────────────────────────────────────

    def compute_node_topology(self):
        """Node adjacency from the six edges of every tetrahedron."""
        print("\nComputing node topology...")

        adjacency = defaultdict(set)
        for tet in self.elements:
            for i, j in _TET_EDGES:
                a, b = int(tet[i]), int(tet[j])
                adjacency[a].add(b)
                adjacency[b].add(a)

        max_neighbors = max(len(v) for v in adjacency.values())
        degrees = np.array([len(adjacency[i]) for i in range(self.num_nodes)])
        print(f"  - Max neighbors: {max_neighbors}, mean degree: {degrees.mean():.2f}")

        topology = np.full((self.num_nodes, max_neighbors), -1, dtype=np.int64)
        for node_id in range(self.num_nodes):
            neighbors = sorted(adjacency[node_id])
            topology[node_id, :len(neighbors)] = neighbors

        return topology

    def generate_node_features(self):
        """
        Node feature matrix (N, 7):
            [Dirichlet, Neumann, Material_ID, Interface, X, Y, Z]

        Dirichlet = 1 on the clamped face z=0; Neumann = 1 on the loaded face
        z=H; Material_ID = 1 inside the inclusion; Interface = 1 for a node
        touching a tet that straddles the phases.
        """
        print("\nGenerating node features...")

        features = np.zeros((self.num_nodes, 7), dtype=np.float64)

        fixed = np.zeros(self.num_nodes, dtype=bool)
        fixed[list(self.fixed_face_nodes)] = True
        loaded = np.zeros(self.num_nodes, dtype=bool)
        loaded[list(self.loaded_face_nodes)] = True
        incl = np.zeros(self.num_nodes, dtype=bool)
        incl[list(self.inclusion_nodes)] = True

        # Interface nodes: any node of an interface tet that is itself inside the
        # inclusion (matching the 2-D definition, which flagged the inclusion-side
        # nodes of straddling elements).
        iface = np.zeros(self.num_nodes, dtype=bool)
        if self.interface_elements:
            touched = np.unique(self.elements[sorted(self.interface_elements)].ravel())
            iface[touched] = True
        iface &= incl

        features[:, 0] = fixed.astype(np.float64)
        features[:, 1] = loaded.astype(np.float64)
        features[:, 2] = incl.astype(np.float64)
        features[:, 3] = iface.astype(np.float64)
        features[:, 4:7] = self.nodes

        print(f"  - Feature shape: {features.shape}")
        print(f"    Dirichlet {int(fixed.sum())}, Neumann {int(loaded.sum())}, "
              f"inclusion {int(incl.sum())}, interface {int(iface.sum())}")
        return features

    def generate_element_material_ids(self):
        """(E,) array: 0 = matrix, 1 = inclusion, 2 = interface."""
        material_ids = np.zeros(self.num_elements, dtype=np.int64)
        material_ids[list(self.inclusion_elements)] = 1
        material_ids[list(self.interface_elements)] = 2
        return material_ids

    def generate_interior_mask(self):
        """1.0 for nodes not on any external boundary of the rod."""
        boundary = self.fixed_face_nodes | self.loaded_face_nodes | self.lateral_nodes
        mask = np.ones(self.num_nodes, dtype=np.float64)
        mask[list(boundary)] = 0.0
        return mask

    # ── output ───────────────────────────────────────────────────────────────

    def export_to_vtu(self, output_path):
        """Write the tetrahedral mesh with material tags for ParaView.

        This file is also the mesh the FEniCSx comparison reads, so the node
        ordering here must stay identical to ``nodes.npy``.
        """
        material_ids = self.generate_element_material_ids()
        node_mat = np.zeros(self.num_nodes, dtype=np.int64)
        node_mat[list(self.inclusion_nodes)] = 1
        r = np.hypot(self.nodes[:, 0] - self.axis_x, self.nodes[:, 1] - self.axis_y)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                    'byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')
            f.write(f'    <Piece NumberOfPoints="{self.num_nodes}" '
                    f'NumberOfCells="{self.num_elements}">\n')

            f.write('      <Points>\n')
            f.write('        <DataArray type="Float64" NumberOfComponents="3" '
                    'format="ascii">\n')
            for x, y, z in self.nodes:
                f.write(f'          {x:.10g} {y:.10g} {z:.10g}\n')
            f.write('        </DataArray>\n      </Points>\n')

            f.write('      <Cells>\n')
            f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
            for n0, n1, n2, n3 in self.elements:
                f.write(f'          {int(n0)} {int(n1)} {int(n2)} {int(n3)}\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
            for i in range(1, self.num_elements + 1):
                f.write(f'          {4 * i}\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            for _ in range(self.num_elements):
                f.write('          10\n')      # VTK_TETRA
            f.write('        </DataArray>\n      </Cells>\n')

            f.write('      <PointData Scalars="Material_ID">\n')
            f.write('        <DataArray type="Int64" Name="Material_ID" format="ascii">\n')
            for v in node_mat:
                f.write(f'          {int(v)}\n')
            f.write('        </DataArray>\n')
            f.write('        <DataArray type="Float64" Name="radius" format="ascii">\n')
            for v in r:
                f.write(f'          {v:.10g}\n')
            f.write('        </DataArray>\n')
            f.write('      </PointData>\n')

            f.write('      <CellData Scalars="Element_Material_ID">\n')
            f.write('        <DataArray type="Int64" Name="Element_Material_ID" '
                    'format="ascii">\n')
            for v in material_ids:
                f.write(f'          {int(v)}\n')
            f.write('        </DataArray>\n')
            f.write('      </CellData>\n')

            f.write('    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n')

        print(f"[ok] Saved: {output_path}")

    def visualize_composite(self, output_path=None):
        """Overview figure: cross-section material map + the extruded rod."""
        fig = plt.figure(figsize=(16, 7))

        # Left: the cross-section at mid-height, coloured by phase.
        ax = fig.add_subplot(1, 2, 1)
        z = self.nodes[:, 2]
        z_mid = 0.5 * (z.min() + z.max())
        if self.n_section_nodes:
            # Layer-major ordering: pick the layer nearest mid-height.
            layer = int(round((z_mid - z.min()) / max(self.height, 1e-30)
                              * (self.n_layers or 1)))
            sel = np.arange(self.n_section_nodes) + layer * self.n_section_nodes
        else:
            sel = np.where(np.abs(z - z_mid) < 1e-9)[0]
        node_mat = np.zeros(self.num_nodes)
        node_mat[list(self.inclusion_nodes)] = 1.0
        sc = ax.scatter(self.nodes[sel, 0], self.nodes[sel, 1],
                        c=node_mat[sel], cmap='coolwarm', s=8)
        plt.colorbar(sc, ax=ax, label='0 = matrix, 1 = inclusion')
        for shape in self.shapes:
            shape.draw(ax, edgecolor='k', linewidth=2)
        if self.plate_radius:
            ax.add_patch(plt.Circle((self.axis_x, self.axis_y), self.plate_radius,
                                    fill=False, edgecolor='k', linewidth=2))
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_title(f'Cross-section at z = {z_mid:.3f}', fontweight='bold')

        # Right: 3-D scatter of the surface nodes, so the rod shape is visible.
        ax3 = fig.add_subplot(1, 2, 2, projection='3d')
        surf = sorted(self.lateral_nodes | self.fixed_face_nodes | self.loaded_face_nodes)
        step = max(1, len(surf) // 4000)
        s = np.array(surf[::step])
        ax3.scatter(self.nodes[s, 0], self.nodes[s, 1], self.nodes[s, 2],
                    c=node_mat[s], cmap='coolwarm', s=3, alpha=0.6)
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        ax3.set_title(f'Rod surface — clamped z=0, twisted z={self.height:.2f}',
                      fontweight='bold')
        try:
            ax3.set_box_aspect((1, 1, self.height / max(2 * (self.plate_radius or 1), 1e-9)))
        except Exception:
            pass

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"[ok] Saved: {output_path}")
        plt.close(fig)

    def save_all_outputs(self, output_dir='.'):
        """Save all required output files for PI-GNN"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "="*70)
        print("SAVING OUTPUT FILES FOR PI-GNN")
        print("="*70)

        np.save(output_dir / 'nodes.npy', self.nodes)
        print(f"[ok] Saved: nodes.npy {self.nodes.shape}")

        np.save(output_dir / 'elements.npy', self.elements)
        print(f"[ok] Saved: elements.npy {self.elements.shape}")

        matrix_node_mask = np.zeros(self.num_nodes, dtype=np.int64)
        matrix_node_mask[list(self.matrix_nodes)] = 1
        inclusion_node_mask = np.zeros(self.num_nodes, dtype=np.int64)
        inclusion_node_mask[list(self.inclusion_nodes)] = 1

        np.save(output_dir / 'matrix_node_mask.npy', matrix_node_mask)
        print(f"[ok] Saved: matrix_node_mask.npy (sum={matrix_node_mask.sum()})")
        np.save(output_dir / 'inclusion_node_mask.npy', inclusion_node_mask)
        print(f"[ok] Saved: inclusion_node_mask.npy (sum={inclusion_node_mask.sum()})")

        element_material_ids = self.generate_element_material_ids()
        np.save(output_dir / 'element_material_ids.npy', element_material_ids)
        print(f"[ok] Saved: element_material_ids.npy {element_material_ids.shape}")
        print(f"    Matrix: {np.sum(element_material_ids == 0)}, "
              f"Inclusion: {np.sum(element_material_ids == 1)}, "
              f"Interface: {np.sum(element_material_ids == 2)}")

        for name, ids in (('matrix_elements', self.matrix_elements),
                          ('inclusion_elements', self.inclusion_elements),
                          ('interface_elements', self.interface_elements)):
            arr = self.elements[sorted(ids)] if ids else np.zeros((0, 4), dtype=np.int64)
            np.save(output_dir / f'{name}.npy', arr)
            print(f"[ok] Saved: {name}.npy {arr.shape}")

        interior_mask = self.generate_interior_mask()
        np.save(output_dir / 'interior_points.npy', interior_mask)
        print(f"[ok] Saved: interior_points.npy {interior_mask.shape}")

        node_features = self.generate_node_features()
        np.save(output_dir / 'node_features.npy', node_features)
        print(f"[ok] Saved: node_features.npy {node_features.shape}")
        print(f"    Features: [Dirichlet, Neumann, Material_ID, Interface, X, Y, Z]")

        np.save(output_dir / 'loading_surface_facets.npy', self.loading_facets)
        print(f"[ok] Saved: loading_surface_facets.npy {self.loading_facets.shape}")

        interface_facets = np.asarray(self.interface_facets, dtype=np.int64)
        np.save(output_dir / 'interface_facets.npy', interface_facets)
        print(f"[ok] Saved: interface_facets.npy {interface_facets.shape}")

        node_topology = self.compute_node_topology()
        np.save(output_dir / 'node_topology.npy', node_topology)
        print(f"[ok] Saved: node_topology.npy {node_topology.shape}")

        boundary_info = {
            'fixed_face_nodes': sorted(self.fixed_face_nodes),
            'loaded_face_nodes': sorted(self.loaded_face_nodes),
            'lateral_nodes': sorted(self.lateral_nodes),
        }
        np.save(output_dir / 'boundary_nodes.npy', boundary_info)
        print(f"[ok] Saved: boundary_nodes.npy (dictionary with boundary node sets)")

        material_node_ids = np.zeros(self.num_nodes, dtype=np.int64)
        material_node_ids[list(self.inclusion_nodes)] = 1
        np.save(output_dir / 'material_node_ids.npy', material_node_ids)
        print(f"[ok] Saved: material_node_ids.npy {material_node_ids.shape}")

        self.export_to_vtu(output_dir / 'composite_mesh.vtu')
        self.visualize_composite(output_dir / 'composite_mesh_visualization.png')

        loaded_area = (float(self._facet_areas(self.loading_facets).sum())
                       if len(self.loading_facets) else 0.0)

        summary = {
            'problem_type': 'inclusion_rod_torsion',
            'dimension': 3,
            'cell_type': 'tetrahedron',
            # ── Rod geometry. The torsion axis is stored explicitly because the
            #    traction field t = (tau/R)*(-(y-cy), (x-cx), 0) and every
            #    cylindrical stress component are referred to it. Re-deriving it
            #    downstream from a bounding box would be wrong for a petal, whose
            #    bbox is not centred on the shape.
            'plate_radius': self.plate_radius,
            'axis_x': self.axis_x,
            'axis_y': self.axis_y,
            'height': self.height,
            'n_layers': self.n_layers,
            'n_section_nodes': self.n_section_nodes,
            'n_section_elements': self.n_section_elements,
            'loaded_face_area': loaded_area,
            'num_nodes': int(self.num_nodes),
            'num_elements': int(self.num_elements),
            'matrix_nodes': int(len(self.matrix_nodes)),
            'inclusion_nodes': int(len(self.inclusion_nodes)),
            'matrix_elements': int(len(self.matrix_elements)),
            'inclusion_elements': int(len(self.inclusion_elements)),
            'interface_elements': int(len(self.interface_elements)),
            'interface_facets': int(len(self.interface_facets)),
            'loading_facets': int(len(self.loading_facets)),
            'fixed_face_nodes': int(len(self.fixed_face_nodes)),
            'loaded_face_nodes': int(len(self.loaded_face_nodes)),
            'inclusions': self.inclusions,
            # Provenance: the BC set this case is meant to be solved with. The
            # mesh geometry/topology does not depend on it — both the PI-GNN and
            # the FEM locate these faces from the coordinates — but recording it
            # keeps a case directory self-describing.
            'boundary_conditions': {
                'z_min':   'clamped, u = 0 (all three components)',
                'z_max':   'Neumann, tangential traction t = (tau/R)*(-(y-cy), (x-cx), 0) '
                           '-> pure torque about the z axis, zero net force',
                'lateral': 'free, traction-free cylindrical surface',
                'note':    'the clamped end face removes all six rigid-body modes; '
                           'the twist angle is an OUTPUT of the applied torque',
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
        print("    - nodes.npy (N,3), elements.npy (E,4 tetrahedra)")
        print("  Material classification:")
        print("    - matrix_node_mask.npy, inclusion_node_mask.npy")
        print("    - matrix_elements.npy, inclusion_elements.npy, interface_elements.npy")
        print("    - element_material_ids.npy, material_node_ids.npy")
        print("  Features and topology:")
        print("    - node_features.npy (7 features incl. Z)")
        print("    - node_topology.npy")
        print("  Boundaries and interfaces:")
        print("    - interface_facets.npy (matrix-inclusion interface)")
        print("    - loading_surface_facets.npy (z=H triangles carrying the torque)")
        print("    - boundary_nodes.npy (all boundary node sets)")
        print("  Visualization:")
        print("    - composite_mesh.vtu (ParaView; also read by the FEM)")
        print("    - composite_mesh_visualization.png")
