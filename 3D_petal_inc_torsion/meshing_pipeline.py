"""
Meshing Pipeline: JSON Config -> Cross-Section Points -> Extruded 3-D Mesh -> NPY
================================================================================
Builds the PI-GNN-ready mesh of the composite rod:

1. Load the inclusion cross-section configuration from JSON
2. Generate adaptive collocation points on the circular cross-section
3. Delaunay-triangulate it and classify matrix / inclusion / interface
4. Extrude along +z into tetrahedra (straight prismatic inclusion)
5. Process to the .npy arrays + VTU the trainer and the FEM comparison read

"""

import numpy as np
import json
import sys
from pathlib import Path
import argparse

from collocation_generator import (CompositeCollocationGenerator,
                                             load_inclusion_config)
from mesh_generator import CompositeMeshGenerator
from mesh_processor import CompositeMeshProcessor
from inclusion_shapes import describe, infer_plate_center


def run_meshing_pipeline(config_file: str,
                           plate_radius: float = None,
                           plate_center=None,
                           height: float = 2.0,
                           n_layers: int = 8,
                           matrix_near_field_factor: float = 3.0,
                           inclusion_refinement_factor: float = 0.5,
                           min_density_factor: float = 0.05,
                           max_density_factor: float = 0.3,
                           growth_rate: float = None,
                           output_dir: str = './output',
                           run_mesh_processor: bool = True):
    """
    Build the composite rod mesh: circular cross-section with a prismatic
    inclusion, extruded to tetrahedra.

    Parameters
    ----------
    config_file : str
        JSON file with the inclusion cross-section configuration.
    plate_radius : float, optional
        Outer radius R of the rod (auto-inferred from the inclusion if None).
    plate_center : (float, float), optional
        Torsion axis (defaults to the inclusion centre, giving a concentric rod).
    height : float
        Rod length H along z. Clamped at z=0, twisted at z=H.
    n_layers : int
        Number of element layers through the depth. Aim for a layer thickness
        H/n_layers comparable to the in-plane element size — the generator
        reports the ratio, and slivers cost accuracy in both solvers.
    matrix_near_field_factor : float
        Matrix refinement extends to (char_length x this) outward.
    inclusion_refinement_factor : float
        Inclusion refinement zone = (char_length x this) inward.
    min_density_factor : float
        Minimum spacing = inclusion char_length x this.
    max_density_factor : float
        Maximum spacing = (2 x plate_radius) x this.
    growth_rate : float, optional
        Element growth rate for the fine->coarse transition.
    output_dir : str
        Directory to save all outputs.
    run_mesh_processor : bool
        If True, run the mesh processor to emit the .npy arrays.
    """

    print("="*70)
    print(" COMPOSITE ROD MESH GENERATION PIPELINE ".center(70))
    print("="*70)
    print()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # ========================================================================
    # STAGE 1: Load Configuration
    # ========================================================================
    print("="*70)
    print(" STAGE 1: Loading Configuration ".center(70))
    print("="*70)

    inclusions, inferred_radius = load_inclusion_config(config_file)
    if plate_radius is None:
        plate_radius = inferred_radius
        print(f"Auto-detected cross-section radius: {plate_radius}")
    if plate_center is None:
        plate_center = infer_plate_center(inclusions)
        print(f"Torsion axis (from inclusion centre): "
              f"({plate_center[0]:.6f}, {plate_center[1]:.6f})")

    print()
    print("Configuration:")
    print(f"  Config file:                 {config_file}")
    print(f"  Cross-section radius R:      {plate_radius}")
    print(f"  Torsion axis:                ({plate_center[0]:.6f}, {plate_center[1]:.6f})")
    print(f"  Rod height H:                {height}")
    print(f"  Extrusion layers:            {n_layers}")
    print(f"  Aspect ratio H/(2R):         {height / (2.0 * plate_radius):.3f}")
    print(f"  Matrix near-field factor:    {matrix_near_field_factor}")
    print(f"  Inclusion refinement factor: {inclusion_refinement_factor}")
    print(f"  Min density factor:          {min_density_factor}")
    print(f"  Max density factor:          {max_density_factor}")
    print(f"  Growth rate:                 {growth_rate if growth_rate else 'auto (near-field band)'}")
    print(f"  Output directory:            {output_dir}")
    print()

    print(f"Loaded {len(inclusions)} inclusions:")
    for i, inc in enumerate(inclusions):
        print(f"  Inclusion {i+1}: {describe(inc)}")
    print()

    # ========================================================================
    # STAGE 2: Cross-Section Collocation Points
    # ========================================================================
    print("="*70)
    print(" STAGE 2: Generating Cross-Section Collocation Points ".center(70))
    print("="*70)

    generator = CompositeCollocationGenerator(
        plate_radius=plate_radius,
        inclusions=inclusions,
        plate_center=plate_center,
        matrix_near_field_factor=matrix_near_field_factor,
        inclusion_refinement_factor=inclusion_refinement_factor,
        min_density_factor=min_density_factor,
        max_density_factor=max_density_factor,
        growth_rate=growth_rate
    )

    points = generator.generate()

    points_file = output_path / 'collocation_points_composite.npy'
    np.save(points_file, points)
    print(f"\n[ok] Saved: {points_file}")

    generator.visualize_density_field(output_path / 'density_field_composite.png')
    generator.visualize_points(points, output_path / 'collocation_points_composite.png')

    # ========================================================================
    # STAGE 3: Cross-Section Mesh + Material Classification
    # ========================================================================
    print("\n" + "="*70)
    print(" STAGE 3: Cross-Section Mesh (Material Classification) ".center(70))
    print("="*70)

    mesh_gen = CompositeMeshGenerator(points, inclusions, plate_radius, plate_center)
    section_nodes, section_tris = mesh_gen.generate_mesh()
    mesh_gen.analyze_mesh_quality()

    mesh_gen.visualize_mesh(output_path / 'mesh_composite_full.png', show_materials=True)
    mesh_gen.visualize_mesh(output_path / 'mesh_composite_simple.png', show_materials=False)

    for i, shape in enumerate(mesh_gen.shapes):
        cx, cy = shape.center()
        mesh_gen.visualize_mesh_detail(
            center=(cx, cy),
            size=shape.char_length() * 3,
            output_path=output_path / f'mesh_composite_detail_{i+1}.png'
        )

    # ========================================================================
    # STAGE 4: Extrusion to 3-D Tetrahedra
    # ========================================================================
    print("\n" + "="*70)
    print(" STAGE 4: Extruding to 3-D Tetrahedra ".center(70))
    print("="*70)

    nodes_3d, tets, tet_mat = mesh_gen.extrude_to_3d(height=height, n_layers=n_layers)
    mesh_gen.analyze_tet_quality()

    # ========================================================================
    # STAGE 5: Process Mesh to NPY Files for PI-GNN
    # ========================================================================
    if run_mesh_processor:
        print("\n" + "="*70)
        print(" STAGE 5: Generating NPY Files for PI-GNN ".center(70))
        print("="*70)

        # The case directory carries its own copy of the inclusion config, so it
        # is self-describing: the FEM comparison and the near-field metrics both
        # read the geometry back from here rather than from the original path.
        config_out = output_path / 'config.json'
        with open(config_out, 'w') as f:
            json.dump(inclusions[0] if len(inclusions) == 1 else inclusions, f, indent=2)

        processor = CompositeMeshProcessor(
            nodes_3d, tets, str(config_out),
            plate_radius=plate_radius,
            plate_center=plate_center,
            height=height,
            n_layers=n_layers,
            n_section_nodes=len(section_nodes),
            n_section_elements=len(section_tris),
        )
        processor.classify_materials()
        processor.identify_boundaries()
        processor.find_loading_facets()
        processor.find_interface_facets()
        processor.save_all_outputs(str(output_path))

        # Append the meshing parameters to the summary the processor wrote.
        mesh_summary_path = output_path / 'mesh_summary.json'
        if mesh_summary_path.exists():
            with open(mesh_summary_path, 'r') as f:
                mesh_summary = json.load(f)

            mesh_summary['mesh_generation_parameters'] = {
                'config_file': str(config_file),
                'problem_type': 'inclusion_rod_torsion',
                'plate_radius': float(plate_radius),
                'plate_center': [float(plate_center[0]), float(plate_center[1])],
                'height': float(height),
                'n_layers': int(n_layers),
                'matrix_near_field_factor': float(matrix_near_field_factor),
                'inclusion_refinement_factor': float(inclusion_refinement_factor),
                'min_density_factor': float(min_density_factor),
                'max_density_factor': float(max_density_factor),
                'growth_rate': (float(growth_rate) if growth_rate else None)
            }

            with open(mesh_summary_path, 'w') as f:
                json.dump(mesh_summary, f, indent=2)
            print("[ok] Updated: mesh_summary.json with mesh_generation_parameters")

        print("\n[ok] NPY files generated successfully!")

    # ========================================================================
    print("\n" + "="*70)
    print(" PIPELINE COMPLETE ".center(70))
    print("="*70)
    print()
    print("Generated files:")
    print(f"  {output_dir}/")
    print(f"    collocation_points_composite.npy  - cross-section points")
    print(f"    density_field_composite.png       - density function")
    print(f"    collocation_points_composite.png  - points visualization")
    print(f"    mesh_composite_full.png           - cross-section with materials")
    print(f"    mesh_composite_detail_*.png       - interface detail views")
    print(f"    config.json                       - inclusion config")

    if run_mesh_processor:
        print()
        print("    PI-GNN data files:")
        print(f"    nodes.npy (N,3), elements.npy (E,4)")
        print(f"    node_features.npy (N,7), node_topology.npy")
        print(f"    element_material_ids.npy, material_node_ids.npy")
        print(f"    loading_surface_facets.npy (z=H triangles), interface_facets.npy")
        print(f"    boundary_nodes.npy, composite_mesh.vtu, mesh_summary.json")

    print()
    print("Next steps:")
    print("  1. Check mesh_composite_full.png for the cross-section refinement")
    print("  2. Open composite_mesh.vtu in ParaView to inspect the rod")
    print("  3. Train:  python run_pipeline.py --skip-meshing --mesh-dir <dir>")
    print()
    print("="*70)


def main():
    """Main entry point with command-line argument parsing"""

    parser = argparse.ArgumentParser(
        description='Generate the composite rod mesh and NPY files from an inclusion config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default rod: R inferred from the inclusion, H=2.0, 8 layers
  python meshing_pipeline.py petal_config.json

  # Explicit geometry
  python meshing_pipeline.py petal_config.json --plate-radius 1.0 --height 2.0 --n-layers 10

  # Coarser / finer cross-section
  python meshing_pipeline.py petal_config.json --min-density 0.10 --max-density 0.08

  # Skip the NPY generation (mesh preview only)
  python meshing_pipeline.py petal_config.json --no-process
        """
    )

    parser.add_argument('config', type=str,
                       help='JSON file with the inclusion cross-section configuration')
    parser.add_argument('--plate-radius', type=float, default=None,
                       help='Outer radius R of the rod (auto-inferred if not given)')
    parser.add_argument('--height', type=float, default=2.0,
                       help='Rod length H along z (default: 2.0)')
    parser.add_argument('--n-layers', type=int, default=10,
                       help='Element layers through the depth (default: 10)')
    parser.add_argument('--matrix-near-field', type=float, default=2.0,
                       help='Matrix near-field factor (default: 2.0)')
    parser.add_argument('--inclusion-refinement', type=float, default=1.5,
                       help='Inclusion refinement factor (default: 1.5)')
    parser.add_argument('--min-density', type=float, default=0.16,
                       help='Min spacing factor at the interface (default: 0.16)')
    parser.add_argument('--max-density', type=float, default=0.075,
                       help='Max spacing factor in the far field (default: 0.075)')
    parser.add_argument('--growth-rate', type=float, default=None,
                       help='Element growth rate for the fine->coarse transition '
                            '(e.g. 0.12). Default: auto (spans the near-field band).')
    parser.add_argument('--output', '-o', type=str, default='./output_rod_mesh',
                       help='Output directory (default: ./output_rod_mesh)')
    parser.add_argument('--no-process', action='store_true',
                       help='Skip mesh processing to NPY files')

    args = parser.parse_args()

    try:
        run_meshing_pipeline(
            config_file=args.config,
            plate_radius=args.plate_radius,
            height=args.height,
            n_layers=args.n_layers,
            matrix_near_field_factor=args.matrix_near_field,
            inclusion_refinement_factor=args.inclusion_refinement,
            min_density_factor=args.min_density,
            max_density_factor=args.max_density,
            growth_rate=args.growth_rate,
            output_dir=args.output,
            run_mesh_processor=not args.no_process
        )
        return 0
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
