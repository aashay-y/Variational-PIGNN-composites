"""
Meshing Pipeline: JSON Config -> Collocation Points -> Hole Plate Mesh -> NPY
=============================================================================
Builds the PI-GNN-ready mesh of the 2-D plate with circular holes:

1. Load the hole configuration from JSON
2. Generate adaptive collocation points (refined at each hole boundary)
3. Delaunay-triangulate and carve out the voids (orphaned nodes dropped and the
   connectivity renumbered), leaving one homogeneous matrix phase whose hole
   rims are exterior traction-free free surfaces
4. Process to the .npy arrays + VTU the trainer and the FEM comparison read

The mesh arrays are handed from the generator to the processor DIRECTLY, in
memory. There is no DOLFIN-XML intermediate: it gave the mesh two on-disk
representations that could drift apart, and ``composite_mesh.vtu`` — which the
FEM reads — is written from the same arrays the trainer loads.
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
from inclusion_shapes import describe


def run_meshing_pipeline(config_file: str,
                         plate_size: float = None,
                         matrix_near_field_factor: float = 3.0,
                         inclusion_refinement_factor: float = 0.5,
                         min_density_factor: float = 0.05,
                         max_density_factor: float = 0.3,
                         growth_rate: float = None,
                         output_dir: str = './output',
                         run_mesh_processor: bool = True):
    """
    Build the hole-plate mesh: square plate, circular voids, graded triangles.

    Parameters
    ----------
    config_file : str
        JSON file with the hole configuration (circular voids).
    plate_size : float, optional
        Side length of the square plate (auto-inferred from the config if None).
    matrix_near_field_factor : float
        Matrix refinement extends to (char_length x this) outward from each rim.
    inclusion_refinement_factor : float
        Hole-interior refinement zone = char_length x this inward. Those points
        are seeded so the triangulation is well graded up to the rim, then
        removed with the void.
    min_density_factor : float
        Minimum spacing at the hole boundary = char_length x this.
    max_density_factor : float
        Maximum spacing in the far field = plate_size x this.
    growth_rate : float, optional
        Element growth rate for the fine->coarse transition.
    output_dir : str
        Directory to save all outputs.
    run_mesh_processor : bool
        If True, run the mesh processor to emit the .npy arrays.
    """

    print("="*70)
    print(" HOLE-PLATE MESH GENERATION PIPELINE ".center(70))
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

    inclusions, inferred_size = load_inclusion_config(config_file)
    if plate_size is None:
        plate_size = inferred_size
        print(f"Auto-detected plate size: {plate_size}x{plate_size}")

    print()
    print("Configuration:")
    print(f"  Config file:                 {config_file}")
    print(f"  Plate size:                  {plate_size}")
    print(f"  Matrix near-field factor:    {matrix_near_field_factor}")
    print(f"  Hole-interior refine factor: {inclusion_refinement_factor}")
    print(f"  Min density factor:          {min_density_factor}")
    print(f"  Max density factor:          {max_density_factor}")
    print(f"  Growth rate:                 {growth_rate if growth_rate else 'auto (near-field band)'}")
    print(f"  Output directory:            {output_dir}")
    print()

    print(f"Loaded {len(inclusions)} holes:")
    for i, inc in enumerate(inclusions):
        print(f"  Hole {i+1}: {describe(inc)}")
    print()

    # ========================================================================
    # STAGE 2: Collocation Points
    # ========================================================================
    print("="*70)
    print(" STAGE 2: Generating Collocation Points (matrix + hole interiors) ".center(70))
    print("="*70)

    generator = CompositeCollocationGenerator(
        plate_size=plate_size,
        inclusions=inclusions,
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
    # STAGE 3: Delaunay Mesh + Void Removal
    # ========================================================================
    print("\n" + "="*70)
    print(" STAGE 3: Generating Mesh (carving out the voids) ".center(70))
    print("="*70)

    mesh_gen = CompositeMeshGenerator(points, inclusions, plate_size)
    nodes, elements = mesh_gen.generate_mesh()
    mesh_gen.analyze_mesh_quality()

    mesh_gen.visualize_mesh(output_path / 'mesh_composite_full.png', show_materials=True)
    mesh_gen.visualize_mesh(output_path / 'mesh_composite_simple.png', show_materials=False)

    for i, shape in enumerate(mesh_gen.shapes):
        cx, cy = shape.center()
        mesh_gen.visualize_mesh_detail(
            center=(cx, cy),
            size=shape.char_length() * 8,
            output_path=output_path / f'mesh_composite_detail_{i+1}.png'
        )

    # ========================================================================
    # STAGE 4: Process Mesh to NPY Files for PI-GNN
    # ========================================================================
    if run_mesh_processor:
        print("\n" + "="*70)
        print(" STAGE 4: Generating NPY Files for PI-GNN ".center(70))
        print("="*70)

        # The case directory carries its own copy of the hole config, so it is
        # self-describing: the FEM comparison and the near-field metrics both
        # read the geometry back from here rather than from the original path.
        config_out = output_path / 'config.json'
        with open(config_out, 'w') as f:
            json.dump(inclusions[0] if len(inclusions) == 1 else inclusions, f, indent=2)

        processor = CompositeMeshProcessor(
            nodes, elements, str(config_out),
            plate_size=plate_size,
        )
        processor.classify_materials()
        processor.identify_boundaries()
        processor.find_loading_facets()
        processor.find_hole_boundary_facets()
        processor.save_all_outputs(str(output_path))

        # Append the meshing parameters to the summary the processor wrote.
        mesh_summary_path = output_path / 'mesh_summary.json'
        if mesh_summary_path.exists():
            with open(mesh_summary_path, 'r') as f:
                mesh_summary = json.load(f)

            mesh_summary['mesh_generation_parameters'] = {
                'config_file': str(config_file),
                'problem_type': 'hole_plate_tension',
                'plate_size': float(plate_size),
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
    print(f"    collocation_points_composite.npy  - generated points")
    print(f"    density_field_composite.png       - density function")
    print(f"    collocation_points_composite.png  - points visualization")
    print(f"    mesh_composite_full.png           - mesh with the hole rims")
    print(f"    mesh_composite_detail_*.png       - detail views (hole rims)")
    print(f"    config.json                       - hole config")

    if run_mesh_processor:
        print()
        print("    PI-GNN data files:")
        print(f"    nodes.npy (N,2), elements.npy (E,3)")
        print(f"    node_features.npy (N,6), node_topology.npy")
        print(f"    element_material_ids.npy, material_node_ids.npy")
        print(f"    loading_surface_facets.npy (right-edge segments)")
        print(f"    hole_boundary_facets.npy (void rims, traction-free)")
        print(f"    boundary_nodes.npy, composite_mesh.vtu, mesh_summary.json")

    print()
    print("Next steps:")
    print("  1. Check mesh_composite_full.png for the rim refinement")
    print("  2. Open composite_mesh.vtu in ParaView to inspect the mesh")
    print("  3. Train:  python run_pipeline.py --skip-meshing --mesh-dir <dir>")
    print()
    print("="*70)


def main():
    """Main entry point with command-line argument parsing"""

    parser = argparse.ArgumentParser(
        description='Generate the hole-plate mesh and NPY files from a hole configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default plate: size inferred from the hole config
  python meshing_pipeline.py hole_config.json

  # Explicit geometry
  python meshing_pipeline.py hole_config.json --plate-size 20

  # Coarser / finer mesh
  python meshing_pipeline.py hole_config.json --min-density 0.03 --max-density 0.04

  # Skip the NPY generation (mesh preview only)
  python meshing_pipeline.py hole_config.json --no-process
        """
    )

    parser.add_argument('config', type=str,
                        help='JSON file with the hole configuration (circular voids)')
    parser.add_argument('--plate-size', type=float, default=None,
                        help='Square plate side length (auto-inferred if not given)')
    parser.add_argument('--matrix-near-field', type=float, default=2.0,
                        help='Matrix near-field factor (default: 2.0)')
    parser.add_argument('--inclusion-refinement', type=float, default=1.5,
                        help='Hole-interior refinement factor (default: 1.5)')
    parser.add_argument('--min-density', type=float, default=0.02,
                        help='Min spacing factor at the hole rim (default: 0.02)')
    parser.add_argument('--max-density', type=float, default=0.03,
                        help='Max spacing factor in the far field (default: 0.03)')
    parser.add_argument('--growth-rate', type=float, default=None,
                        help='Element growth rate for the fine->coarse transition '
                             '(e.g. 0.12). Default: auto (spans the near-field band).')
    parser.add_argument('--output', '-o', type=str, default='./output_plate_hole',
                        help='Output directory (default: ./output_plate_hole)')
    parser.add_argument('--no-process', action='store_true',
                        help='Skip mesh processing to NPY files')

    args = parser.parse_args()

    try:
        run_meshing_pipeline(
            config_file=args.config,
            plate_size=args.plate_size,
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
