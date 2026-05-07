#!/usr/bin/env python3
"""
unwrap_center_ala.py

Unwrap the solute (make it continuous across PBC) and optionally center it in the box each frame.

Requires MDAnalysis.

Example:
  python3 unwrap_center_ala.py --top alanine_amber_solvated.pdb --traj alanine_amber_prod.dcd --out_dcd unwrapped_alanine_prod.dcd --center
"""
import argparse
import MDAnalysis as mda
from MDAnalysis.transformations import unwrap, center_in_box

def main():
    p = argparse.ArgumentParser(description="Unwrap solute across PBC and optionally center it.")
    p.add_argument("--top", required=True, help="Topology PDB used in simulation (solvated PDB).")
    p.add_argument("--traj", required=True, help="Input DCD trajectory (wrapped).")
    p.add_argument("--out_dcd", default="unwrapped_prod.dcd", help="Output unwrapped DCD filename.")
    p.add_argument("--out_pdb", default="unwrapped_final.pdb", help="Final PDB (last frame after transforms).")
    p.add_argument("--center", action="store_true", help="Center the solute in the box each frame (recommended).")
    p.add_argument("--solute_sel", default="not resname TIP3 WAT SOL HOH", help="Selection for solute (default excludes common water names).")
    args = p.parse_args()

    print("Loading Universe...")
    u = mda.Universe(args.top, args.traj)
    print(f"Atoms: {u.atoms.n_atoms}, frames: {len(u.trajectory)}")

    # Select solute (default: everything except common water residue names)
    try:
        solute = u.select_atoms(args.solute_sel)
    except Exception:
        solute = u.select_atoms("not resname TIP3 WAT SOL HOH")
    print(f"Selected solute atoms: {solute.n_atoms}")

    # Build transformations: unwrap molecules (make fragments whole)
    # unwrap(u.atoms) will unwrap by fragments (residues/chains) so molecules don't jump
    transforms = [unwrap(u.atoms)]

    # Center solute each frame (optional). This recenters the selected solute in the periodic box.
    if args.center:
        transforms.append(center_in_box(solute))

    # Add transforms to trajectory iterator
    u.trajectory.add_transformations(*transforms)


    # Write transformed trajectory with only solute (non-water) atoms
    print("Writing transformed trajectory to:", args.out_dcd)
    with mda.Writer(args.out_dcd, solute.n_atoms) as W:
        for ts in u.trajectory:
            W.write(solute)

    # Write final PDB of last transformed frame (solute only)
    # print("Writing final PDB:", args.out_pdb)
    # with mda.Writer(args.out_pdb, solute.n_atoms) as P:
    #     P.write(solute)

    print("Done. Output DCD:", args.out_dcd)

if __name__ == "__main__":
    main()
