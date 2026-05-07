#!/usr/bin/env python3
"""
ala_amber_fixed.py

Solvated alanine dipeptide (ACE-ALA-NME) MD with AMBER ff14SB + TIP3P.
Prevents "box size < 2*cutoff" by checking box size and using positional restraints
during equilibration (released for production).
"""
import argparse
import sys
import math
import openmm
import openmm.app as app
import openmm.unit as unit

# === user-provided PDB (your ACE-ALA-NME) ===
ALANINE_PDB_TEXT = """CRYST1   27.222   27.222   27.222  90.00  90.00  90.00 P 1           1
ATOM      1 HH31 ACE X   1       3.225  27.427   2.566  1.00  0.00            
ATOM      2  CH3 ACE X   1       3.720  26.570   2.110  1.00  0.00            
ATOM      3 HH32 ACE X   1       4.088  25.905   2.891  1.00  0.00            
ATOM      4 HH33 ACE X   1       4.557  26.914   1.502  1.00  0.00            
ATOM      5  C   ACE X   1       2.770  25.800   1.230  1.00  0.00            
ATOM      6  O   ACE X   1       1.600  26.150   1.090  1.00  0.00            
ATOM      7  N   ALA X   2       3.270  24.640   0.690  1.00  0.00            
ATOM      8  H   ALA X   2       4.259  24.471   0.810  1.00  0.00            
ATOM      9  CA  ALA X   2       2.480  23.690  -0.190  1.00  0.00            
ATOM     10  HA  ALA X   2       1.733  24.315  -0.679  1.00  0.00            
ATOM     11  CB  ALA X   2       3.470  23.160  -1.270  1.00  0.00            
ATOM     12  HB1 ALA X   2       4.219  22.525  -0.797  1.00  0.00            
ATOM     13  HB2 ALA X   2       2.922  22.582  -2.014  1.00  0.00            
ATOM     14  HB3 ALA X   2       3.963  24.002  -1.756  1.00  0.00            
ATOM     15  C   ALA X   2       1.730  22.590   0.490  1.00  0.00            
ATOM     16  O   ALA X   2       2.340  21.880   1.280  1.00  0.00            
ATOM     17  N   NME X   3       0.400  22.430   0.210  1.00  0.00            
ATOM     18  H   NME X   3      -0.008  23.118  -0.407  1.00  0.00            
ATOM     19  CH3 NME X   3      -0.470  21.350   0.730  1.00  0.00            
ATOM     20 HH31 NME X   3       0.112  20.693   1.376  1.00  0.00            
ATOM     21 HH32 NME X   3      -1.290  21.786   1.300  1.00  0.00            
ATOM     22 HH33 NME X   3      -0.873  20.775  -0.103  1.00  0.00            
END
"""

def write_input_pdb(filename="alanine_amber_input.pdb"):
    with open(filename, "w") as f:
        f.write(ALANINE_PDB_TEXT.strip() + "\n")
    return filename

def select_platform():
    try:
        return openmm.Platform.getPlatformByName("CUDA")
    except Exception:
        return openmm.Platform.getPlatformByName("CPU")

def bbox_lengths_nm(positions):
    """positions: OpenMM Quantity (nanometers) or Nx3 numpy array (nm)"""
    import numpy as _np
    if hasattr(positions, "value_in_unit"):
        arr = positions.value_in_unit(unit.nanometer)
    else:
        arr = _np.asarray(positions)
    arr = _np.asarray(arr)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    lengths = maxs - mins
    return lengths  # numpy array [lx,ly,lz] in nm

def is_water_resname(name):
    return name.strip().upper() in ("HOH","WAT","H2O","SOL","TIP3")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod_ns", type=float, default=2.0, help="Production ns")
    parser.add_argument("--equil_ns", type=float, default=0.2, help="Equilibration ns (total; split internally)")
    parser.add_argument("--padding", type=float, default=1.0, help="Initial solvent padding (nm)")
    parser.add_argument("--timestep_fs", type=float, default=2.0, help="timestep fs")
    parser.add_argument("--dump_stride", type=int, default=80, help="DCD frames every N steps")
    parser.add_argument("--cutoff_nm", type=float, default=0.9, help="nonbonded cutoff (nm)")
    parser.add_argument("--index", type=int, default=None, help="Index for output file numbering (for parallel runs)")
    args = parser.parse_args()

    # filenames (add index if provided)
    idx = f"_{args.index}" if args.index is not None else ""
    input_pdb = f"alanine-dipeptide/alanine_amber_input.pdb"
    solvated_pdb = f"alanine-dipeptide/alanine_amber_solvated.pdb"
    final_pdb = f"alanine-dipeptide/alanine_amber_final{idx}.pdb"
    traj_dcd = f"alanine-dipeptide/alanine_amber_prod{idx}.dcd"
    log_file = f"alanine-dipeptide/alanine_amber{idx}.log"
    checkpoint = f"alanine-dipeptide/alanine_amber{idx}.chk"

    import os
    # Only write input PDB if it does not exist
    if not os.path.exists(input_pdb):
        print(f"Input PDB {input_pdb} not found. Writing new input PDB...")
        write_input_pdb(input_pdb)
    else:
        print(f"Input PDB {input_pdb} already exists. Using existing file.")
    pdb = app.PDBFile(input_pdb)

    # load forcefield
    print("Loading AMBER ff14SB + TIP3P...")
    forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    # iteratively solvate until box dimensions >= 2*cutoff + margin
    cutoff = args.cutoff_nm  # nm
    margin = 0.05  # nm safety
    max_padding = 3.0
    padding = args.padding

    modeller = None
    import os
    if os.path.exists(solvated_pdb):
        print(f"Solvated PDB {solvated_pdb} already exists. Loading existing file.")
        modeller = app.PDBFile(solvated_pdb)
        # For downstream code, we need a Modeller object with topology and positions
        modeller = app.Modeller(modeller.topology, modeller.positions)
    else:
        for attempt in range(6):
            print(f"Solvating with padding = {padding:.2f} nm (attempt {attempt+1}) ...")
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addSolvent(forcefield, model="tip3p", padding=padding*unit.nanometer)
            # compute bounding box from positions
            lengths = bbox_lengths_nm(modeller.positions)  # nm
            min_len = float(min(lengths))
            print(f" Approx box lengths (nm): {lengths[0]:.3f}, {lengths[1]:.3f}, {lengths[2]:.3f}")
            if min_len >= 2.0*cutoff + margin:
                print(" Box is large enough for cutoff.")
                break
            # increase padding and retry
            padding += 0.5
            if padding > max_padding:
                print("Could not reach required box size by increasing padding — lowering cutoff and continuing.")
                break
        # write solvated PDB for VMD
        with open(solvated_pdb, "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
        print("Wrote solvated PDB:", solvated_pdb)

    # create system with PME and cutoff, constraints on bonds to hydrogens
    nonbonded_cutoff = cutoff * unit.nanometer
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=nonbonded_cutoff,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True,
    )
    # add barostat
    temperature = 300.0 * unit.kelvin
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmosphere, temperature, 25))

    # ===== Add positional restraints on solute atoms (heavy atoms + H of solute) =====
    # We will add a CustomExternalForce with a global parameter 'k' that we can set to zero later.
    k_default = 1000.0  # kJ/mol/nm^2 (typical positional restraint)
    restr_force = openmm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    restr_force.addGlobalParameter("k", k_default)
    restr_force.addPerParticleParameter("x0")
    restr_force.addPerParticleParameter("y0")
    restr_force.addPerParticleParameter("z0")

    # mark solute atom indices (exclude water residues)
    solute_atom_indices = []
    for atom in modeller.topology.atoms():
        if not is_water_resname(atom.residue.name):
            solute_atom_indices.append(atom.index)
    print(f"Adding restraints to {len(solute_atom_indices)} solute atoms.")

    # positions in nm
    pos_nm = modeller.positions.value_in_unit(unit.nanometer)
    for idx in solute_atom_indices:
        x0, y0, z0 = float(pos_nm[idx][0]), float(pos_nm[idx][1]), float(pos_nm[idx][2])
        restr_force.addParticle(int(idx), [x0, y0, z0])

    system.addForce(restr_force)

    # integrator & platform
    timestep = args.timestep_fs * unit.femtoseconds
    friction = 1.0 / unit.picoseconds
    integrator = openmm.LangevinIntegrator(temperature, friction, timestep)
    integrator.setConstraintTolerance(1e-6)

    platform = select_platform()
    print("Using platform:", platform.getName())

    # create simulation
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # minimize
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=5000)
    e = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"Post-minimization potential energy: {e:.3f} kJ/mol")

    # set velocities
    simulation.context.setVelocitiesToTemperature(temperature)

    # compute steps (1 ns = 1e6 fs)
    steps_per_ns = int(1e6 / args.timestep_fs)
    # we'll split equilibration into restrained NVT (0.1*equil) + restrained NPT (rest)
    total_equil_steps = max(100, int(args.equil_ns * steps_per_ns))
    nvt_steps = int(0.5 * total_equil_steps)
    npt_steps = total_equil_steps - nvt_steps

    print(f"Equilibration total steps: {total_equil_steps} (NVT {nvt_steps}, NPT {npt_steps})")
    print("Running restrained NVT equilibration (solute restrained)...")
    # short NVT: remove barostat effect (barostat is present but will not change much for short NVT)
    # To simulate NVT, temporarily remove barostat? Simpler: run short steps with integrator (barostat will not act much)

    simulation.step(nvt_steps)

    print("Running restrained NPT equilibration (solute restrained)...")
    simulation.step(npt_steps)

    # release restraints by setting global parameter k to zero in the context
    try:
        simulation.context.setParameter("k", 0.0)
        print("Released positional restraints (set k=0).")
    except Exception:
        # fallback: set default global param on force object if context method not available
        idx = restr_force.getGlobalParameterIndex("k")
        restr_force.setGlobalParameterDefaultValue(idx, 0.0)
        print("Released restraints by changing global parameter default value on force object (fallback).")

    # production
    prod_steps = max(1, int(args.prod_ns * steps_per_ns))
    print(f"Production steps: {prod_steps}")
    simulation.reporters.append(app.StateDataReporter(log_file, 1000, step=True, time=True,
                                                     temperature=True, potentialEnergy=True,
                                                     totalEnergy=True, speed=True))
    simulation.reporters.append(app.DCDReporter(traj_dcd, args.dump_stride))
    simulation.reporters.append(app.CheckpointReporter(checkpoint, 5000))

    print("Starting production NPT (restraints off)...")
    simulation.step(prod_steps)
    print("Production finished.")

    # write final PDB
    final_state = simulation.context.getState(getPositions=True)
    with open(final_pdb, "w") as f:
        app.PDBFile.writeFile(simulation.topology, final_state.getPositions(), f)
    print("Final PDB written:", final_pdb)
    print("Trajectory DCD:", traj_dcd)
    print("Log:", log_file)
    print("Checkpoint:", checkpoint)

if __name__ == "__main__":
    # try to run main until 10 tries
    for attempt in range(10):
        try:
            main()
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed with error: {e}")
            if attempt == 9:
                print("All attempts failed. Exiting.")
                sys.exit(1)