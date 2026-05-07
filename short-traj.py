import openmm.app as app
import openmm
import openmm.unit as unit
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import RDKitToolkitWrapper
from openmmforcefields.generators import GAFFTemplateGenerator
import mdtraj as md
import os

# take dump frequency from command line argument
import sys
if len(sys.argv) > 1:
    try:
        dump_freq = int(sys.argv[1])
        output_dcd = str(sys.argv[2])
    except ValueError:
        print("Invalid dump frequency provided. Using default value of 1.")
        dump_freq = 1
else:
    print("No dump frequency provided. Using default value of 1.")

# === 1. Generate high-energy pentane conformation ===
smiles = "CCCCC"
off_molecule = Molecule.from_smiles(smiles)
off_molecule.generate_conformers(n_conformers=1, toolkit_registry=RDKitToolkitWrapper())
rdkit_mol = off_molecule.to_rdkit()
AllChem.EmbedMolecule(rdkit_mol)
AllChem.UFFOptimizeMolecule(rdkit_mol)

# Set specific dihedral angles to high-energy conformation
conf = rdkit_mol.GetConformer()
SetDihedralDeg(conf, 0, 1, 2, 3, -180.0)    # Dihedral 1
SetDihedralDeg(conf, 1, 2, 3, 4, -180.0)  # Dihedral 2

# Convert back to OpenFF molecule and save PDB
off_molecule = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
pdb_filename = "pentane_generated.pdb"
off_molecule.to_file(pdb_filename, "PDB")

# === 2. Load PDB and setup system ===
pdb = app.PDBFile(pdb_filename)

gaff = GAFFTemplateGenerator(molecules=[off_molecule])
forcefield = app.ForceField("gaff.xml")
forcefield.registerTemplateGenerator(gaff.generator)

modeller = app.Modeller(pdb.topology, pdb.positions)
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.NoCutoff,
    constraints=app.AllBonds
)

# === 3. Create Langevin Integrator for MD ===
temperature = 300 * unit.kelvin
friction = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
integrator = openmm.LangevinIntegrator(temperature, friction, timestep)

# === 4. Simulation Setup ===
platform = openmm.Platform.getPlatformByName("CUDA")  # or "CPU" if no GPU
simulation = app.Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# Energy minimization
print("Minimizing energy...")
simulation.minimizeEnergy()

# Optional: get minimized energy
state = simulation.context.getState(getEnergy=True)
min_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
print(f"Minimized energy: {min_energy:.3f} kJ/mol")

# === 5. Run MD ===
# n_steps = 500000  # 1 ns with 2 fs timestep
n_steps = dump_freq * 10000
print(f"Running MD for {n_steps} integration steps...")
# output_dcd = f"multiple-short-trajectory-{dump_freq}.dcd"
output_dcd += f"-freq-{dump_freq}.dcd"

# Add reporters
simulation.reporters.append(app.StateDataReporter(
    f"short-log-{dump_freq}.txt", 1000,
    step=True, temperature=True, potentialEnergy=True,
    totalEnergy=True, speed=True
))
simulation.reporters.append(app.DCDReporter(output_dcd, dump_freq))

print("Running MD...")
simulation.step(n_steps)
print("MD complete. Trajectory saved.")
