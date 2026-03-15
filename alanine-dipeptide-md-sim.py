#!/usr/bin/env python3
"""
ala_run.py

Single-file pipeline:
  1. Solvate alanine dipeptide (ACE-ALA-NME) with AMBER ff14SB + TIP3P
  2. Minimize + equilibrate (NVT then NPT, solute restrained)
  3. Production NPT (restraints released)
  4. Unwrap solute trajectory with MDAnalysis and write solute-only DCD

Randomness per run:
  - velocity seed   : --seed (int); default = random
  - solvation seed  : --solv_seed (int); default = random
  - padding jitter  : ±0.1 nm around --padding, seeded by --solv_seed

Usage (single run):
  python3 ala_run.py --index 0 --seed 42 --solv_seed 7

Usage (called by run_ensemble.sh):
  python3 ala_run.py --index $i --seed $SEED --solv_seed $SOLV_SEED
"""

import argparse
import os
import sys
import math
import random
import numpy as np
import time

import openmm
import openmm.app as app
import openmm.unit as unit

# ── embedded PDB ────────────────────────────────────────────────────────────
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

# ── helpers ──────────────────────────────────────────────────────────────────
def write_input_pdb(path):
    with open(path, "w") as f:
        f.write(ALANINE_PDB_TEXT.strip() + "\n")

def select_platform():
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return openmm.Platform.getPlatformByName(name)
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform found.")

def bbox_lengths_nm(positions):
    if hasattr(positions, "value_in_unit"):
        arr = np.asarray(positions.value_in_unit(unit.nanometer))
    else:
        arr = np.asarray(positions)
    return arr.max(axis=0) - arr.min(axis=0)

def is_water(resname):
    return resname.strip().upper() in ("HOH", "WAT", "H2O", "SOL", "TIP3")

# ── MD pipeline ──────────────────────────────────────────────────────────────
def run_md(args, out_dir, seed, solv_seed):
    """Full MD: solvate → minimize → equilibrate → production. Returns solvated_pdb path."""

    os.makedirs(out_dir, exist_ok=True)

    # ── seeding ──────────────────────────────────────────────────────────────
    rng = random.Random(solv_seed)
    np.random.seed(solv_seed)

    # padding jitter: ±0.1 nm around base padding, seeded per run
    padding_jitter = rng.uniform(-0.1, 0.1)
    base_padding   = args.padding + padding_jitter
    print(f"[run {args.index}] velocity seed={seed}  solv_seed={solv_seed}  "
          f"padding={base_padding:.3f} nm")

    # ── file paths (all flat in out_dir, index-suffixed) ─────────────────────
    idx          = args.index
    # input.pdb is shared across runs — no index suffix needed
    input_pdb    = os.path.join(out_dir, "input.pdb")
    solvated_pdb = os.path.join(out_dir, f"solvated_{idx}.pdb")
    final_pdb    = os.path.join(out_dir, f"final_{idx}.pdb")
    traj_dcd     = os.path.join(out_dir, f"prod_{idx}.dcd")
    log_file     = os.path.join(out_dir, f"prod_{idx}.log")
    checkpoint   = os.path.join(out_dir, f"prod_{idx}.chk")

    # ── input PDB ────────────────────────────────────────────────────────────
    if not os.path.exists(input_pdb):
        write_input_pdb(input_pdb)
    pdb = app.PDBFile(input_pdb)

    # ── forcefield ───────────────────────────────────────────────────────────
    print(f"[run {args.index}] Loading AMBER ff14SB + TIP3P ...")
    ff = app.ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")

    # ── solvation (with box-size check) ──────────────────────────────────────
    cutoff = args.cutoff_nm
    margin = 0.05

    if os.path.exists(solvated_pdb):
        print(f"[run {args.index}] Re-using existing {solvated_pdb}")
        tmp = app.PDBFile(solvated_pdb)
        modeller = app.Modeller(tmp.topology, tmp.positions)
    else:
        padding = base_padding
        for attempt in range(6):
            print(f"[run {args.index}] Solvating, padding={padding:.3f} nm (attempt {attempt+1})")
            modeller = app.Modeller(pdb.topology, pdb.positions)
            modeller.addSolvent(ff, model="tip3p", padding=padding * unit.nanometer)
            lengths  = bbox_lengths_nm(modeller.positions)
            min_len  = float(min(lengths))
            print(f"[run {args.index}]  box ~ {lengths[0]:.3f} x {lengths[1]:.3f} x {lengths[2]:.3f} nm")
            if min_len >= 2.0 * cutoff + margin:
                break
            padding += 0.5
            if padding > 3.5:
                print(f"[run {args.index}]  Box still small — continuing anyway.")
                break
        with open(solvated_pdb, "w") as f:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
        print(f"[run {args.index}] Wrote {solvated_pdb}")

    # ── system ───────────────────────────────────────────────────────────────
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod   = app.PME,
        nonbondedCutoff   = cutoff * unit.nanometer,
        constraints       = app.HBonds,
        rigidWater        = True,
        removeCMMotion    = True,
    )
    temperature = 300.0 * unit.kelvin
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmosphere, temperature, 25))

    # ── positional restraints on solute heavy atoms ───────────────────────────
    restr = openmm.CustomExternalForce(
        "0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    restr.addGlobalParameter("k", 1000.0)
    restr.addPerParticleParameter("x0")
    restr.addPerParticleParameter("y0")
    restr.addPerParticleParameter("z0")

    solute_idx = [a.index for a in modeller.topology.atoms()
                  if not is_water(a.residue.name)]
    pos_nm = modeller.positions.value_in_unit(unit.nanometer)
    for i in solute_idx:
        x0, y0, z0 = float(pos_nm[i][0]), float(pos_nm[i][1]), float(pos_nm[i][2])
        restr.addParticle(i, [x0, y0, z0])
    system.addForce(restr)
    print(f"[run {args.index}] Restraints on {len(solute_idx)} solute atoms.")

    # ── integrator & simulation ───────────────────────────────────────────────
    # Timestep fixed at 2 fs (do not change)
    TIMESTEP_FS = 2.0
    timestep    = TIMESTEP_FS * unit.femtoseconds
    integrator  = openmm.LangevinMiddleIntegrator(temperature, 1.0/unit.picoseconds, timestep)
    integrator.setRandomNumberSeed(seed)       # ← per-run velocity seed
    integrator.setConstraintTolerance(1e-6)

    platform   = select_platform()
    print(f"[run {args.index}] Platform: {platform.getName()}")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    # ── minimization ─────────────────────────────────────────────────────────
    print(f"[run {args.index}] Minimizing ...")
    simulation.minimizeEnergy(maxIterations=5000)
    e = (simulation.context
         .getState(getEnergy=True)
         .getPotentialEnergy()
         .value_in_unit(unit.kilojoules_per_mole))
    print(f"[run {args.index}] Post-minimization PE: {e:.1f} kJ/mol")

    # randomise velocities with the per-run seed
    simulation.context.setVelocitiesToTemperature(temperature, seed)

    # ── equilibration ─────────────────────────────────────────────────────────
    steps_per_ns   = int(1e6 / TIMESTEP_FS)          # 500 000 steps/ns
    total_eq_steps = max(100, int(args.equil_ns * steps_per_ns))
    nvt_steps      = total_eq_steps // 2
    npt_steps      = total_eq_steps - nvt_steps

    print(f"[run {args.index}] Equilibrating: NVT {nvt_steps} + NPT {npt_steps} steps (restraints on)")
    simulation.step(nvt_steps)
    simulation.step(npt_steps)

    # release restraints
    simulation.context.setParameter("k", 0.0)
    print(f"[run {args.index}] Restraints released (k=0).")

    # ── production ────────────────────────────────────────────────────────────
    # DCD dump stride fixed at every 80 steps (do not change)
    DUMP_STRIDE = 80
    prod_steps  = max(1, int(args.prod_ns * steps_per_ns))
    print(f"[run {args.index}] Production: {prod_steps} steps, DCD every {DUMP_STRIDE} steps")

    simulation.reporters.append(
        app.StateDataReporter(log_file, 1000, step=True, time=True,
                              temperature=True, potentialEnergy=True,
                              totalEnergy=True, speed=True))
    simulation.reporters.append(app.DCDReporter(traj_dcd, DUMP_STRIDE))
    simulation.reporters.append(app.CheckpointReporter(checkpoint, 5000))

    simulation.step(prod_steps)
    print(f"[run {args.index}] Production done.")

    # state = simulation.context.getState(getPositions=True)
    # with open(final_pdb, "w") as f:
    #     app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)
    # print(f"[run {args.index}] Wrote {final_pdb}")

    return solvated_pdb, traj_dcd


# ── unwrap / center pipeline ─────────────────────────────────────────────────
def run_unwrap(out_dir, idx, solvated_pdb, traj_dcd, center=True,
               solute_sel="not resname TIP3 WAT SOL HOH"):
    """Unwrap solute trajectory with MDAnalysis and write solute-only DCD."""
    import MDAnalysis as mda
    from MDAnalysis.transformations import unwrap, center_in_box

    out_dcd = os.path.join(out_dir, f"unwrapped_{idx}.dcd")
    out_pdb = os.path.join(out_dir, f"unwrapped_{idx}.pdb")

    print(f"[unwrap] Loading {solvated_pdb} + {traj_dcd} ...")
    u = mda.Universe(solvated_pdb, traj_dcd)
    print(f"[unwrap] Atoms: {u.atoms.n_atoms}  Frames: {len(u.trajectory)}")

    try:
        solute = u.select_atoms(solute_sel)
    except Exception:
        solute = u.select_atoms("not resname TIP3 WAT SOL HOH")
    print(f"[unwrap] Solute atoms selected: {solute.n_atoms}")

    transforms = [unwrap(u.atoms)]
    if center:
        transforms.append(center_in_box(solute))
    u.trajectory.add_transformations(*transforms)

    print(f"[unwrap] Writing {out_dcd} ...")
    with mda.Writer(out_dcd, solute.n_atoms) as W:
        for ts in u.trajectory:
            W.write(solute)

    # write final frame PDB
    # with mda.Writer(out_pdb, solute.n_atoms) as P:
        # P.write(solute)

    print(f"[unwrap] Done. Solute DCD: {out_dcd}")
    # print(f"[unwrap]       Final PDB:  {out_pdb}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Alanine dipeptide MD (AMBER ff14SB + TIP3P) + unwrap, single run."
    )
    parser.add_argument("--index",       type=int,   default=0,
                        help="Run index (used for output directory name).")
    parser.add_argument("--outdir",      type=str,   default="alanine-dipeptide",
                        help="Base output directory. Run i → <outdir>/run_<index>/")
    parser.add_argument("--seed",        type=int,   default=None,
                        help="Velocity / integrator RNG seed. Default: random.")
    parser.add_argument("--solv_seed",   type=int,   default=None,
                        help="Solvation / padding-jitter seed. Default: random.")
    parser.add_argument("--prod_ns",     type=float, default=2.0,
                        help="Production length in ns.")
    parser.add_argument("--equil_ns",    type=float, default=0.2,
                        help="Equilibration length in ns.")
    parser.add_argument("--padding",     type=float, default=1.0,
                        help="Base solvent padding in nm (±0.1 nm jitter applied).")
    parser.add_argument("--cutoff_nm",   type=float, default=0.9,
                        help="Nonbonded cutoff in nm.")
    parser.add_argument("--no_unwrap",   action="store_true",
                        help="Skip the MDAnalysis unwrap step.")
    parser.add_argument("--no_center",   action="store_true",
                        help="Skip centering solute in box during unwrap.")
    parser.add_argument("--solute_sel",  type=str,
                        default="not resname TIP3 WAT SOL HOH",
                        help="MDAnalysis selection string for solute.")
    args = parser.parse_args()

    # resolve random seeds
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
    if args.solv_seed is None:
        args.solv_seed = random.randint(0, 2**31 - 1)

    out_dir = args.outdir
    print(f"=== Run {args.index} | dir: {out_dir}/ (index suffix: _{args.index}) ===")
    print(f"    seed={args.seed}  solv_seed={args.solv_seed}")

    # ── MD ───────────────────────────────────────────────────────────────────
    num_md_attempts = 20
    for attempt in range(num_md_attempts):
        try:
            solvated_pdb, traj_dcd = run_md(args, out_dir, args.seed, args.solv_seed)
            break
        except Exception as e:
            print(f"[run {args.index}] MD attempt {attempt+1} failed: {e}")
            if attempt == num_md_attempts - 1:
                print(f"[run {args.index}] All MD attempts failed. Exiting.")
                sys.exit(1)
            time.sleep(1)

    # ── unwrap ───────────────────────────────────────────────────────────────
    if not args.no_unwrap:
        try:
            run_unwrap(
                out_dir, args.index, solvated_pdb, traj_dcd,
                center     = not args.no_center,
                solute_sel = args.solute_sel,
            )
            # delete the original wrapped DCD to save space
            os.remove(traj_dcd)
            os.remove(solvated_pdb)
        except ImportError:
            print("[unwrap] MDAnalysis not found — skipping unwrap step.")
        except Exception as e:
            print(f"[unwrap] Failed: {e}")
    else:
        print("[unwrap] Skipped (--no_unwrap).")

    print(f"=== Run {args.index} complete ===")


if __name__ == "__main__":
    main()