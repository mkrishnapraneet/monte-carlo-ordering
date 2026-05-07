"""
Monte Carlo simulation of alanine dipeptide using OpenMM.

Strategy: Hybrid MC/MD
  - MC moves: random backbone phi/psi dihedral rotations on the solute
  - MD relaxation: short NVT bursts on the full system (solute + water)
    after each accepted move to re-equilibrate water
  - Convergence: Jensen-Shannon divergence of phi/psi 2D histograms
    between consecutive blocks, checked every CONVERGENCE_CHECK_INTERVAL sweeps

Output: DCD trajectory containing ONLY alanine dipeptide coordinates.
The input PDB (solute only, no water) serves as the topology for downstream
analysis.

Usage:
    python mc_alanine_dipeptide.py --pdb alanine_dipeptide.pdb [options]

Dependencies:
    pip install openmm mdanalysis numpy scipy
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

# OpenMM
from openmm import (
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    XmlSerializer,
    unit,
)
from openmm.app import (
    DCDFile,
    ForceField,
    Modeller,
    PDBFile,
    PME,
    Simulation,
    HBonds,
)
from openmm import unit as u

# ─────────────────────────── configuration defaults ───────────────────────────

TEMPERATURE        = 300 * u.kelvin
PRESSURE           = 1.0 * u.bar
PADDING            = 14.0 * u.angstrom          # water box padding (must be > cutoff)
NONBONDED_CUTOFF   = 10.0 * u.angstrom

# MC settings
MC_MAX_SWEEPS               = 1000_000            # hard cap
MC_DIHEDRAL_STEP_SIZE       = 15.0              # degrees, max random rotation
MD_RELAXATION_STEPS         = 50                # MD steps after accepted MC move
MD_TIMESTEP                 = 2.0 * u.femtosecond
MD_FRICTION                 = 1.0 / u.picosecond

# Output
DCD_WRITE_INTERVAL          = 10                # sweeps between DCD frames

# Convergence
CONVERGENCE_CHECK_INTERVAL  = 1000              # sweeps between checks
CONVERGENCE_JS_THRESHOLD    = 0.01             # JS divergence threshold
CONVERGENCE_MIN_SWEEPS      = 10_000           # minimum sweeps before checking
RAMACHANDRAN_BINS           = 36               # bins per axis (360/36 = 10° bins)

# Minimisation
MINIMISATION_STEPS          = 1000

# ─────────────────────────── logging setup ────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────── helpers ─────────────────────────────────────────

def get_platform():
    """Pick the fastest available OpenMM platform."""
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            p = Platform.getPlatformByName(name)
            log.info(f"Using OpenMM platform: {name}")
            return p
        except Exception:
            continue
    return None


def build_system(pdb_path: str, padding=PADDING):
    """
    Load the solute PDB, solvate with TIP3P, apply AMBER ff14SB,
    and return (simulation, modeller, solute_indices).
    """
    log.info(f"Loading PDB: {pdb_path}")
    pdb = PDBFile(pdb_path)

    ff = ForceField("amber14-all.xml", "amber14/tip3p.xml")

    # Solvate
    log.info(f"Solvating with TIP3P (padding = {padding}) ...")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(ff, padding=padding, model="tip3p")
    log.info(
        f"System has {modeller.topology.getNumAtoms()} atoms "
        f"({pdb.topology.getNumAtoms()} solute)"
    )

    # Sanity check: box must be > 2 * cutoff in each dimension
    box = modeller.topology.getPeriodicBoxVectors()
    cutoff_nm = NONBONDED_CUTOFF.value_in_unit(u.nanometer)
    min_box = min(box[i][i].value_in_unit(u.nanometer) for i in range(3))
    if min_box < 2 * cutoff_nm + 0.5:   # 0.5 nm safety margin
        needed = (2 * cutoff_nm + 0.5) * u.nanometer
        new_padding = padding + 0.2 * u.nanometer
        log.warning(
            f"Box too small ({min_box:.2f} nm) for cutoff ({cutoff_nm:.2f} nm). "
            f"Retrying with padding = {new_padding} ..."
        )
        return build_system(pdb_path, padding=new_padding)

    # Create OpenMM system
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=NONBONDED_CUTOFF,
        constraints=HBonds,
        rigidWater=True,
    )

    # Barostat for NPT equilibration (removed during production MC)
    # Use a low update frequency (50 steps) to avoid aggressive box resizing
    system.addForce(MonteCarloBarostat(PRESSURE, TEMPERATURE, 50))

    integrator = LangevinMiddleIntegrator(TEMPERATURE, MD_FRICTION, MD_TIMESTEP)
    platform = get_platform()

    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Identify solute atom indices (those present in the original PDB)
    n_solute = pdb.topology.getNumAtoms()
    solute_indices = list(range(n_solute))
    log.info(f"Solute atom indices: 0 – {n_solute - 1}")

    return sim, modeller, solute_indices, pdb


def minimise(sim: Simulation):
    log.info("Minimising energy ...")
    sim.minimizeEnergy(maxIterations=MINIMISATION_STEPS)
    log.info("Minimisation done.")


def equilibrate_npt(sim: Simulation, steps: int = 50_000):
    """Short NPT MD to equilibrate box size and water."""
    log.info(f"NPT equilibration for {steps} steps ...")
    sim.context.setVelocitiesToTemperature(TEMPERATURE)
    sim.step(steps)
    log.info("Equilibration done.")


def remove_barostat(sim: Simulation):
    """Remove the Monte Carlo barostat before production."""
    from openmm import MonteCarloBarostat as _MCB
    system = sim.context.getSystem()
    for i in range(system.getNumForces() - 1, -1, -1):
        if isinstance(system.getForce(i), _MCB):
            system.removeForce(i)
            log.info("Barostat removed for production NVT MC.")
            break
    # Re-initialise context with updated system
    sim.context.reinitialize(preserveState=True)


# ─────────────── dihedral angle utilities ────────────────────────────────────

def _dihedral_indices_from_topology(topology):
    """
    Find the phi and psi backbone dihedral atom indices for alanine dipeptide.

    Alanine dipeptide (ACE-ALA-NME) canonical dihedrals:
      phi : C(ACE) – N(ALA) – CA(ALA) – C(ALA)
      psi : N(ALA) – CA(ALA) – C(ALA) – N(NME)

    We identify atoms by residue name and atom name, which is robust to
    different PDB atom orderings.
    """
    atoms = {(a.residue.name, a.name): a.index for a in topology.atoms()}

    def _get(*keys):
        for k in keys:
            if k in atoms:
                return atoms[k]
        raise KeyError(f"Could not find atom matching any of {keys} in topology.\n"
                       f"Available: {list(atoms.keys())}")

    # phi: C(ACE)-N(ALA)-CA(ALA)-C(ALA)
    phi_indices = [
        _get(("ACE", "C")),
        _get(("ALA", "N")),
        _get(("ALA", "CA")),
        _get(("ALA", "C")),
    ]
    # psi: N(ALA)-CA(ALA)-C(ALA)-N(NME)
    psi_indices = [
        _get(("ALA", "N")),
        _get(("ALA", "CA")),
        _get(("ALA", "C")),
        _get(("NME", "N"), ("NAC", "N")),
    ]

    log.info(f"phi atom indices: {phi_indices}")
    log.info(f"psi atom indices: {psi_indices}")
    return phi_indices, psi_indices


def _calc_dihedral(positions, indices):
    """Calculate dihedral angle (radians) from four atom positions."""
    p = [np.array(positions[i].value_in_unit(u.nanometer)) for i in indices]
    b1 = p[1] - p[0]
    b2 = p[2] - p[1]
    b3 = p[3] - p[2]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return np.arctan2(y, x)


def _rotate_dihedral(positions, indices, delta_rad, topology=None):
    """
    Rotate all atoms on the 'far side' of bond (indices[1]-indices[2])
    by delta_rad around that bond axis.
    Returns new positions array.
    """
    pos = positions.value_in_unit(u.nanometer)
    i1, i2, i3, i4 = indices

    # Rotation axis: bond between atoms 1 and 2 (0-indexed as indices[1] and indices[2])
    axis = pos[i3] - pos[i2]
    axis = axis / np.linalg.norm(axis)
    origin = pos[i2]

    # Rotation matrix via Rodrigues' formula
    c, s = np.cos(delta_rad), np.sin(delta_rad)
    K = np.array([
        [0,       -axis[2],  axis[1]],
        [axis[2],  0,       -axis[0]],
        [-axis[1], axis[0],  0      ],
    ])
    R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)

    # Determine which atoms to rotate: those bonded to i3 side
    # Simple approach: rotate i4 and all atoms on its side of the bond.
    # For alanine dipeptide we build the "far side" set via BFS on the bond graph.
    if topology is not None:
        bonds = {a.index: set() for a in topology.atoms()}
        for b in topology.bonds():
            bonds[b[0].index].add(b[1].index)
            bonds[b[1].index].add(b[0].index)

        # BFS from i3, not crossing back through i2
        visited = set()
        queue = [i3]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            for nb in bonds[node]:
                if nb != i2 and nb not in visited:
                    queue.append(nb)
        rotate_set = visited
    else:
        # Fallback: just rotate i4
        rotate_set = {i4}

    new_pos = pos.copy()
    for idx in rotate_set:
        v = pos[idx] - origin
        new_pos[idx] = origin + R @ v

    return new_pos * u.nanometer


# ─────────────── Ramachandran histogram utilities ─────────────────────────────

def _phi_psi_to_hist(phi_list, psi_list, bins=RAMACHANDRAN_BINS):
    """Return normalised 2D histogram of (phi, psi) in degrees."""
    edges = np.linspace(-180, 180, bins + 1)
    h, _, _ = np.histogram2d(
        np.degrees(phi_list), np.degrees(psi_list),
        bins=[edges, edges]
    )
    h = h + 1e-10  # avoid zeros
    return h / h.sum()


def _js_divergence(h1, h2):
    """Jensen-Shannon divergence between two flat histograms."""
    return jensenshannon(h1.ravel(), h2.ravel()) ** 2


# ─────────────── main MC loop ─────────────────────────────────────────────────

def run_mc(sim: Simulation, topology, solute_indices, output_dcd: str,
           phi_indices, psi_indices, rng: np.random.Generator):
    """
    Main hybrid MC/MD loop.

    Each sweep:
      1. Propose a random phi OR psi rotation.
      2. Compute energy before/after.
      3. Accept/reject via Metropolis.
      4. If accepted, run MD_RELAXATION_STEPS of MD to relax water.
      5. Write solute coords to DCD every DCD_WRITE_INTERVAL sweeps.
      6. Check convergence every CONVERGENCE_CHECK_INTERVAL sweeps.
    """
    kT = (u.BOLTZMANN_CONSTANT_kB * TEMPERATURE * u.AVOGADRO_CONSTANT_NA).value_in_unit(
        u.kilojoule_per_mole
    )

    log.info("Starting hybrid MC/MD production run ...")
    log.info(f"  Max sweeps        : {MC_MAX_SWEEPS:,}")
    log.info(f"  Dihedral step     : ±{MC_DIHEDRAL_STEP_SIZE}°")
    log.info(f"  MD relaxation     : {MD_RELAXATION_STEPS} steps / accepted move")
    log.info(f"  DCD write interval: every {DCD_WRITE_INTERVAL} sweeps")
    log.info(f"  Convergence check : every {CONVERGENCE_CHECK_INTERVAL} sweeps "
             f"(JS threshold = {CONVERGENCE_JS_THRESHOLD})")

    n_accepted = 0
    phi_traj, psi_traj = [], []

    # Block storage for convergence
    block_phis, block_psis = [], []
    prev_hist = None
    converged = False
    converged_at = None

    t0 = time.time()

    # Build a solute-only topology for the DCD writer so atom count matches
    from openmm.app import Topology as OMTopo
    solute_set = set(solute_indices)
    solute_topology = OMTopo()
    atom_map = {}
    for chain in topology.chains():
        new_chain = solute_topology.addChain(chain.id)
        for res in chain.residues():
            atoms_in_res = [a for a in res.atoms() if a.index in solute_set]
            if not atoms_in_res:
                continue
            new_res = solute_topology.addResidue(res.name, new_chain, res.id)
            for a in atoms_in_res:
                new_atom = solute_topology.addAtom(a.name, a.element, new_res)
                atom_map[a.index] = new_atom
    for b in topology.bonds():
        if b[0].index in solute_set and b[1].index in solute_set:
            solute_topology.addBond(atom_map[b[0].index], atom_map[b[1].index])

    with open(output_dcd, "wb") as dcd_handle:
        dcd_writer = DCDFile(dcd_handle, solute_topology, MD_TIMESTEP)

        for sweep in range(1, MC_MAX_SWEEPS + 1):

            # ── 1. get current state ──────────────────────────────────────────
            state = sim.context.getState(getPositions=True, getEnergy=True)
            positions = state.getPositions()
            e_old = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)

            # ── 2. propose dihedral move ──────────────────────────────────────
            move_phi = rng.integers(0, 2) == 0   # True → phi, False → psi
            dihedral_idx = phi_indices if move_phi else psi_indices
            delta = np.radians(rng.uniform(-MC_DIHEDRAL_STEP_SIZE, MC_DIHEDRAL_STEP_SIZE))
            new_positions = _rotate_dihedral(positions, dihedral_idx, delta, topology)

            # ── 3. evaluate energy of proposed state ──────────────────────────
            sim.context.setPositions(new_positions)
            e_new = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                u.kilojoule_per_mole
            )

            # ── 4. Metropolis acceptance ──────────────────────────────────────
            delta_e = e_new - e_old
            accept = delta_e < 0 or rng.random() < np.exp(-delta_e / kT)

            if accept:
                n_accepted += 1
                # MD relaxation of water (and solute) around new conformation
                sim.step(MD_RELAXATION_STEPS)
            else:
                # Restore old positions
                sim.context.setPositions(positions)

            # ── 5. record phi/psi ─────────────────────────────────────────────
            cur_state = sim.context.getState(getPositions=True)
            cur_pos = cur_state.getPositions()
            phi = _calc_dihedral(cur_pos, phi_indices)
            psi = _calc_dihedral(cur_pos, psi_indices)
            phi_traj.append(phi)
            psi_traj.append(psi)
            block_phis.append(phi)
            block_psis.append(psi)

            # ── 6. write DCD frame ────────────────────────────────────────────
            if sweep % DCD_WRITE_INTERVAL == 0:
                all_pos_nm = cur_pos.value_in_unit(u.nanometer)
                solute_pos_nm = np.array([all_pos_nm[i] for i in solute_indices])
                solute_pos_quantity = solute_pos_nm * u.nanometer
                dcd_writer.writeModel(
                    solute_pos_quantity,
                    periodicBoxVectors=cur_state.getPeriodicBoxVectors()
                )

            # ── 7. convergence check ──────────────────────────────────────────
            if sweep % CONVERGENCE_CHECK_INTERVAL == 0:
                acc_rate = n_accepted / sweep
                elapsed = time.time() - t0
                log.info(
                    f"Sweep {sweep:>8,} | acc rate {acc_rate:.3f} | "
                    f"elapsed {elapsed:.1f}s"
                )

                if sweep >= CONVERGENCE_MIN_SWEEPS and len(block_phis) >= CONVERGENCE_CHECK_INTERVAL:
                    curr_hist = _phi_psi_to_hist(block_phis, block_psis)
                    if prev_hist is not None:
                        jsd = _js_divergence(prev_hist, curr_hist)
                        log.info(f"  JS divergence (block vs prev block): {jsd:.5f} "
                                 f"(threshold {CONVERGENCE_JS_THRESHOLD})")
                        if jsd < CONVERGENCE_JS_THRESHOLD:
                            log.info(f"  ✓ Converged at sweep {sweep:,}!")
                            converged = True
                            converged_at = sweep
                    prev_hist = curr_hist
                    block_phis, block_psis = [], []

            if converged:
                break

    total = sweep
    acc_rate = n_accepted / total
    elapsed = time.time() - t0
    log.info("─" * 60)
    log.info(f"MC run finished.")
    log.info(f"  Total sweeps   : {total:,}")
    log.info(f"  Accepted moves : {n_accepted:,} ({acc_rate:.3f})")
    log.info(f"  Converged      : {converged} " + (f"at sweep {converged_at:,}" if converged else "(hit max cap)"))
    log.info(f"  Elapsed time   : {elapsed:.1f}s")
    log.info(f"  DCD output     : {output_dcd}")

    return np.array(phi_traj), np.array(psi_traj), converged


# ─────────────── entry point ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid MC/MD simulation of alanine dipeptide.")
    p.add_argument("--pdb", required=True, help="Input PDB file (solute only, no water).")
    p.add_argument("--output-dcd", default="ala_dipeptide_mc.dcd",
                   help="Output DCD trajectory (solute only). Default: ala_dipeptide_mc.dcd")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--equil-steps", type=int, default=50_000,
                   help="NPT equilibration MD steps. Default: 50000")
    p.add_argument("--max-sweeps", type=int, default=MC_MAX_SWEEPS,
                   help=f"Max MC sweeps. Default: {MC_MAX_SWEEPS:,}")
    p.add_argument("--step-size", type=float, default=MC_DIHEDRAL_STEP_SIZE,
                   help=f"Max dihedral move size (degrees). Default: {MC_DIHEDRAL_STEP_SIZE}")
    p.add_argument("--relaxation-steps", type=int, default=MD_RELAXATION_STEPS,
                   help=f"MD steps per accepted move. Default: {MD_RELAXATION_STEPS}")
    return p.parse_args()


def main():
    args = parse_args()

    # Override globals with CLI args
    global MC_MAX_SWEEPS, MC_DIHEDRAL_STEP_SIZE, MD_RELAXATION_STEPS
    MC_MAX_SWEEPS = args.max_sweeps
    MC_DIHEDRAL_STEP_SIZE = args.step_size
    MD_RELAXATION_STEPS = args.relaxation_steps

    rng = np.random.default_rng(args.seed)

    # ── Build solvated system ─────────────────────────────────────────────────
    sim, modeller, solute_indices, orig_pdb = build_system(args.pdb)

    # ── Minimise ──────────────────────────────────────────────────────────────
    minimise(sim)

    # ── NPT equilibration ─────────────────────────────────────────────────────
    equilibrate_npt(sim, steps=args.equil_steps)

    # ── Switch to NVT for production (remove barostat) ────────────────────────
    remove_barostat(sim)

    # ── Find dihedral indices using the FULL (solvated) topology ─────────────
    phi_indices, psi_indices = _dihedral_indices_from_topology(modeller.topology)

    # ── Run MC ────────────────────────────────────────────────────────────────
    phi_traj, psi_traj, converged = run_mc(
        sim, modeller.topology, solute_indices,
        args.output_dcd, phi_indices, psi_indices, rng
    )

    # ── Save phi/psi trajectory as numpy arrays ───────────────────────────────
    npz_path = Path(args.output_dcd).with_suffix(".npz")
    np.savez(npz_path, phi=phi_traj, psi=psi_traj)
    log.info(f"  phi/psi saved  : {npz_path}")

    if not converged:
        log.warning("Simulation hit the max sweep cap without meeting the convergence criterion.")
        log.warning("Consider increasing --max-sweeps or relaxing the JS threshold.")

    log.info("Done.")


if __name__ == "__main__":
    main()