import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from abp import ABP
import os

jobid = os.environ.get("SLURM_JOB_ID", "local")

# Define dimensionality of the system
d = 2


def phase_diagram_single(filename, N, T, dt, D, G, Ps, Pf):
    """
    Collect data for a single point on the phase diagram of the ABP system, 
    in the swim Peclet-flow Peclet number plane. Save to file.

    Arguments:
        filename: filepath to save the data
        N: number of particles
        T: number of timesteps
        dt: timestep
        D: dimensionless diffusion constant
        G: elongation factor
        Ps: swim Peclet number
        Pf: flow Peclet number
    """
    abp = ABP(N, T, dt, Ps, D, Pf, G)
    pos, _ = abp.Run()
    # Get MSD scaling exponent
    msd_xy, msd = abp.get_MSD(pos)
    a, _ = abp.get_powerlaw(msd_xy[:, 0])
    # Get x-velocity and decorrelate
    vx = abp.get_xspeed(pos)
    vx_independent = vx[::abp.step][10:]
    # Take average and divide by Ps to get mean velocity in units of v0
    mean_vx = np.mean(vx_independent) / Ps
    # Get variance scaling exponent
    var_x = abp.get_variance(pos[:, :, 0])
    b, c = abp.get_powerlaw(var_x)
    # Calculate effective diffusivity in the x-direction
    D_eff = np.exp(c) / 2
    # Check if Peclet numbers are in region of interest
    if (G == 1 and (Pf == (2 * Ps - 1))):
        # Save 100 square displacements to file
        x100 = pos[:, :100, 0]
        sd100 = (x100[1:] - x100[0])**2
        np.savetxt(f'SDs/{Ps}_{Pf}.npz', sd100)
    # Save to file
    with open(filename, 'w') as f:
        f.write(f"{Ps},{Pf},{a},{b},{D_eff},{mean_vx}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1000, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-T', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-D', type=float, default=0.01, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-f', type=str, default=None, help='Filepath to save data')
    parser.add_argument('-F', type=str, default=None, help='Folder in which to store saved data')
    parser.add_argument('--PD1', action='store_true', help='Collect a single data point for the phase diagram')
    parser.add_argument('-G', type=float, default=0, help='Geometrical factor related to particle aspect ratio')
    parser.add_argument('-Ps', type=float, default=5, help='Swim Peclet number')
    parser.add_argument('-Pf', type=float, default=5, help='Flow Peclet number')
    args = parser.parse_args()

    if args.PD1:
        phase_diagram_single(args.f, args.N, args.T, args.dt, args.D, args.G, args.Ps, args.Pf)