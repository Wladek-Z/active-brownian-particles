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

def phase_diagram(filename, N, T, dt, D, l1, u1, l2, u2, n, G):
    """
    Collect data for various phase diagrams of the ABP system by varying the ration of both
    Peclet numbers and the ratio of the persistence length and the channel width.

    Arguments:
        filename: file to record the data
        N: number of particles
        T: number of timesteps
        dt: timestep
        D: dimensionless diffusion constant
        l1: lower bound of data along first axis
        u1: upper bound of data along first axis
        l2: lower bound of data along second axis
        u2: upper bound of data along second axis
        n: number of data points
        G: elongation factor
    """
    lp_w_list = np.round(np.linspace(l1, u1, n), 3)
    Pf_Ps_list = np.round(np.linspace(l2, u2, n), 3)

    with open(filename, 'w') as f:
        f.write(f"# G = {G}\n")
        f.write("lp_w,Pf_Ps,alpha,beta,mean_vx\n")

        for Ps in lp_w_list:
            for Pf_Ps in Pf_Ps_list:
                Pf = Pf_Ps * Ps
                abp = ABP(N, T, dt, Ps, D, Pf, G)
                pos, _ = abp.Run()
                # Get MSD scaling exponent
                msd_xy, msd = abp.get_MSD(pos)
                a, _ = abp.get_powerlaw(msd_xy[:, 0])
                # Get mean x-velocity
                vx = abp.get_xspeed(pos)
                vx_independent = vx[::abp.step][10:]
                mean_vx = np.mean(vx_independent)
                # Get variance scaling exponent
                var_x = abp.get_variance(pos[:, :, 0])
                b, _ = abp.get_powerlaw(var_x)
                # Track progress
                print(f"Ps = {Ps}, Pf = {Pf}, alpha = {a}, beta = {b}, mean x-velocity = {mean_vx}")
                f.write(f"{Ps},{Pf_Ps},{a},{b},{mean_vx}\n")

def phase_diagram_x(filename, N, T, dt, D, l1, u1, l2, u2, n, G):
    """
    Collect data for various phase diagrams of the ABP system by varying the ratio of both
    Peclet numbers and the ratio of the persistence length and the channel width.

    Arguments:
        filename: file to record the data
        N: number of particles
        T: number of timesteps
        dt: timestep
        D: dimensionless diffusion constant
        l1: lower bound of data along first axis
        u1: upper bound of data along first axis
        l2: lower bound of data along second axis
        u2: upper bound of data along second axis
        n: number of data points
        G: elongation factor
    """
    lp_w_list = np.round(np.linspace(l1, u1, n), 6)
    Pf_Ps_list = np.round(np.linspace(l2, u2, n), 6)

    with open(filename, 'w') as f:
        f.write("lp_w,Pf_Ps,alpha,alpha_x\n")

        for Ps in lp_w_list:
            for Pf_Ps in Pf_Ps_list:
                Pf = Pf_Ps * Ps
                abp = ABP(N, T, dt, Ps, D, Pf, G)
                r, e = abp.Run()
                msd_xy, msd = abp.get_MSD(r)
                a, b = abp.get_powerlaw(msd)
                msd_x = msd_xy[:, 0]
                a_x, b_x = abp.get_powerlaw(msd_x)
                # Track progress
                print(f"Ps = {Ps}, Pf = {Pf}, alpha = {a}, alphaX = {a_x}")
                f.write(f"{Ps},{Pf_Ps},{a},{a_x}\n")

def collect_histogram(folder, N, T, dt, Ps, w, D, mu, Pf, G, bins):
    """
    Collect data for histograms of the ABP system for large N.
    Automatically write to npz file.

    Arguments:
        folder: folder in which to save the data
        N: number of particles
        T: number of timesteps
        dt: timestep
        Ps: swim Peclet number
        w: width of channel
        D: dimensionless diffusion constant
        mu: dimensionless repulsion strength
        Pf: flow Peclet number
        G: elongation factor
        bins: number of bins for histogram
    """
    # Run the simulation
    abp = ABP(N, T, dt, Ps, D, Pf, G)
    pos, orient = abp.Run()
    # Decorrelate the data
    p = pos[::abp.step][10:-1]
    o = orient[::abp.step][10:-1]
    # Isolate vertical position from position data
    y = p[:, :, 1].flatten() / w
    # Construct positional PDF
    pdf1, edges1 = np.histogram(y, bins=bins)
    # Isolate orientation angles from orientation data
    theta = np.arctan2(o[:, :, 1], o[:, :, 0]).flatten()
    # Construct orientational PDF
    pdf2, edges2 = np.histogram(theta, bins=bins)
    # Find particle orientations for upstream swimmers near the surface
    vx = abp.get_xspeed(pos)
    vx_independent = vx[::abp.step][10:]
    trap = abp.trapping_index(p, vx_independent)
    theta_sfc = theta[trap.flatten()]
    # Construct orientational PDF near the surface
    pdf3, edges3 = np.histogram(theta_sfc, bins=bins)
    # Save data to npz file
    np.savez(f'{folder}/data_{jobid}.npz', pdf1=pdf1, edges1=edges1, pdf2=pdf2, edges2=edges2, pdf3=pdf3, edges3=edges3)

def effective_constants(filename, N, T, dt, Ps, D, G, l, u, n):
    """
    Calculate the effective diffusivity or Peclet number as a function of the
    Peclet number ratio.
    
    Arguments:
        filename: path to save data
        N: number of particles
        T: number of timesteps
        dt: timestep
        Ps: swim Peclet number
        D: dimensionless translational diffusivity
        G: elongation factor
        l: lower bound of data
        u: upper bound of data
        n: number of data points to consider
    """
    nums = np.linspace(l, u, n)
    Pf_list = np.round(nums * Ps, 6)
    # Record parameters
    params = np.array([Ps, D, G])
    # Initialise empty arrays for effective diffusivities/Peclet numbers
    D_eff = np.empty(0)
    P_eff = np.empty(0)
    Pf_D = np.empty(0)
    Pf_P = np.empty(0)
    # Iterate over every flow Peclet number
    for Pf in Pf_list:
        # Construct ABP object
        abp = ABP(N, T, dt, Ps, D, Pf, G)
        # Run simulation
        pos, orient = abp.Run()
        # Get MSDx scaling exponent
        msd_xy, msd = abp.get_MSD(pos)
        a, b = abp.get_powerlaw(msd_xy[:, 0])
        # Save to different arrays depending on power of exponent
        if a < 1.5:
            B = np.round(np.exp(b)/2/d, 6)
            D_eff = np.append(D_eff, B)
            Pf_D = np.append(Pf_D, Pf)
        else:
            B = np.round(np.sqrt(np.exp(b)), 6)    
            P_eff = np.append(P_eff, B)
            Pf_P = np.append(Pf_P, Pf)   
    # Save data to file
    np.savez(filename, p=params, Pf_D=Pf_D, D_eff=D_eff, Pf_P=Pf_P, P_eff=P_eff)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=100, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-T', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-D', type=float, default=0.1, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-f', type=str, default=None, help='Filepath to save data')
    parser.add_argument('-F', type=str, default=None, help='Folder in which to store saved data')
    parser.add_argument('--PD', action='store_true', help='Construct the phase diagram')
    parser.add_argument('--PDX', action='store_true', help='Record MSD scaling exponents, including x-direction')
    parser.add_argument('-l1', type=float, help='Lower bound for data collection (first axis)')
    parser.add_argument('-u1', type=float, help='Upper bound for data collection (first axis)')
    parser.add_argument('-l2', type=float, help='Lower bound for data collection (second axis)')
    parser.add_argument('-u2', type=float, help='Upper bound for data collection (second axis)')
    parser.add_argument('-n', type=int, help='Number of data points')
    parser.add_argument('-G', type=float, default=0, help='Geometrical factor related to particle aspect ratio')
    parser.add_argument('-Ps', type=float, default=5, help='Swim Peclet number')
    parser.add_argument('-Pf', type=float, default=5, help='Flow Peclet number')
    parser.add_argument('--MSD', action='store_true', help='Find the mean square displacement')
    parser.add_argument('--trajectory', action='store_true', help='Find the particle trajectory')
    parser.add_argument('-bins', type=int, default=50, help='Number of bins to use in histogram')
    parser.add_argument('--hist', action='store_true', help='Collect data to construct histograms')
    parser.add_argument('--eff', action='store_true', help='Collect data on effective diffusivities/Peclet numbers along an axis')
    args = parser.parse_args()

    if args.PD:
        phase_diagram(args.f, args.N, args.T, args.dt, args.D, args.l1, args.u1, args.l2, args.u2, args.n, args.G)
    elif args.PDX:
        phase_diagram_x(args.f, args.N, args.T, args.dt, args.D, args.l1, args.u1, args.l2, args.u2, args.n, args.G)
    elif args.hist:
        collect_histogram(args.F, args.N, args.T, args.dt, args.Ps, args.D, args.Pf, args.G, args.bins)
    elif args.eff:
        effective_constants(args.f, args.N, args.T, args.dt, args.Ps, args.D, args.G, args.l1, args.u1, args.n)
