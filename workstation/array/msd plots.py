import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import argparse

plt.style.use('science')
plt.rcParams['text.usetex'] = False

d = 2
D = 0.01

def plot_MSD(G, Ps, Pf, offset):
    """
    Plot the mean square displacement for a given set of parameters.

    Arguments:
        G: elongation factor
        Ps: swim Peclet number
        Pf: flow Peclet number
        offset: number of skipped logscale blocks in data
    """
    # Resolve filepath
    filename = f"G {G} results/MSD o{offset}/{Ps} {Pf}.txt"
    t, msd = np.loadtxt(filename, unpack=True)

    # Obtain theory curves
    msd_theory = 2 * d * D * t + 2 * float(Ps)**2 * t - 2 * float(Ps)**2 * (1 - np.exp(-t))
    # Theoretical msd for ballistic and diffusive regimes
    msd_b = float(Ps)**2 * t**2 + 2 * d * D * t
    msd_d = 2 * t * (d * D + float(Ps)**2)

    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$: $Pe_s$ = {Ps}, $Pe_f$ = {Pf}, $G$ = {G}")
    plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
    plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory (no flow)')
    plt.loglog(t, msd_b, color='blue', linestyle='--', label='ballistic limit')
    plt.loglog(t, msd_d, color='green', linestyle='--', label='diffusive limit')
    plt.axvline(1, color='black', linestyle='dotted', label=r'$t=\tau_r$')
    plt.xlabel(r"$t/\tau_r$")
    plt.ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_all_MSD(G, Ps_list, Pf_list, offset, old):
    """
    Plot all MSDs from a list of swim and flow Peclet numbers on one graph.
    
    Arguments:
        G: elongation factor
        Ps_list: list of swim Peclet numbers
        Pf_list: list of flow Peclet numbers
        offset: skipped logscale blocks
        old: True if MSDs are to be plotted using a single origin
    """
    # Resolve folder path
    if old:
        folder = f"G {G} results/MSD old o{offset}"
    else:
        folder = f"G {G} results/MSD o{offset}"
    # Retrieve Peclet numbers to plot
    Ps = np.loadtxt(Ps_list, dtype=str)
    Pf = np.loadtxt(Pf_list, dtype=str)

    # Generate colours for plotting
    cmap = plt.get_cmap("rainbow")
    samples = np.linspace(0, 1, len(Ps))
    colours = cmap(samples)

    # Set up the figure
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$ comparison: $G$ = {G}, offset = {offset} blocks")
    plt.axvline(1, color='black', linestyle='dotted')
    plt.text(1+1e-1, 1e-5, r'$t=\tau_r$', ha='left', va='bottom', fontsize=12)
    plt.xlabel(r"$t/\tau_r$")
    plt.ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    plt.xscale('log')
    plt.yscale('log')

    # Initialise loop index
    i = -1
    # Iterate over each MSD file
    for ps, pf in zip(Ps, Pf):
        # Increment loop index
        i += 1
        # Read MSD from file
        file = f"{folder}/{ps} {pf}.txt"
        t, msd = np.loadtxt(file, unpack=True)
        # Add MSD to figure
        plt.scatter(t, msd, color=colours[i], marker='.', s=10, label=f'$Pe_s$ = {ps}, $Pe_f$ = {pf}')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_pp_MSD(G, Ps, Pf, sample):
    """
    Plot the individual MSDs over a sample of trajectories. Plot includes both full
    MSD and zoom at long times (t > 10^3).
    
    Arguments:
        G: elongation factor
        Ps: swim Peclet number
        Pf: flow Peclet number
        sample: number of trajectories to plot
    """
    # Resolve folder filepath
    filename = f"G {G} results/MSD ind/{Ps} {Pf} n{sample}.npz"
    # Retrieve MSD data
    data = np.load(filename)
    t = data['time']
    msds = data['MSD']

    # Generate colours for plotting
    cmap = plt.get_cmap("rainbow")
    samples = np.linspace(0, 1, sample)
    colours = cmap(samples)

    # Set up the figure
    fig, ax = plt.subplots(1, 2, figsize=[12, 6])
    fig.suptitle(f"MSD$_x$ (per trajectory): $Pe_s$ = {Ps}, $Pe_f$ = {Pf}, $G$ = {G}")
    ax[0].set_title('full trajectory')
    ax[0].axvline(1, color='black', linestyle='dotted')
    ax[0].text(1+1e-1, 1e-5, r'$t=\tau_r$', ha='left', va='bottom', fontsize=12)
    ax[0].set_xlabel(r"$t/\tau_r$")
    ax[0].set_ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[1].set_title('late-time trajectory')
    ax[1].set_xlabel(r"$t/\tau_r$")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    # Iterate over each MSD, colour
    for msd, colour in zip(msds, colours):
        # Add MSD to figures
        ax[0].scatter(t, msd, color=colour, marker='.', s=10)
        ax[1].scatter(t[t > 1000], msd[t > 1000], color=colour, marker='.', s=10)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', type=str, help="Geometrical elongation factor")
    parser.add_argument('-Ps', type=str, help="Swim Peclet number")
    parser.add_argument('-Pf', type=str, help="flow Peclet number")
    parser.add_argument('-off', type=int, default=0, help="Skipped logscale blocks")
    parser.add_argument('--MSD', action='store_true', help="Plot the MSD")
    parser.add_argument('--ppMSD', action='store_true', help="Plot MSDs for individual trajectories")
    parser.add_argument('--all', action='store_true', help="Plot all MSDs from parameter list")
    parser.add_argument('-PsL', type=str, help="List of swim Peclet numbers")
    parser.add_argument('-PfL', type=str, help="List of flow Peclet numbers")
    parser.add_argument('-s', type=int, help="Number of trajectories to consider")
    parser.add_argument('-old', action='store_true', help="Specify whether to plot single origin MSDs")
    args = parser.parse_args()

    if args.MSD:
        plot_MSD(args.G, args.Ps, args.Pf, args.off)
    elif args.all:
        plot_all_MSD(args.G, args.PsL, args.PfL, args.off, args.old)
    elif args.ppMSD:
        plot_pp_MSD(args.G, args.Ps, args.Pf, args.s)