import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import argparse

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def plot_single_MSD(G, Ps, Pf, offset):
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

    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$: $Pe_s$ = {Ps}, $Pe_f$ = {Pf}, $G$ = {G}")
    plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
    plt.axvline(1, color='black', linestyle='dotted', label=r'$t=\tau_r$')
    plt.xlabel(r"$t/\tau_r$")
    plt.ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

def plot_all_MSD(G, Ps_list, Pf_list, offset):
    """
    Plot all MSDs from a list of swim and flow Peclet numbers on one graph.
    
    Arguments:
        G: elongation factor
        Ps_list: list of swim Peclet numbers
        Pf_list: list of flow Peclet numbers
        offset: skipped logscale blocks
    """
    # Resolve folder filepath
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


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', type=str, help="Geometrical elongation factor")
    parser.add_argument('-Ps', type=str, help="Swim Peclet number")
    parser.add_argument('-Pf', type=str, help="flow Peclet number")
    parser.add_argument('-off', type=int, default=0, help="Skipped logscale blocks")
    parser.add_argument('--single', action='store_true', help="Plot single MSD")
    parser.add_argument('--all', action='store_true', help="Plot all MSDs from parameter list")
    parser.add_argument('-PsL', type=str, help="List of swim Peclet numbers")
    parser.add_argument('-PfL', type=str, help="List of flow Peclet numbers")
    args = parser.parse_args()

    if args.single:
        plot_single_MSD(args.G, args.Ps, args.Pf, args.off)
    elif args.all:
        plot_all_MSD(args.G, args.PsL, args.PfL, args.off)