import numpy as np
from matplotlib import pyplot as plt
import scienceplots
import argparse

plt.style.use('science')
plt.rcParams['text.usetex'] = False

dt = 0.001

def plot_displacement(G, Ps, Pf, sample, timechain, start):
    """
    Plot the longitudinal displacement over time for a sample number of particles, for a given
    point in Peclet number-space.
    
    Arguments:
        G: elongation factor
        Ps: swim Peclet number of interest
        Pf: flow Peclet number of interest
        sample: number of trajectories to consider
        timechain: logscale timechain file
        start: from which trajectory to start sampling
    """
    # Read in data
    filename = f"G {G} results/x trajectories/trajs {Ps} {Pf}.npz"
    data = np.load(filename)['x']

    # Set sample to max number of trajectories if none provided
    if sample is None:
        sample = len(data)

    # Check if number of specified samples exceeds amount in data
    assert sample <= len(data)

    # Extract samples from data
    displacements = data[start:(start + sample)]

    # Generate colours for plotting
    cmap = plt.get_cmap("rainbow")
    samples = np.linspace(0, 1, sample)
    colours = cmap(samples)

    # Generate logscale timechain from file
    tc = np.loadtxt(timechain) * dt

    # Create figure
    fig = plt.figure(figsize=[10, 6])
    plt.title(f"Displacement over time: $G$ = {G}, $Pe_s$ = {Ps}, $Pe_f$ = {Pf}")
    plt.ylabel("$x/w$")
    plt.xlabel(r"$t/\tau_r$")
    plt.axhline(0, linestyle='--', color='black')

    # Plot each displacement over time
    for x, colour in zip(displacements, colours):
        plt.plot(tc, x, color=colour)

    # Show figure
    plt.tight_layout()
    plt.show()

def plot_velocities(G, Ps_params, Pf_params):
    """
    Plot the histogram of instantaneous velocities for all phase points in a list
    of swim and flow Peclet numbers, for a given elongation factor.
    
    Arguments:
        G: elongation factor
        Ps_params: file containing swim Peclet numbers
        Pf_params: file containing flow Peclet numbers
    """
    # Read in swim and flow Peclet numbers from file
    Ps_list = np.loadtxt(Ps_params, dtype=str)
    Pf_list = np.loadtxt(Pf_params, dtype=str)
    sample = len(Ps_list)

    # Generate colours for plotting
    cmap = plt.get_cmap("rainbow")
    samples = np.linspace(0, 1, sample)
    colours = cmap(samples)

    # Set up figure
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"Instantaneous velocity PDF: $G$ = {G}")
    plt.xlabel("$v_x/v_0$")
    plt.ylabel("$P(v_x/v_0)$")

    # Iterate over each phase point to plot
    for Ps, Pf, colour in zip(Ps_list, Pf_list, colours):
        # Read in data
        filename = f"G {G} results/velocities/v_hist {Ps} {Pf}.txt"
        v = np.loadtxt(filename)
        # Construct histogram
        pdf, edges = np.histogram(v, bins='auto', density=True)
        # Estimate mean velocity via mode of distribution
        max_arg = np.argmax(pdf)
        mode_v = np.round((edges[max_arg] + edges[max_arg + 1]) / 2, 3)
        # Plot histogram using stairs
        plt.stairs(pdf, edges, color=colour, label=f"$Pe_s$ = {Ps}, $Pe_f$ = {Pf}, " + r"$\overline{v}_x \approx$ " + f"{mode_v}")

    # Display figure
    plt.legend()
    plt.tight_layout()
    plt.show()


def TD3(G, filename1, filename2, filename3, btd):
    """
    Plot the trapping time distributions of three different phase points on one graph.
    
    Arguments:
        G: elongation factor
        filename1: filename of first dataset
        filename2: filename of second dataset
        filename3: filename of third dataset
        btd: if True, plot bulk time rather than trapping time distributions
    """
    # Select number of bins
    num_bins = 'auto'
    # Select trapping or bulk
    if btd:
        td_text = ["btd", "Bulk"]
    else:
        td_text = ["ttd", "Trapping"]

    # Read parameters and trapping/bulk times from first datafile
    Ps1, string = filename1.split(' ')
    Pf1 = string.split('.txt')[0]
    data1 = np.loadtxt(f"G {G} results/{td_text[0]} data/{filename1}", dtype=float)
    
    # Construct first histogram
    counts1, bins = np.histogram(data1, bins=num_bins, density=True)
    bin_centres1 = (bins[:-1] + bins[1:]) / 2

    # Read parameters and trapping/bulk times from second datafile
    Ps2, string = filename2.split(' ')
    Pf2 = string.split('.txt')[0]
    data2 = np.loadtxt(f"G {G} results/{td_text[0]} data/{filename2}", dtype=float)

    # Construct second histogram
    counts2, bins = np.histogram(data2, bins=num_bins, density=True)
    bin_centres2 = (bins[:-1] + bins[1:]) / 2

    # Read parameters and trapping/bulk times from third datafile
    Ps3, string = filename3.split(' ')
    Pf3 = string.split('.txt')[0]
    data3 = np.loadtxt(f"G {G} results/{td_text[0]} data/{filename3}", dtype=float)

    # Construct third histogram
    counts3, bins = np.histogram(data3, bins=num_bins, density=True)
    bin_centres3 = (bins[:-1] + bins[1:]) / 2

    title = f"{td_text[1]} time distributions: $G$ = {G}"
    label1 = f'$Pe_s$ = {Ps1}, $Pe_f$ = {Pf1}'
    label2 = f'$Pe_s$ = {Ps2}, $Pe_f$ = {Pf2}'
    label3 = f'$Pe_s$ = {Ps3}, $Pe_f$ = {Pf3}'

    # Find minimum and maximum of independent data
    bin_min = np.min([np.min(bin_centres1), np.min(bin_centres2), np.min(bin_centres3)])
    bin_max = np.max([np.max(bin_centres1), np.max(bin_centres2), np.max(bin_centres3)])
    # Generate exponential decay x- and y-axis data
    x = np.linspace(bin_min, bin_max, 50)
    y = np.exp(-x)

    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(title)
    plt.scatter(bin_centres1, counts1, color='green', marker='.', s=10, label=label1)
    plt.scatter(bin_centres2, counts2, color='red', marker='.', s=10, label=label2)
    plt.scatter(bin_centres3, counts3, color='blue', marker='.', s=10, label=label3)
    # Plot exponential decay
    plt.plot(x, y, color='black', label=r'$e^{-t/\tau_r}$')
    plt.xlabel(r"$t/\tau_r$")
    plt.ylabel("probability density")
    plt.yscale('log')
    plt.axvline(1, color='black', linestyle='dotted', label=r'$t=\tau_r$')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', type=str, help="Geometrical elongation factor")
    parser.add_argument('-Ps', type=str, help="Swim Peclet number")
    parser.add_argument('-Pf', type=str, help="flow Peclet number")
    parser.add_argument('--displacement', action='store_true', help="Plot the longitudinal displacement over time")
    parser.add_argument('-s', type=int, default=None, help="Number of trajectories to consider")
    parser.add_argument('-tc', type=str, default='timechain10000000.txt', help="File containing logscale timechain in timesteps")
    parser.add_argument('-start', type=int, default=0, help="Start samples at this trajectory number")
    parser.add_argument('--velocities', action='store_true', help="Plot the histograms of velocity from a list of phase points")
    parser.add_argument('--TTD3', action='store_true', help="Plot the trapping time distributions of three datasets")
    parser.add_argument('--BTD3', action='store_true', help="Plot the bulk time distributions of three datasets")
    parser.add_argument('-PsL', type=str, help='Filepath to swim Peclet number parameter file')
    parser.add_argument('-PfL', type=str, help='Filepath to flow Peclet number parameter file')
    parser.add_argument('-f1', type=str, default=None, help='Filepath to first dataset')
    parser.add_argument('-f2', type=str, default=None, help='Filepath to second dataset, if applicable')
    parser.add_argument('-f3', type=str, default=None, help='Filepath to third dataset, if applicable')
    args = parser.parse_args()

    if args.displacement:
        plot_displacement(args.G, args.Ps, args.Pf, args.s, args.tc, args.start)
    elif args.velocities:
        plot_velocities(args.G, args.PsL, args.PfL)
    elif args.TTD3 or args.BTD3:
        TD3(args.G, args.f1, args.f2, args.f3, args.BTD3)