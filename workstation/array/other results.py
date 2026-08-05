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
        # Plot histogram using stairs
        plt.stairs(pdf, edges, color=colour, label=f"$Pe_s$ = {Ps}, $Pe_f$ = {Pf}")

    # Display figure
    plt.legend(loc='upper left')
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
    parser.add_argument('--velocity', action='store_true', help="Plot the histograms of velocity from a list of phase points")
    parser.add_argument('-PsL', type=str, help='Filepath to swim Peclet number parameter file')
    parser.add_argument('-PfL', type=str, help='Filepath to flow Peclet number parameter file')
    args = parser.parse_args()

    if args.displacement:
        plot_displacement(args.G, args.Ps, args.Pf, args.s, args.tc, args.start)
    elif args.velocity:
        plot_velocities(args.G, args.PsL, args.PfL)