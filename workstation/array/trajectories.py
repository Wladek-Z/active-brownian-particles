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
    args = parser.parse_args()

    if args.displacement:
        plot_displacement(args.G, args.Ps, args.Pf, args.s, args.tc, args.start)