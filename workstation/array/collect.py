import numpy as np
from pathlib import Path
import argparse

def collect_mean_vx(folder, Ps, Pf, tc, dt):
    """
    Calculate the mean instantaneous longitudinal velocity for a given combination of
    Peclet numbers by taking the velocity at each pair of consecutive timesteps, separated 
    by logscale blocks. Save result to file.
    
    Arguments:
        folder: directory containing the logscaled trajectories for each (Ps, Pf)
        Ps: swim Peclet number
        tc: logscale timechain
        dt: simulation timestep

    Returns:
        mean longitudinal velocity
    """
    # Get filepath to directory containing data regarding specified Ps, Pf
    subfolder = f"{folder}/{Ps} {Pf}"
    # Find the second measurement of each logscale block (+1 to skip header)
    measurements = np.arange(1, len(tc)-1)[np.diff((tc[1:] - tc[:-1]) == 1)][::2] + 1
    # Initialise sum of x-velocities, particle counter
    sum_vx = 0.0
    particles = 0

    # Iterate over each particle trajectory in directory
    for file in Path(subfolder).glob("*.txt"):
        # Increment number of particles
        particles += 1
        with open(file, "r") as f:
            # Initialise loop index
            i = -1
            # Calculate velocity once per logscale block
            for line in f:
                # Increment loop index
                i += 1
                if (i + 1) in measurements:
                    x_prev = np.fromstring(line, sep=',')[0]
                elif i in measurements:
                    x = np.fromstring(line, sep=',')[0]
                    sum_vx += (x - x_prev) / dt

    # Calculate mean velocity in terms of Ps
    mean_vx = sum_vx / len(measurements) / particles / Ps   

    # Save result to file
    output = f"{folder} results/mean velocities/{Ps} {Pf}.txt"     
    with open(output, "w") as f:
        f.write(f"{Ps} {Pf} {mean_vx}")

def get_mean_vx2(folder, Ps, tc, dt):
    """
    Calculate the mean longitudinal velocity by dividing the difference between the
    first and last x-position by the total time elapsed between them.
    
    Arguments:
        folder: directory containing the raw data for each trajectory
        Ps: swim Peclet number
        tc: logscale timechain
        dt: simulation timestep

    Returns:
        mean longitudinal velocity
    """
    # Calculate elapsed time
    t_elapsed = (tc[-1] - tc[0]) * dt
    # Initialise sum of x-velocities, particle counter
    sum_vx = 0.0
    counter = 0

    # Iterate over each trajectory
    for file in Path(folder).glob("*.txt"):
        x_data = np.loadtxt(file, skiprows=1, usecols=1)
        sum_vx += (x_data[-1] - x_data[0]) / t_elapsed
        counter += 1

    # Calculate and return mean velocity by Ps
    return sum_vx / counter / Ps

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', type=str, default=None, help='Directory containing data for each phase point')
    parser.add_argument('--VX', action='store_true', help='Collect mean velocity data for a phase point')
    parser.add_argument('-Ps', type=str, help='Filepath to swim Peclet number parameter file')
    parser.add_argument('-Pf', type=str, help='Filepath to flow Peclet number parameter file')
    parser.add_argument('-tc', type=str, help='Filepath to the logscale timechain file')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-o', type=str, help='Name of output file')
    args = parser.parse_args()

    if args.VX:
        collect_mean_vx(args.F, args.Ps, args.Pf, args.tc, args.dt)
