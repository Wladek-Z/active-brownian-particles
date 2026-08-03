import numpy as np
from pathlib import Path
import argparse

def collect_mean_vx(input, output, Ps, timechain, dt):
    """
    Calculate the mean instantaneous longitudinal velocity for a given combination of
    Peclet numbers by taking the velocity at each pair of consecutive timesteps, separated 
    by logscale blocks. Save result to file.
    
    Arguments:
        input: directory containing raw trajectories
        output: file to store results
        Ps: swim Peclet number
        timechain: logscale timechain file
        dt: simulation timestep
    """
    # Read in timechain from file
    tc = np.loadtxt(timechain, dtype=np.int64)
    # Find the second measurement of each logscale block (+1 to skip header)
    measurements = np.arange(1, len(tc)-1)[np.diff((tc[1:] - tc[:-1]) == 1)][::2] + 1
    # Initialise sum of x-velocities, particle counter
    sum_vx = 0.0
    particles = 0

    # Iterate over each particle trajectory in directory
    for file in Path(input).glob("*.txt"):
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
    with open(output, "w") as f:
        f.write(f"{mean_vx}")

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

def mean_vx_to_file(input, output, Ps_params, Pf_params):
    """
    Collect mean velocities from results folder into a single file sorted into a
    phase diagram-readable format.
    
    Arguments:
        input: directory containing the individual mean velocity files
        output: file in which to store collected mean velocity data
        Ps_params: list of swim Peclet numbers sorted into phase diagram columns
        Pf_params: list of flow Peclet numbers sorted into phase diagram rows
    """
    # Write file header
    with open(output, "w") as f:
        f.write("# Ps Pf mean_vx\n")

    # Read in swim/flow Peclet number list
    Ps_list = np.loadtxt(Ps_params, dtype=str)
    Pf_list = np.loadtxt(Pf_params, dtype=str)

    # Iterate over Peclet number parameters
    for Ps, Pf in zip(Ps_list, Pf_list):
        # Resolve filepath
        filename = f"{input}/{Ps} {Pf}.txt"
        # Extract mean velocity
        vx = np.loadtxt(filename, dtype=float)
        # Append to output file
        with open(output, "a") as f:
            f.write(f"{Ps} {Pf} {vx}\n")
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default=None, help='Directory containing input data')
    parser.add_argument('--VX', action='store_true', help='Collect mean velocity data for a phase point')
    parser.add_argument('--VXf', action='store_true', help='Collect mean velocity data into a single file')
    parser.add_argument('-Ps', type=float, help='swim Peclet number')
    parser.add_argument('-Pf', type=float, help='flow Peclet number parameter')
    parser.add_argument('-PsL', type=str, help='Filepath to swim Peclet number parameter file')
    parser.add_argument('-PfL', type=str, help='Filepath to flow Peclet number parameter file')
    parser.add_argument('-tc', type=str, help='Filepath to the logscale timechain file')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-o', type=str, help='Name of output file')
    args = parser.parse_args()

    if args.VX:
        collect_mean_vx(args.i, args.o, args.Ps, args.tc, args.dt)
    elif args.VXf:
        mean_vx_to_file(args.i, args.o, args.PsL, args.PfL)
