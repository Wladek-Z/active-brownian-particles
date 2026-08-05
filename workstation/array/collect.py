import numpy as np
from pathlib import Path
import argparse

NBLOCKS = 125       # Number of logscale blocks
NCONFIGS = 17       # Number of measurements per block
N = 1000            # Number of particles
BLOCKSIZE = 80000   # Number of timesteps per block

def collect_velocities(input, output, Ps, dt):
    """
    Collect the instantaneous velocities for a histogram pertaining to a certain combination
    of swim and flow Peclet numbers.
    
    Arguments:
        input: directory containing raw trajectories
        output: file to store velocity data
        Ps: swim Peclet number
        dt: simulation timestep
    """
    # Initialise instantaneous velocity array
    velocities = np.zeros((N, NBLOCKS))
    # Initialise particle index
    n = -1

    # Iterate over each particle trajectory in directory
    for file in Path(input).glob("*.txt"):
        # Increment particle index
        n += 1
        # Read in x-position data
        x = np.loadtxt(file, skiprows=1, delimiter=',', usecols=0)
        # Calculate instantaneous velocities
        v = (x[1::NCONFIGS] - x[::NCONFIGS]) / dt
        # Save to velocities array
        velocities[n] = v / Ps

    # Save flattened velocity data to file for histogram
    np.savetxt(output, velocities.flatten())

def collect_mean_vx(input, output, Ps, dt):
    """
    Calculate the mean instantaneous longitudinal velocity for a given combination of
    Peclet numbers by taking the velocity at each pair of consecutive timesteps, separated 
    by logscale blocks. Save result to file.
    
    Arguments:
        input: directory containing raw trajectories
        output: file to store results
        Ps: swim Peclet number
        dt: simulation timestep
    """
    # Initialise sum of x-velocities
    sum_vx = 0.0

    # Iterate over each particle trajectory in directory
    for file in Path(input).glob("*.txt"):
        # Increment number of particles
        particles += 1
        # Read in x-position data
        x = np.loadtxt(file, skiprows=1, delimiter=',', usecols=0)
        # Calculate instantaneous velocities
        v = (x[1::NCONFIGS] - x[::NCONFIGS]) / dt
        # Add sum to sum
        sum_vx += np.sum(v)

    # Calculate mean velocity in terms of Ps
    mean_vx = sum_vx / NBLOCKS / N / Ps   

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

def get_powerlaw(t, msd, l):
    """
    Obtain the late-time power-law dependence of the MSD.
    
    Argument:
        t: measurement times
        msd: mean square displacements
        l: when to start counting late-time data
    
    Returns:
        a: fitted power
        b: fitted logarithm of prefactor
    """
    # Consider only very late times
    late = t >= l
    # Take the logarithms of time and MSD
    y = np.log(msd[late])
    x = np.log(t[late])
    # Fit to a 1st degree polynomial
    a, b = np.polyfit(x, y, 1)
    # return fitted parameters
    return a, b

def collect_alpha(input, output, Ps_params, Pf_params, late):
    """
    Calculate the MSD scaling exponent (alpha) by fitting a power-law to the 
    late-time (t > 1000) MSD data. Save to file.
    
    Arguments:
        input: directory containing the MSD files for each point in Peclet number-space
        output: file to save the resulting scaling exponent
        Ps_params: file containing swim Peclet numbers
        Pf_params: file containing flow Peclet numbers
        late: when to consider data as 'late-time'
    """
    # Read in Peclet number lists
    Ps_list = np.loadtxt(Ps_params, dtype=str)
    Pf_list = np.loadtxt(Pf_params, dtype=str)

    # Write header to file
    with open(output, 'w') as f:
        f.write("# Ps Pf alpha\n")

    # Iterate over each point in phase space
    for Ps, Pf in zip(Ps_list, Pf_list):
        # Read in MSD data
        input_file = f"{input}/{Ps} {Pf}.txt"
        t, msd = np.loadtxt(input_file, unpack=True)
        # Calculate alpha
        alpha, _ = get_powerlaw(t, msd, late)
        # Write to file
        with open(output, 'a') as f:
            f.write(f"{Ps} {Pf} {alpha}\n")

def collect_trajectories(folder, sample, Ps, Pf, output):
    """
    Extract the displacements along the x-direction for a sample number of particles, for a
    given point in Peclet number space. Save to file.
    
    Arguments:
        folder: directory containing all N trajectories for a given set of Peclet numbers
        sample: how many trajectories to sample from the folder
        Ps: swim Peclet number
        Pf: flow Peclet number
        output: name of output file
    """
    # Initialise empty data array
    data = np.empty((sample, NBLOCKS * NCONFIGS))
    
    # Verify existence of specified directory
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    # Sort raw trajectory files
    files = sorted(folder.glob("*.txt"))

    # Iterate over each particle
    for n, file in enumerate(files):
        # Read data from file
        d = np.loadtxt(file, delimiter=',', skiprows=1, usecols=0)
        # Insert into data array
        data[n] = d
        # Break after retrieving sample trajectories 
        if n == sample - 1:
            break

    # Save to npz file
    np.savez(output, x=data)


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
    parser.add_argument('--alpha', action='store_true', help="Collect entire set of MSD scaling exponents")
    parser.add_argument('-l', type=int, default=1000, help="Late-time data start")
    parser.add_argument('-s', type=int, default=1000, help="Number of samples to take from raw data")
    parser.add_argument('--trajectory', action='store_true', help="Collect the trajectories of a number of particles")
    parser.add_argument('--velocities', action='store_true', help="Collect the instantaneous velocities for a phase point")
    args = parser.parse_args()

    if args.VX:
        collect_mean_vx(args.i, args.o, args.Ps, args.dt)
    elif args.VXf:
        mean_vx_to_file(args.i, args.o, args.PsL, args.PfL)
    elif args.alpha:
        collect_alpha(args.i, args.o, args.PsL, args.PfL, args.l)
    elif args.trajectory:
        collect_trajectories(args.i, args.s, args.Ps, args.Pf, args.o)
    elif args.velocities:
        collect_velocities(args.i, args.o, args.Ps, args.dt)
