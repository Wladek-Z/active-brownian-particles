import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

NBLOCKS = 125       # Number of logscale blocks
NCONFIGS = 17       # Number of measurements per block
N = 1000            # Number of particles
BLOCKSIZE = 80000   # Number of timesteps per block

# Calculate logarithmic sample times within a block
sample_times = 2**np.arange(NCONFIGS)

def get_data(folder, column, offset, sample=N):
    """
    Extract a column of trajectory data from a directory containing the logscaled trajectories,
    then manipulate into a more useful shape for data analysis. Consider only measurements
    made after the first 'offset' number of blocks (13 blocks = first 1000 units of time).
    
    Arguments:
        folder: directory containing all N trajectories for a given set of Peclet numbers
        column: which set of data to extract (0 = x, 1 = y, 2 = theta)
        offset: how many blocks of data to skip at the start of each trajectory
        sample: how many trajectories to sample from the folder
    
    Returns:
        data: reshaped data, shape = (NBLOCKS - offset, NCONFIGS, N)
    """
    # Initialise empty data array
    data = np.empty((NBLOCKS - offset, NCONFIGS, sample))
    
    # Verify existence of specified directory
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    # Sort raw trajectory files
    files = sorted(folder.glob("*.txt"))

    # Check number of trajectories matches number of particles
    if len(files) != N:
        raise RuntimeError(f"Expected {N} trajectory files, found {len(files)} in {folder}")

    # Iterate over each particle
    for n, file in enumerate(files):
        # Calculate number of entries to skip (+ 1 for header)
        skips = offset * NCONFIGS + 1
        # Read data from file
        d = np.loadtxt(file, delimiter=',', skiprows=skips, usecols=column)
        # Reshape data
        d = np.reshape(d, (NBLOCKS - offset, NCONFIGS))
        # Insert into data array
        data[:, :, n] = d
        # Return after retrieving sample trajectories 
        if n == sample - 1:
            return data

def get_old_data(folder, column, offset, sample=N):
    """
    Extract the positions along each particle trajectory for a point in Peclet
    number-space.
    
    Arguments:
        folder: directory containing all N trajectories for a given set of Peclet numbers
        column: which set of data to extract (0 = x, 1 = y, 2 = theta)
        offset: how many blocks of data to skip at the start of each trajectory
        sample: how many trajectories to sample from the folder
    
    Returns:
        data: reshaped position data ((NBLOCKS - offset) * NCONFIGS, N)
    """
    # Initialise empty data array
    data = np.empty(((NBLOCKS - offset) * NCONFIGS, sample))
    
    # Verify existence of specified directory
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    # Sort raw trajectory files
    files = sorted(folder.glob("*.txt"))

    # Check number of trajectories matches number of particles
    if len(files) != N:
        raise RuntimeError(f"Expected {N} trajectory files, found {len(files)} in {folder}")

    # Iterate over each particle
    for n, file in enumerate(files):
        # Calculate number of entries to skip (+ 1 for header)
        skips = offset * NCONFIGS + 1
        # Read data from file
        d = np.loadtxt(file, delimiter=',', skiprows=skips, usecols=column)
        # Insert into data array
        data[:, n] = d
        # Return after retrieving sample trajectories 
        if n == sample - 1:
            return data

def get_lags(filename, offset):
    """
    Save the set of unique time intervals between each combination of measurements
    along a particle's trajectory, going strictly forwards in time, in ascending order.

    Arguments:
        filename: filepath to save each possible time interval
        offset: number of blocks by which to offset the data
    """
    # Initialise the set of possible time intervals between measurements
    lags = set()
    # Calculate number of blocks after offset
    nblocks = NBLOCKS - offset

    # Iterate over each block (separation between blocks)
    for block_sep in range(nblocks):
        # Iterate over each measurement within a block
        for i in range(NCONFIGS):
            # Avoid measurements going backwards in time
            j_start = i + 1 if block_sep == 0 else 0
            for j in range(j_start, NCONFIGS):
                # Add time interval to lags set
                lags.add(block_sep * BLOCKSIZE + sample_times[j] - sample_times[i])

    # Sort time intervals in ascending order
    lags = np.array(sorted(lags), dtype=np.int64)
    # Save to npz file
    np.savez(filename, lags)

def get_MSD(folder, Ps, Pf, output, dt, dim, offset, lagsfile):
    """
    Calculate the MSD in one dimension for a given combination of Peclet numbers by
    averaging over every available time interval, save output to file.
    
    Arguments:
        folder: directory containing the raw trajectories for a point in Peclet number-space
        Ps: swim Peclet number of interest
        Pf: flow Peclet number of interest
        output: file in which to store the calculated MSD
        dt: simulation timestep for calculating the relevant time intervals
        dim: dimension along which to calculate MSD (0 = x, 1 = y)
        offset: how many blocks of data to skip at the start of each trajectory
        lagsfile: input file containing the unique measurement time intervals
    """
    # Read in and reshape position data
    x = get_data(folder, dim, offset)

    # Calculate number of blocks, adjusting for offset
    nblocks = x.shape[0]
  
    # Retrieve the set of possible time intervals between measurements from input file
    lags = np.load(lagsfile)['arr_0']
    # Create lookup table for indices corresponding to each lag
    lag_index = {lag: k for k, lag in enumerate(lags)}

    # Initialise sum of MSDs array, counts array
    msd_sum = np.zeros(len(lags))
    count = np.zeros(len(lags), dtype=int)

    # Iterate over each possible block separation
    for block_sep in range(nblocks):
        # Iterate over each measurement within a block
        for i in range(NCONFIGS):
            # Avoid measurements going backwards in time
            j_start = i + 1 if block_sep == 0 else 0
            for j in range(j_start, NCONFIGS):
                # Compute the time interval corresponding to this set of measurements
                lag = (block_sep * BLOCKSIZE + sample_times[j] - sample_times[i])
                # Check if computed lag definitely appears within dictionary
                assert lag in lag_index
                # Obtain lag index
                k = lag_index[lag]
                # Calculate displacement
                dx = x[block_sep:, j] - x[:nblocks - block_sep, i]
                # Add to MSD corresponding to this lag
                msd_sum[k] += np.sum(dx**2)
                # Increment count corresponding to this lag
                count[k] += dx.size

    # Check for division by zero, in case of any count = 0
    if np.any(count == 0):
        raise RuntimeError("Some lag times received no samples.")
    # Calculate mean square displacement, time intervals
    msd = msd_sum / count
    times = lags * dt

    # Get path to output
    filename = f"{output}/{Ps} {Pf}.txt"
    # Save to file
    np.savetxt(filename, np.column_stack((times, msd)), header='time MSD')

def get_old_MSD(folder, Ps, Pf, output, dt, dim, offset, timechain):
    """
    Calculate the MSD in one dimension for a given combination of Peclet numbers 
    using a single origin and averaging over each particle.
    
    Arguments:
        folder: directory containing the raw trajectories for a point in Peclet number-space
        Ps: swim Peclet number of interest
        Pf: flow Peclet number of interest
        output: file in which to store the calculated MSD
        dt: simulation timestep for calculating the relevant time intervals
        dim: dimension along which to calculate MSD (0 = x, 1 = y)
        offset: how many blocks of data to skip at the start of each trajectory
        timechain: file containing the logscale timechain
    """
    # Read in and reshape position data
    x = get_old_data(folder, dim, offset)
    # Read in timechain
    tc = np.loadtxt(timechain)[offset:]

    # Calculate MSD
    msd = np.mean((x[1:] - x[0])**2)
    # Obtain measurement times
    times = tc[offset:] * dt

    # Get path to output
    filename = f"{output}/{Ps} {Pf}.txt"
    # Save to file
    np.savetxt(filename, np.column_stack((times, msd)), header='time MSD')

def get_particle_MSD(folder, Ps, Pf, output, dt, dim, offset, lagsfile, sample):
    """
    Calculate the MSD in one dimension for the trajectory of individual particles,
    averaging only over different origins, for a given combination of Peclet numbers. 
    Save output to file.
    
    Arguments:
        folder: directory containing the raw trajectories for a point in Peclet number-space
        Ps: swim Peclet number of interest
        Pf: flow Peclet number of interest
        output: file in which to store the calculated SDs
        dt: simulation timestep for calculating the relevant time intervals
        dim: dimension along which to calculate SD (0 = x, 1 = y)
        offset: how many blocks of data to skip at the start of each trajectory
        lagsfile: input file containing the unique measurement time intervals
        sample: number of trajectories to consider
    """
    # Read in and reshape position data
    x = get_data(folder, dim, offset, sample)

    # Calculate number of blocks, adjusting for offset
    nblocks = x.shape[0]
  
    # Retrieve the set of possible time intervals between measurements from input file
    lags = np.load(lagsfile)['arr_0']
    # Create lookup table for indices corresponding to each lag
    lag_index = {lag: k for k, lag in enumerate(lags)}

    # Initialise sum of SDs array, counts array
    sd_sum = np.zeros((sample, len(lags)))
    count = np.zeros((sample, len(lags)), dtype=int)

    # Iterate over each possible block separation
    for block_sep in range(nblocks):
        # Iterate over each measurement within a block
        for i in range(NCONFIGS):
            # Avoid measurements going backwards in time
            j_start = i + 1 if block_sep == 0 else 0
            for j in range(j_start, NCONFIGS):
                # Compute the time interval corresponding to this set of measurements
                lag = (block_sep * BLOCKSIZE + sample_times[j] - sample_times[i])
                # Check if computed lag definitely appears within dictionary
                assert lag in lag_index
                # Obtain lag index
                k = lag_index[lag]
                # Calculate displacement
                dx = x[block_sep:, j] - x[:nblocks - block_sep, i]
                # Add sum of squares corresponding to each particle at this lag
                sd_sum[:, k] += np.sum(dx**2, axis=0)
                # Increment count corresponding to this lag
                count[:, k] += dx.shape[0]

    # Check for division by zero, in case of any count = 0
    if np.any(count == 0):
        raise RuntimeError("Some lag times received no samples.")
    # Calculate mean square displacement, time intervals
    msd = sd_sum / count
    times = lags * dt

    # Get path to output
    filename = f"{output}/{Ps} {Pf} n{sample}.npz"
    # Save to file
    np.savez(filename, time=times, MSD=msd)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', type=str, help='Folder containing data for each phase point')
    parser.add_argument('-Ps', type=str, help='Swim Peclet number')
    parser.add_argument('-Pf', type=str, help='Flow Peclet number')
    parser.add_argument('-tc', type=str, help='Filepath to the logscale timechain file')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-f', type=str, help='Path to input file')
    parser.add_argument('-o', type=str, help='Path to output file/directory')
    parser.add_argument('-d', type=int, default=0, help='Dimension along which to compute MSD')
    parser.add_argument('--lag', action='store_true', help='Calculate the possible measurement time intervals')
    parser.add_argument('--MSD', action='store_true', help='Calculate the mean square displacement')
    parser.add_argument('--old', action='store_true', help='Calculate the mean square displacement using a single origin approach')
    parser.add_argument('--ppMSD', action='store_true', help='Calculate the per-particle mean square displacement')
    parser.add_argument('-off', default=0, type=int, help='Number of blocks by which to offset the data')
    parser.add_argument('-s', default=N, type=int, help="Number of particle trajectories to use for results")
    args = parser.parse_args()

    if args.lag:
        get_lags(args.o, args.off)
    elif args.MSD:
        get_MSD(args.F, args.Ps, args.Pf, args.o, args.dt, args.d, args.off, args.f)
    elif args.ppMSD:
        get_particle_MSD(args.F, args.Ps, args.Pf, args.o, args.dt, args.d, args.off, args.f, args.s)
    elif args.old:
        get_old_MSD(args.F, args.Ps, args.Pf, args.o, args.dt, args.dim, args.off, args.tc)

