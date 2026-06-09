import argparse
from abp import ABP

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('--MSD', action='store_true', help='Find the mean square displacement')
    parser.add_argument('--trajectory', action='store_true', help='Find the particle trajectory')
    parser.add_argument('-t', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-v', type=float, default=10, help='ABP velocity')
    parser.add_argument('-mu', type=float, default=10, help='ABP mobility')
    parser.add_argument('-w', default=10, type=float, help='Width of channel')
    parser.add_argument('-x', type=float, default=10, help='Distance along x-axis for first-passage time distribution')
    args = parser.parse_args()