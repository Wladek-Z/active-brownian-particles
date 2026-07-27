import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import uniform_direction
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import argparse
import random
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

plt.style.use('science')
plt.rcParams['text.usetex'] = False

# Developer tools ;)
d = 2
tau = 1 / (d - 1)
vorticity = 1
noise_r = 1
centre_start = False

@njit
def update(N, r, theta, dt, Ps, D, Pf, G):
        """
        Update the positions and orientations of each particle.
        
        Arguments:
            N: number of particles
            r: particle positions
            theta: particle orientations (angular form)
            dt: timestep 
            Ps: swim Peclet number
            D: effective diffusivity
            Pf: flow Peclet number
            G: elongation factor

        Returns:
            r_new: updated positions
            theta_new: updated orientations
        """
        r_new = np.zeros_like(r)
        theta_new = np.zeros_like(theta)
        # Iterate over every particle
        for i in range(N):
            # Calcaulte orientation
            e = np.array([np.cos(theta[i]), np.sin(theta[i])])
            # Compute swim velocity term
            r_swim = dt * Ps * e 
            # Generate translational noise term
            r_noise = np.sqrt(2 * D * dt) * np.random.normal(0, 1, 2) 
            # Update position via forward difference scheme
            r_new[i] = r[i] + r_swim + r_noise
            # Incorporate correction due to fluid flow
            r_new[i, 0] += 4 * dt * Pf * r[i, 1] * (1 - r[i, 1]) 
            # Impose reflection at boundaries
            if r_new[i, 1] < 0 or r_new[i, 1] > 1:
                r_new[i, 1] = r[i, 1]
            # Update orientation
            theta_new[i] = orientation(theta[i], dt, Pf, r_new[i, 1], G)
        # Return updated position and orientations
        return r_new, theta_new

@njit
def orientation(theta, dt, Pf, y, G):
    """
    Calculate and return the new orientation vector in 2D via rotation matrix update.
    
    Arguments:
        theta: original orientation angle
        dt: timestep
        Pf: flow Peclet number
        y: height of particle within channel
        G: elongation factor
        
    Returns:
        the updated orientation angle
    """
    # Calculate change in orientation due to rotational noise
    d_theta_noise = noise_r * np.sqrt(2 * dt) * np.random.normal(0, 1)
    # Calculate angular velocity due to vorticity and shear
    d_theta_omega = 2 * dt * Pf * (1 - 2 * y) * (G * np.cos(2 * theta) - vorticity)
    # Calculate the new angle
    new_theta = theta + d_theta_noise + d_theta_omega
    # Return the new angle wrapped to [-pi, pi]
    return (new_theta + np.pi) % (2 * np.pi) - np.pi

@njit
def positions(N):
    """
    Generate the positions of N particles near the beginning of a channel.
    
    Arguments:
        N: number of particles
    
    Returns:
        r: positions of each particle
    """
    x_min = -0.25
    x_max = 0.25
    y_min = 0.25
    y_max = 0.75
    # Initialise positions
    r = np.zeros((N, 2))
    # Randomly choose x and y components within specified range
    for i in range(N):
        r[i, 0] = random.uniform(x_min, x_max)
        r[i, 1] = random.uniform(y_min, y_max)
    # Return position array
    return r

class ABPLog:
    """
    ABP simulation code for systems with a large number of particles and timesteps. Position and orientation
    history is saved to file using a loscale timechain, and only the current positions and orientations are
    stored in the code as it is run.
    """

    def __init__(self, N, T, dt, Ps, D, Pf, G):
        """
        Initialise N realisations of the same particle at the origin with random orientations.
        
        Arguments:
            N: number of realisations of the particle
            T: total number of timesteps
            dt: timestep
            Ps: swim Peclet number
            D: diffusion number
            Pf: flow Peclet number
            G: geometrical elongation factor
        """
        # Initialise variables
        self.N = N
        self.T = T
        self.dt = dt
        self.Ps = Ps
        self.D = D
        self.Pf = Pf
        self.G = G
        self.step = int(1 / dt)
        # Initialise positions
        if centre_start:
            self.r = np.full((N, 2), [0, 0.5])
            self.e = np.full((N, 2), [1/np.sqrt(2), 1/np.sqrt(2)])
        else:
            self.r = positions(N)
            # Initialise orientation vector of each particle from a uniform rotationally symmetric distribution
            distribution = uniform_direction(2)
            self.e = distribution.rvs(N)
        
 
    def Run(self, filename, folder):
        """
        Run the simulation and return the results

        Arguments:
            filename: file containing logscale timechain indices
            folder: directory in which to store raw data
        """
        # Initialise data files for each ABP
        for j in range(self.N):
            with open(f"{folder}/{j}.txt", "w") as f:
                f.write('x,y,theta\n')
        # Get positions and orientations
        r = self.r
        theta = np.arctan2(self.e[:, 1], self.e[:, 0])
        # Read in timechain
        timechain = np.loadtxt(filename, dtype=np.int64)
        # Initialise logscale time index
        k = 0
        # Perform T iterations of the update procedure
        for i in range(1, self.T+1):
            # Update position and orientation
            r, theta = update(self.N, r, theta, self.dt, self.Ps, self.D, self.Pf, self.G)
            # Check if current index in timechain
            if k < len(timechain) and i == timechain[k]:
                # Increment logscale time
                k += 1
                # Iterate over each ABP
                for j in range(self.N):
                    with open(f"{folder}/{j}.txt", "a") as f:
                        f.write(f"{r[j, 0]},{r[j, 1]},{theta[j]}\n")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-T', type=int, default=10000000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-Ps', type=float, default=5, help='Swim Peclet number')
    parser.add_argument('-Pf', type=float, default=5, help='Flow Peclet number')
    parser.add_argument('-D', type=float, default=0.01, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-G', type=float, default=0, help='Geometrical factor related to particle aspect ratio')
    parser.add_argument('-tc', type=str, help='File containing logscale timechain')
    parser.add_argument('-F', type=str, default=None, help="Folder to store the raw data")
    args = parser.parse_args()

    # Create ABP (logscale data) object
    abp = ABPLog(args.N, args.T, args.dt, args.Ps, args.D, args.Pf, args.G)
    # Run simulation
    abp.Run(args.tc, args.F)
