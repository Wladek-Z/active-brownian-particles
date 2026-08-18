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
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))

plt.style.use('science')
plt.rcParams['text.usetex'] = False

# Developer tools ;)
d = 2
tau = 1 / (d - 1)
vorticity = 1
noise_r = 1

@njit
def update_OLD(N, r, theta, dt, Ps, D, Pf, G, T_timer, B_timer):
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
            T_timer: timer for tracking traps per particle
            B_timer: timer for tracking bulk swimming per particle

        Returns:
            r_new: updated positions
            theta_new: updated orientations
            t_timer: increments for trapping timer
            b_timer: increments for bulk timer
            trap_times: trapping times per particle
            bulk_times: times in the bulk per particle
        """
        # Initialise updated position/orientation variables
        r_new = np.zeros_like(r)
        theta_new = np.zeros_like(theta)
        # Initialise trapping criteria parameters
        bottom = 0.05
        top = 0.95
        min_time = 100
        # Initialise trapping/bulk time arrays in case of successful trap/bulk path completion
        trap_times = np.full(N, np.nan)
        bulk_times = np.full(N, np.nan)
        # Initialise incremental timers
        t_timer = np.zeros(N)
        b_timer = np.zeros(N)

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

            # Extract transverse coordinates
            y = r_new[i, 1]
            y_old = r[i, 1]

            # Check for traps/bulk
            if y_old == y and T_timer[i] == 0:
                bulk_times[i] = B_timer[i] * dt
                t_timer[i] += 1
            elif T_timer[i] > 0 and (bottom >= y or y >= top):
                t_timer[i] += 1
            elif T_timer[i] > min_time and (bottom <= y <= top):
                trap_times[i] = T_timer[i] * dt
                b_timer[i] += 1
            else:
                b_timer[i] += 1
            
            
        # Return updated position and orientations
        return r_new, theta_new, t_timer, b_timer, trap_times, bulk_times

@njit
def update(N, r, theta, dt, Ps, D, Pf, G, T_timer):
        """
        Update the positions and orientations of each particle, track traps.
        
        Arguments:
            N: number of particles
            r: particle positions
            theta: particle orientations (angular form)
            dt: timestep 
            Ps: swim Peclet number
            D: effective diffusivity
            Pf: flow Peclet number
            G: elongation factor
            T_timer: timer for tracking traps

        Returns:
            r_new: updated positions
            theta_new: updated orientations
            trap_complete: True if particle escapes trap
            is_trap: True for particles apparently in a trap
        """
        # Initialise updated position/orientation variables
        r_new = np.zeros_like(r)
        theta_new = np.zeros_like(theta)
        # Initialise trapping criteria parameters
        bottom = 0.05
        top = 0.95
        min_time = 100
        # Check status if particle escapes trap
        trap_complete = np.full(N, False)
        # Check status if particle appears trapped
        is_trap = np.full(N, False)

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

            # Extract transverse coordinates
            y = r_new[i, 1]
            y_old = r[i, 1]

            # Check for traps
            if (y_old == y and T_timer[i] == 0) or (T_timer[i] > 0 and (bottom > y or y > top)):
                is_trap[i] = True
            elif T_timer[i] > min_time and (bottom <= y <= top):
                trap_complete[i] = True
            
        # Return updated position and orientations
        return r_new, theta_new, trap_complete, is_trap

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

def read_configurations(folder, N):
    """
    Read final configurations from raw logscale trajectories in a directory pertaining to
    a specific combination of swim and flow Peclet numbers.
    
    Arguments:
        folder: directory containing raw trajectory files
        N: number of particles
    
    Returns:
        r: ABP positions
        e: ABP orientations
    """
    # Initialise positions/orientations
    r = np.zeros((N, 2))
    e = np.zeros((N, 2))

    # Iterate over each file in directory
    for i in range(N):
        # Resolve filepath
        filename = f"{folder}/{i}.txt"

        # Read in last line of data
        with open(filename, 'rb') as f:
            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()

        # Retrieve position and orientation from last line
        x, y, theta = last_line.split(',')
        # Assign x and y to position vector
        r[i, 0] = float(x)
        r[i, 1] = float(y)
        # Obtain orientation vector from angle
        e[i, 0] = np.cos(float(theta))
        e[i, 1] = np.sin(float(theta))

    # Return configuration of ABPs
    return r, e
        

class ABPTrap:
    """
    ABP simulation code for systems with a large number of particles and timesteps. Only trapping times and
    bulk times are stored by this program.
    """

    def __init__(self, N, T, dt, Ps, D, Pf, G, configs):
        """
        Initialise N realisations of the ABP starting from the final configurations of the 10M timestep
        logscale trajectories.
        
        Arguments:
            N: number of realisations of the particle
            T: total number of timesteps
            dt: timestep
            Ps: swim Peclet number
            D: diffusion number
            Pf: flow Peclet number
            G: geometrical elongation factor
            configs: directory containing the logscale trajectories
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

        # Initialise starting positions/orientations from final configuration of logscale trajectories
        self.r, self.e = read_configurations(configs, N)
        
 
    def Run(self, trap_file, bulk_file):
        """
        Run the simulation and return the results

        Arguments:
            trap_file: file in which to save ttd data
            bulk_file: file in which to save the btd data
        """
        # Get positions and orientations
        r = self.r
        theta = np.arctan2(self.e[:, 1], self.e[:, 0])
        # Initialise trap and bulk timers
        T_timer = np.zeros(self.N)
        B_timer = np.zeros(self.N)
        # Initialise trapping/bulk time output files
        open(trap_file, 'w')
        open(bulk_file, 'w')
        # Only track bulk times after completing first trap
        valid_bulk = np.full(self.N, False)
        # Get number of decimal places for rounding output
        dec_places = len(str(self.dt).split('.')[1])

        # Perform T iterations of the update procedure
        for i in range(1, self.T+1):
            # Update position and orientation
            r, theta, trap_complete, is_trap = update(self.N, 
                                                      r, 
                                                      theta, 
                                                      self.dt, 
                                                      self.Ps, 
                                                      self.D, 
                                                      self.Pf, 
                                                      self.G, 
                                                      T_timer)

            # Iterate over each particle and write trapping/bulk times to file, reset timers
            for n in range(self.N):
                # Check for trap completion
                if trap_complete[n]:
                    # Calculate trap time
                    trap = T_timer[n] * self.dt
                    # Save trap time to file
                    with open(trap_file, 'a') as f:
                        f.write(f"{np.round(trap, dec_places)}\n")

                    # Calculate bulk time
                    if valid_bulk[n]:
                        bulk = (B_timer[n] - T_timer[n]) * self.dt
                        # Save bulk time to file
                        with open(bulk_file, 'a') as f:
                            f.write(f"{np.round(bulk, dec_places)}\n")
                    # Validate bulk measurements
                    else:
                        valid_bulk[n] = True

                    # Reset trap and bulk timers
                    T_timer[n] = 0
                    B_timer[n] = 0

                # Increment timers if particle appears to be in a trap
                elif is_trap[n]:
                    T_timer[n] += 1
                    B_timer[n] += 1

                # Keep trap timer at zero if particle not in trap
                else:
                    T_timer[n] = 0
                    B_timer[n] += 1



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1000, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('-T', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-Ps', type=float, default=5, help='Swim Peclet number')
    parser.add_argument('-Pf', type=float, default=5, help='Flow Peclet number')
    parser.add_argument('-D', type=float, default=0.01, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-G', type=float, default=0, help='Geometrical factor related to particle aspect ratio')
    parser.add_argument('-ttd', type=str, default=None, help="Filepath to store the output ttd data")
    parser.add_argument('-btd', type=str, default=None, help="Filepath to store the output btd data")
    parser.add_argument('-c', type=str, help="Directory containing logscale trajectories to read final ABP configurations")
    args = parser.parse_args()

    # Create ABP (trapping/bulk times) object
    abp = ABPTrap(args.N, args.T, args.dt, args.Ps, args.D, args.Pf, args.G, args.c)
    # Run simulation
    abp.Run(args.ttd, args.btd)
