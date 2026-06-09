import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import uniform_direction
import argparse
import random

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def run(N, r, e, v, mu, dt, T, w, U):
    """
    Run the simulation for T timesteps and keep track of positions and orientations.

    Arguments:
        N: number of particles
        r: particle positions
        e: particle orientations
        v: ABP velocity
        mu: ABP mobility
        dt: timestep
        T: total number of timesteps
        w: width of the channel
        U: maximum flow velocity

    Returns:
        pos_H: history of positions
        pos_O: history of orientations
    """
    # Initialise arrays to keep track of position/orientation data
    pos_H = np.zeros((T+1, N, 2))
    pos_O = np.zeros((T+1, N, 2))
    # Store position and orientation of each particle
    for j in range(N):
        for k in range(2):
            pos_H[0, j, k] = r[j, k]
            pos_O[0, j, k] = e[j, k]
    # Perform T iterations of the update procedure
    for i in range(1, T+1):
        # Update position and orientation
        r_old = r.copy()
        e_old = e.copy()
        r, e = update(N, r_old, e_old, v, mu, dt, w, U)
        # Store position and orientation of each particle
        for j in range(N):
            for k in range(2):
                pos_H[i, j, k] = r[j, k]
                pos_O[i, j, k] = e[j, k]
    return pos_H, pos_O   

@njit
def update(N, r, e, v, mu, dt, w, U):
        """
        Update the positions and orientations of each particle.
        
        Arguments:
            N: number of particles
            r: particle positions
            e: particle orientations
            v: ABP velocity
            mu: ABP mobility
            dt: timestep
            w: width of the channel
            U: maximum flow velocity

        Returns:
            r_new: updated positions
            e_new: updated orientations
        """
        r_new = np.zeros_like(r)
        e_new = np.zeros_like(e)
        # Initialise critical distance to obstacles
        d_crit = 2**(1/6)
        # Iterate over every particle
        for i in range(N):
            # Compute active velocity term
            r_active = v * dt * e[i]
            # Generate positional noise term
            r_noise = np.sqrt(2 * dt) * np.random.normal(0, 1, 2)
            # Update position via forward difference scheme
            r_new[i] = r[i] + r_active + r_noise
            # Incorporate correction due to fluid flow
            r_new[i, 0] += dt * 4 * U * r[i, 1] / w * (1 - r[i, 1] / w)
            # Initialise correction due to repulsive interactions with obstacles
            correction = np.zeros(2)
            # Check for obstacles
            if r[i, 0] < d_crit:
                # Add repulsive interaction due to left wall
                sep = np.array([r[i, 0], 0])
                correction += repulsion(sep)
            if r[i, 1] < d_crit:
                # Add repulsive interaction due to lower wall
                sep = np.array([0, r[i, 1]])
                correction += repulsion(sep)
            if (w - r[i, 1]) < d_crit:
                # Add repulsive interaction due to upper wall
                sep = np.array([0, (r[i, 1] - w)])
                correction += repulsion(sep)
            r_new[i] += dt * mu * correction
            # Update orientation
            e_new[i] = orientation(e[i], dt, w, U, r[i, 1])
        # Return updated position and orientation vectors
        return r_new, e_new

@njit
def repulsion(r):
    """
    Calculate correction due to repulsive interactions with obstactle.
    
    Arguments:
        r: separation vector from particle to obstacle
    
    Returns:
        repulsive force acting on particle
    """
    r_mag = np.linalg.norm(r)
    return 24 * (2 / r_mag**14 - 1 / r_mag**8) * r

@njit
def orientation(e, dt, w, U, y):
    """
    Calculate and return the new orientation vector in 2D via rotation matrix update.
    
    Arguments:
        e: original orientation vector
        dt: timestep
        w: width of channel
        U: maximum flow velocity in channel
        y: height of particle within channel
        
    Returns:
        the updated orientation vector
    """
    # Generate noise term
    noise = np.random.normal(0, 1)
    # Calculate change in orientation due to noise
    d_theta_noise = np.sqrt(2 * dt) * noise
    # Calculate change in orientation due to angular velocity from flow profile
    d_theta_flow = dt * 2 * U / w * (2 * y / w - 1)
    # Calculate angle from x-axis of the original orientation vector
    theta = np.arctan2(e[1], e[0])
    # Calculate new angle
    new_theta = theta + d_theta_noise + d_theta_flow
    # Return updated orientation vector
    return np.array([np.cos(new_theta), np.sin(new_theta)])

@njit
def positions(N, width):
    """
    Generate the positions of N particles near the beginning of a channel.
    
    Arguments:
        N: number of particles
        width: width of channel
    
    Returns:
        r: positions of each particle
    """
    x_min = 2**(1/6)
    x_max = 0.5 * width
    y_min = 2**(1/6)
    y_max = width - 2**(1/6)
    # Initialise positions
    r = np.zeros((N, 2))
    # Randomly choose x and y components within specified range
    for i in range(N):
        r[i, 0] = random.uniform(x_min, x_max)
        r[i, 1] = random.uniform(y_min, y_max)
    # Return position array
    return r


class ABP:
    """
    An ensemble of active Brownian particles submerged in fluid, for a given system geometry (2D).
    """

    def __init__(self, N, dt, width, T):
        """
        Initialise N realisations of the same particle at the origin with random orientations.
        
        Arguments:
            N: number of realisations of the particle
            dt: timestep
            width: width of the channel
            T: total number of timesteps
        """
        # Initialise variables
        self.N = N
        self.dt = dt
        self.width = width
        self.T = T
        # Initialise orientation vector of each particle from a uniform rotationally symmetric distribution
        distribution = uniform_direction(2)
        self.e = distribution.rvs(N)
        # Initialise positions
        self.r = positions(N, width)
        
 
    def Run(self, v, mu, U):
        """
        Run the simulation and return the results.
        
        Arguments:
            v: ABP velocity
            mu: ABP mobility
            U: maximum flow velocity
        
        Returns:
            pos: position history of each particle
            orient: orientation history of each particle
        """
        # Run simulation and retrieve data
        pos, orient = run(self.N, self.r, self.e, v, mu, self.dt, self.T, self.width, U)
        return pos, orient        


    def MSD(self, data, v):
        """
        Calculate the mean sqaure displacement of an ensemble of particles over time.
        Plot the results on a graph.

        Arguments:
            data: ABP position history
            v: ABP velocity
        """
        # Calculate mean square displacement along each Cartesian direction
        msds = np.mean((data - data[0])**2, axis=1)
        # Calculate total mean square displacement, omitting initial position for logscale
        msd = np.sum(msds, axis=1)[1:]

        # Create array of timesteps
        t = np.arange(1, self.T + 1) * self.dt
        # System is 2-dimensional
        d = 2
        # Calculate persistence time
        tau = 1 / (d - 1)
        # Theoretical mean square displacement
        msd_theory = 2 * d * t + 2 * v**2 * tau * t - 2 * v**2 * tau**2 * (1 - np.exp(-t / tau))
        # Theoretical msd for ballistic and diffusive regimes
        msd_b = v**2 * t**2 + 2 * d * t
        msd_d = (2 * d + 2 * v**2 * tau) * t 

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Mean Square Displacement")
        plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d, color='green', linestyle='--', label='diffusive')
        plt.axvline(tau, color='orange', label=r'$\tau_r$')
        plt.xlabel(r"time [$1/D_r$]")
        plt.ylabel(r"$\langle r^2 \rangle$ [$\sigma^2$]")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def Trajectory(self, data):
        """
        Retrieve the positions of an ensemble of particles over time.
        Plot results on a graph.
        
        Arguments:
            data: ABP position history
        """
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Particle Trajectory")
        
        for i in range(self.N):
            x, y = data[:, i, 0], data[:, i, 1]
            start_x, start_y = data[0, i, 0], data[0, i, 1]
            plt.scatter(start_x, start_y, color='red', s=20, zorder=1)
            plt.plot(x, y, color='black', zorder=-1)

        plt.xlabel(r"$x$ position [$\sigma$]")
        plt.ylabel(r"$y$ position [$\sigma$]")
        plt.xlim(0)
        plt.ylim(0, self.width)
        plt.tight_layout()
        plt.show()


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
    parser.add_argument('-u', type=float, default=0, help='Maximum flow velocity')
    args = parser.parse_args()

    abp = ABP(args.N, args.dt, args.w, args.t)
    pos, orient = abp.Run(args.v, args.mu, args.u)

    if args.MSD:
        abp.MSD(pos, args.v)
    if args.trajectory:
        abp.Trajectory(pos)