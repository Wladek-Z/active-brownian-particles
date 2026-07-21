"""
This file contains code that I've written, but eventually decided not to use in my final simulation (e.g. code for 3D system, keyvalue class for argparser).
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import uniform_direction
import argparse
from scipy.stats import binned_statistic

plt.style.use('science')
plt.rcParams['text.usetex'] = False

@njit
def update(N, r, e, v, dt, d):
        """
        Update the positions and orientations of each particle.
        
        Arguments:
            N: number of particles
            r: particle positions
            e: particle orientations
            v: ABP velocity
            dt: timestep
            d: dimensionality of the system

        Returns:
            r_new: updated positions
            e_new: updated orientations
        """
        r_new = np.zeros_like(r)
        e_new = np.zeros_like(e)
        # Iterate over every particle
        for i in range(N):
            # Generate positional noise term
            r_noise = np.random.normal(0, 1, d)
            # Update position via forward difference scheme
            r_new[i] = r[i] + v * dt * e[i] + np.sqrt(2 * dt) * r_noise
            # Update orientation via dimensionally-dependent formula
            if d == 3:
                e_new[i] = orientation_3d(e[i], dt)
            else:
                e_new[i] = orientation_2d(e[i], dt)
        # Return updated position and orientation vectors
        return r_new, e_new

@njit
def orientation_3d(e, dt):
    """
    Calculate and return the new orientation vector in 3D using the rodrigues formula.
    
    Arguments:
        e: original orientation vector
        dt: timestep
    
    Returns:
        the updated orientation 
    """
    # Generate noise term
    noise = np.random.normal(0, 1, 3)
    # Calculate rotation vector
    rot_vec = np.sqrt(2 * dt) * noise
    # Calculate angle of rotation
    theta = np.linalg.norm(rot_vec)
    # Calculate axis of rotation
    k = rot_vec / theta
    # Return updated orientation vector
    return e * np.cos(theta) + np.cross(k, e) * np.sin(theta) + k * np.dot(k, e) * (1 - np.cos(theta))

@njit
def orientation_2d(e, dt):
    """
    Calculate and return the new orientation vector in 2D via rotation matrix update.
    
    Arguments:
        e: original orientation vector
        dt: timestep
        
    Returns:
        the updated orientation vector
    """
    # Calculate angle from x-axis of the original orientation vector
    theta = np.arctan2(e[1], e[0])
    # Generate noise term
    noise = np.random.normal(0, 1)
    # Calculate change in angle of orientation
    d_theta = np.sqrt(2 * dt) * noise
    # Calculate new angle
    new_theta = theta + d_theta
    # Return updated orientation vector
    return np.array([np.cos(new_theta), np.sin(new_theta)])

@njit
def run(N, r, e, v, dt, T, d):
    """
    Run the simulation for T timesteps and keep track of positions and orientations.

    Arguments:
        N: number of particles
        r: particle positions
        e: particle orientations
        v: ABP velocity
        dt: timestep
        T: total number of timesteps
        d: dimensionality of the system

    Returns:
        pos_H: history of positions
        pos_O: history of orientations
    """
    # Initialise arrays to keep track of position/orientation data
    pos_H = np.zeros((T+1, N, 3))
    pos_O = np.zeros((T+1, N, 3))
    # Store position and orientation of each particle
    for j in range(N):
        for k in range(d):
            pos_H[0, j, k] = r[j, k]
            pos_O[0, j, k] = e[j, k]
    # Perform T iterations of the update procedure
    for i in range(1, T+1):
        # Update position and orientation
        r_old = r.copy()
        e_old = e.copy()
        r, e = update(N, r_old, e_old, v, dt, d)
        # Store position and orientation of each particle
        for j in range(N):
            for k in range(d):
                pos_H[i, j, k] = r[j, k]
                pos_O[i, j, k] = e[j, k]
    return pos_H, pos_O    

@njit
def track_traps1(y, dt):
    """
    Calculate trapping times within a single ABP trajectory.
    
    Arguments:
        y: single particle transverse trajectory
        dt: timestep
    
    Returns:
        idx_start: the indexes where the ABP enters the trapped state
        idx_end: the indexes where the ABP leaves the trapped state
        trap_times: the trapping times in a single trajectory
    """
    bottom = 0.05
    top = 0.95
    idx_start = np.full(len(y), False)
    idx_end = np.full(len(y), False)
    waittime = int(tau / 5 / dt)
    timer_trap = 0
    timer_out = 0
    trap_times = []

    for i in range(1, len(y)):
        if y[i] == y[i-1] and timer_trap == 0:
            idx_start[i] = True
            timer_trap += 1
        elif timer_trap > 0 and (y[i] < bottom or y[i] > top):
            timer_trap += 1
            timer_out = 0
        elif timer_trap > 0 and (y[i] >= bottom or y[i] <= top):
            timer_trap += 1
            timer_out += 1
        if timer_out == waittime:
            trap_times.append((timer_trap - waittime) * dt)
            idx_end[i - waittime] = True
            timer_trap = 0
            timer_out = 0

    return idx_start, idx_end, trap_times

@njit
def track_traps2(y, dt):
    """
    Calculate trapping times within a single ABP trajectory using an alternative method.
    
    Arguments:
        y: single particle transverse trajectory
        dt: timestep
    
    Returns:
        idx_start: the indexes where the ABP enters the trapped state
        idx_end: the indexes where the ABP leaves the trapped state
        trap_times: the trapping times in a single trajectory
    """
    bottom = 0.05
    top = 0.95
    idx_start = np.full(len(y), False)
    idx_end = np.full(len(y), False)
    timer_trap = 0
    waittime = int(tau / 10 / dt)
    timer_in = 0
    timer_out = 0
    trap_times = []

    for i in range(1, len(y)):
        if (y[i] < bottom or y[i] > top) and timer_trap == 0:
            timer_in += 1
        elif (y[i] >= bottom or y[i] <= top) and timer_in > 0:
            timer_in = 0
        elif (y[i] < bottom or y[i] > top) and timer_trap > 0:
            timer_trap += 1
            timer_out = 0
        elif (y[i] >= bottom or y[i] <= top) and timer_trap > 0:
            timer_trap += 1
            timer_out += 1
        if timer_in == waittime:
            idx_start[i - waittime] = True
            timer_trap += waittime
            timer_in = 0
        elif timer_out == waittime:
            idx_end[i - waittime] = True
            trap_times.append((timer_trap - waittime) * dt)
            timer_trap = 0
            timer_out = 0

    return idx_start, idx_end, trap_times

class ABP:
    """
    An ensemble of active Brownian particle submerged in a fluid with zero flow profile.
    Can be tested in either 2D or 3D.
    """

    def __init__(self, N, dt, d):
        """
        Initialise N realisations of the same particle at the origin with random orientations.
        
        Arguments:
            N: number of realisations of the particle
            dt: timestep
            d: dimensionality of the system
        """
        self.N = N
        self.dt = dt
        self.dim = d
        # Initialise orientation vector of each particle from a uniform rotationally symmetric distribution
        distribution = uniform_direction(d)
        self.e = distribution.rvs(N)
        # Initialise positions
        self.r = np.zeros([N, d])

    def Run(self, v, T, MSD, trajectory):
        """
        Run the simulation and display the desired results.
        
        Arguments:
            v: ABP velocity
            T: total number of timesteps
            MSD: display mean square displacement
            trajectory: display trajectory
        """
        # Run simulation and retrieve data
        pos, orient = run(self.N, self.r, self.e, v, self.dt, T, self.dim)
        # Calculate and display mean square displacement
        if MSD:
            self.MSD(pos, v, T)
        # Calculate and display trajectories
        if trajectory and (self.dim == 2):
            self.Trajectory(pos)


    def MSD(self, data, v, T):
        """
        Calculate the mean sqaure displacement of an ensemble of particles over time.
        Plot the results on a graph.

        Arguments:
            data: ABP position history
            v: ABP velocity
            T: total number of timesteps
        """
        # Calculate mean square displacement along each Cartesian direction
        msds = np.mean((data - data[0])**2, axis=1)
        # Calculate total mean square displacement, omitting initial position for logscale
        msd = np.sum(msds, axis=1)[1:]

        # Create array of timesteps
        t = np.arange(1, T + 1) * self.dt
        # Calculate persistence time
        tau = 1 / (self.dim - 1)
        # Theoretical mean square displacement
        msd_theory = 2 * self.dim * t + 2 * v**2 * tau * t - 2 * v**2 * tau**2 * (1 - np.exp(-t / tau))
        # Theoretical msd for ballistic and diffusive regimes
        msd_b = v**2 * t**2 + 2 * self.dim * t
        msd_d = (2 * self.dim + 2 * v**2 * tau) * t 

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
            plt.plot(x, y, color='black')
        
        plt.xlabel(r"$x$ position [$\sigma$]")
        plt.ylabel(r"$y$ position [$\sigma$]")
        plt.tight_layout()
        plt.show()

    def blk_sfc_swim(self, data, theta):
        """
        Split the swimming direction data based on the proximity of particles to the channel walls.
        
        Arguments:
            data: position history
            theta: flattened swimming direction data 
        
        Returns:
            theta_blk: swimming direction data in bulk
            theta_sfc: swimming direction data near surface
        """
        y = data[:-1, :, 1].flatten()
        bulk = (y >= 1.5) & (y <= (self.width - 1.5))
        theta_blk = theta[bulk]
        theta_sfc = theta[~bulk]
        return theta_blk, theta_sfc
    
    def blk_sfc_orient(self, data, theta):
        """
        Split the orientation data based on the proximity of particles to the channel walls.
        
        Arguments:
            data: position history
            theta: flattened orientation angle data 
        
        Returns:
            theta_blk: orientation angle data in bulk
            theta_sfc: orientation angle data near surface
        """
        y = data[:, :, 1].flatten()
        bulk = (y >= 1.5) & (y <= (self.width - 1.5))
        theta_blk = theta[bulk]
        theta_sfc = theta[~bulk]
        return theta_blk, theta_sfc


    def PDF(self, pos, orient):
        """
        Obtain the probability density functions in both positional and
        orientational space.

        Arguments:
            pos: position data
            orient: orientation data
        """
        # Select number of bins
        num_bins = 'auto'
        # Decorrelate position and orientation data
        p = pos[::self.step][10:-1]
        o = orient[::self.step][10:-1]
        # Get independent velocity data
        vx = self.get_xspeed(pos)
        vx_independent = vx[::self.step][10:]
        
        
        # Isolate vertical position from position data
        y = p[:, :, 1].flatten()

        # Construct spatial PDF
        pdf1, edges1 = np.histogram(y, bins=num_bins, density=True)

        # Find average velocity at each section along channel width
        mean_vx, _, _ = binned_statistic(y, vx_independent.flatten(), statistic='mean', bins=edges1)
        bin_centres = 0.5 * (edges1[:-1] + edges1[1:])

        fig, ax1 = plt.subplots(figsize=[8, 6])
        ax1.stairs(pdf1, edges1, color='black')
        ax1.set_title(f"Spatial PDF: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $D$ = {self.D}, $G$ = {self.G}")
        ax1.set_xlabel("$y/w$")
        ax1.set_ylabel("$P(y/w)$")
        ax1.set_xlim(0, 1)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:green'
        ax2.set_ylabel(r'$\langle v_x \rangle/v_0$', color=color)
        ax2.plot(bin_centres, mean_vx/self.Ps, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.tight_layout()

        # Isolate orientation angles from orientation data
        theta = np.arctan2(o[:, :, 1], o[:, :, 0]).flatten()

        # Find average orientation at each section along channel width
        mean_theta, _, _ = binned_statistic(y, theta, statistic='mean', bins=edges1)

        # Plot spatial PDF with average orientation overlaid
        fig, ax1 = plt.subplots(figsize=[8, 6])
        ax1.stairs(pdf1, edges1, color='black')
        ax1.set_title(f"Spatial PDF: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $D$ = {self.D}, $G$ = {self.G}")
        ax1.set_xlabel("$y/w$")
        ax1.set_ylabel("$P(y/w)$")
        ax1.set_xlim(0, 1)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel(r'$\langle \theta \rangle$', color=color)  # we already handled the x-label with ax1
        ax2.plot(bin_centres, mean_theta, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Construct orientational PDF
        pdf2, edges2 = np.histogram(theta, bins=num_bins, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf2, edges2, color='black')
        plt.title(f"Orientational PDF: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $D$ = {self.D}, $G$ = {self.G}")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Find particle orientations near the surface for upstream swimmers
        trap = self.trapping_index(p, vx_independent)
        theta_trap = theta[trap.flatten()]

        # Construct orientational PDF near the surface for upstream swimmers
        pdf3, edges3 = np.histogram(theta_trap, bins=num_bins, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf3, edges3, color='black')
        plt.title(f"Orientational PDF (trapped): $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $D$ = {self.D}, $G$ = {self.G}")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Find particle orientations near the surface
        wall = self.wall_index(p)
        theta_wall = theta[wall.flatten()]

        # Construct orientational PDF near the surface 
        pdf4, edges4 = np.histogram(theta_wall, bins=num_bins, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf4, edges4, color='black')
        plt.title(f"Orientational PDF (wall): $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $D$ = {self.D}, $G$ = {self.G}")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        # Find particle orientations in the bulk
        theta_bulk = theta[~wall.flatten()]

        # Construct orientational PDF in the bulk
        pdf5, edges5 = np.histogram(theta_bulk, bins=num_bins, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf5, edges5, color='black')
        plt.title(f"Orientational PDF (bulk): $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $D$ = {self.D}, $G$ = {self.G}")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.tight_layout()

        plt.show()

    def Displacements(self, pos, orient):
        """
        Display the longitudinal displacements for the trajectory of a single ABP over time.
        
        Arguments:
            pos: position history
        """
        # Calculate instantaneous displacements
        dx = np.append([0], pos[1:, 0, 0] - pos[:-1, 0, 0]).flatten()
        t = np.linspace(0, self.T * self.dt, self.T + 1)

        
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Longitudinal displacements: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\Delta x(t)/w$")
        plt.scatter(t, dx, color='black', s=1, marker='.', zorder=-1)

        if show_traps:
            # Calculate orientation angles
            ex, ey = orient[:, 0, 0], orient[:, 0, 1]
            theta = np.arctan2(ey, ex)
            idx_s, idx_e, _ = track_traps(pos[:, 0, 1], self.dt, theta)
            
            plt.axvline(t[idx_s][0], color='blue', label='start')
            for T in t[idx_s][1:]:
                plt.axvline(T, color='blue')
            plt.axvline(t[idx_e][0], color='orange', label='end')
            for T in t[idx_e][1:]:
                plt.axvline(T, color='orange')

        plt.axvline(tau, color='black', linestyle='dotted', label=r'$t=\tau_r$')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.show()


# Create a key value class
class keyvalue(argparse.Action):
    # Constructor calling
    def __call__( self , parser, namespace,
                 values, option_string = None):
        setattr(namespace, self.dest, dict())
        
        for value in values:
            # Split it into key and value
            key, value = value.split('=')
            if key == 'wall':
                value = float(value)
            # Assign into dictionary
            getattr(namespace, self.dest)[key] = value



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.01, help='Simulation timestep')
    parser.add_argument('-MSD', action='store_true', help='Find the mean square displacement')
    parser.add_argument('-trajectory', action='store_true', help='Find the particle trajectory')
    parser.add_argument('-t', type=int, default=10000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-v', type=float, default=1, help='ABP velocity')
    parser.add_argument('-d', choices=[2, 3], type=int, default=2, help='Dimensionality of the system')
    args = parser.parse_args()

    x = np.linspace(0, 4, 9)
    print(x)