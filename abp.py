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
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

plt.style.use('science')
plt.rcParams['text.usetex'] = False

# Developer tools ;)
d = 2
tau = 1 / (d - 1)
vorticity = 1
noise_r = 1
arrow_spacing = 100
centre_start = False
show_traps = False

@njit
def run(N, p, e, T, dt, Ps, D, Pf, G):
    """
    Run the simulation for T timesteps, recording all positions and orientations.

    Arguments:
        N: number of particles
        p: particle positions
        e: particle orientations
        T: total number of timesteps
        dt: timestep
        Ps: swim Peclet number
        D: diffusion number
        Pf: flow Peclet number
        G: elongation factor

    Returns:
        p_hist: history of positions
        o_hist: history of orientations
    """
    # Initialise arrays to keep track of position/orientation data
    p_hist = np.zeros((T+1, N, 2))
    o_hist = np.zeros((T+1, N, 2))
    # Store position and orientation of each particle
    for j in range(N):
        for k in range(2):
            p_hist[0, j, k] = p[j, k]
            o_hist[0, j, k] = e[j, k]
    # Perform T iterations of the update procedure
    for i in range(1, T+1):
        # Update position and orientation
        p_old = p.copy()
        e_old = e.copy()
        p, e = update(N, p_old, e_old, dt, Ps, D, Pf, G)
        # Store position and orientation of each particle
        for j in range(N):
            for k in range(2):
                p_hist[i, j, k] = p[j, k]
                o_hist[i, j, k] = e[j, k]
    return p_hist, o_hist   

@njit
def update(N, r, e, dt, Ps, D, Pf, G):
        """
        Update the positions and orientations of each particle.
        
        Arguments:
            N: number of particles
            r: particle positions
            e: particle orientations
            dt: timestep
            Ps: swim Peclet number
            D: effective diffusivity
            Pf: flow Peclet number
            G: elongation factor

        Returns:
            r_new: updated positions
            e_new: updated orientations
        """
        r_new = np.zeros_like(r)
        e_new = np.zeros_like(e)
        # Iterate over every particle
        for i in range(N):
            # Compute swim velocity term
            r_swim = dt * Ps * e[i] 
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
            e_new[i] = orientation(e[i], dt, Pf, r_new[i, 1], G)
        # Return updated position and orientation vectors
        return r_new, e_new

@njit
def orientation(e, dt, Pf, y, G):
    """
    Calculate and return the new orientation vector in 2D via rotation matrix update.
    
    Arguments:
        e: original orientation vector
        dt: timestep
        Pf: flow Peclet number
        y: height of particle within channel
        G: elongation factor
        
    Returns:
        the updated orientation vector
    """
    # Calculate change in orientation due to rotational noise
    d_theta_noise = noise_r * np.sqrt(2 * dt) * np.random.normal(0, 1) 
    # Calculate angle from x-axis of the original orientation vector
    theta = np.arctan2(e[1], e[0])
    # Calculate angular velocity due to vorticity and shear
    d_theta_omega = 2 * dt * Pf * (1 - 2 * y) * (G * np.cos(2 * theta) - vorticity)
    # Calculate new angle
    new_theta = theta + d_theta_noise + d_theta_omega
    # Return updated orientation vector
    return np.array([np.cos(new_theta), np.sin(new_theta)])

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

@njit
def positions_fptd(N):
    """
    Generate the positions of N particles along a vertical strip at x = 0.
    
    Arguments:
        N: number of particles
    
    Returns:
        r: positions of each particle
    """
    y_min = 0
    y_max = 1
    # Initialise positions
    r = np.zeros((N, 2))
    # Randomly choose x and y components within specified range
    for i in range(N):
        r[i, 1] = random.uniform(y_min, y_max)
    # Return position array
    return r

@njit
def trapping_times(y, orient, N, dt):
    """
    Get array of trapping times.
    
    Arguments:
        y: transverse position history
        orient: orientation history
        N: number of particles
        dt: timestep
    
    Returns:
        trap_times: array of trapping times
    """
    # Collect trap durations across all particles
    trap_times = np.empty(0)
    # Iterate over each particle
    for n in range(N):
        dx, dy = orient[:, n, 0], orient[:, n, 1]
        theta = np.arctan2(dy, dx)
        _, _, tt = track_traps(y[:, n], dt, theta)
        trap_times = np.append(trap_times, tt)
    # Return trapping times
    return trap_times

@njit
def track_traps(y, dt, theta):
    """
    Calculate trapping times within a single ABP trajectory.
    
    Arguments:
        y: single particle transverse trajectory
        dt: timestep
        theta: single particle orientation history
    
    Returns:
        idx_start: the indexes where the ABP enters the trapped state
        idx_end: the indexes where the ABP leaves the trapped state
        trap_times: the trapping times in a single trajectory
    """
    bottom = False
    top = False
    start = 0
    idx_start = np.full(len(y), False)
    idx_end = np.full(len(y), False)
    timer_trap = 0
    trap_times = []

    for i in range(1, len(y)):
        if y[i-1] == y[i] and timer_trap == 0:
            idx_start[i] = True
            timer_trap += 1
            if y[i] < 0.5:
                bottom = True
            else:
                top = True
        elif timer_trap > 0 and ((bottom and (np.pi/4 < theta[i] < 3*np.pi/4)) or (top and (-3*np.pi/4 < theta[i] < -np.pi/4))):
            idx_end[i - timer_trap + start] = True
            trap_times.append(start * dt)
            timer_trap = 0
            bottom = False
            top = False
            start = 0
        elif timer_trap > 0 and ((bottom and (0 < theta[i] < np.pi)) or (top and (-np.pi < theta[i] < 0))) and start == 0:
            start = timer_trap
            timer_trap += 1
        elif timer_trap > 0 and ((top and (0 <= theta[i] <= np.pi)) or (bottom and (-np.pi <= theta[i] <= 0))) and start > 0:
            start = 0
            timer_trap += 1
        elif timer_trap > 0:
            timer_trap += 1

    return idx_start, idx_end, trap_times

class ABP:
    """
    An ensemble of active Brownian particles experiencing Poisseuille flow in a confined geometry.
    """

    def __init__(self, N, T, dt, Ps, D, Pf, G, fptd=False):
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
            fptd: select true if collecting fptd data
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
        # Initialise orientation vector of each particle from a uniform rotationally symmetric distribution
        distribution = uniform_direction(2)
        self.e = distribution.rvs(N)
        # Initialise positions
        if centre_start:
            self.r = np.full((N, 2), [0, 0.5])
            self.e = np.full((N, 2), [1/np.sqrt(2), 1/np.sqrt(2)])
        elif fptd:
            self.r = positions_fptd(N)
        else:
            self.r = positions(N)
        
 
    def Run(self):
        """
        Run the simulation and return the results.
        
        Returns:
            pos: position history of each particle
            orient: orientation history of each particle
        """
        # Run simulation and retrieve data
        pos, orient = run(self.N, self.r, self.e, self.T, self.dt, self.Ps, self.D, self.Pf, self.G)
        return pos, orient       

    def get_MSD(self, data):
        """
        Calculate and return the mean square displacement.
        
        Arguments:
            data: position history
        
        Returns:
            msd_xy: the MSD along each Cartesian direction
            msd_tot: the total mean square displacement
        """
        # Calculate mean square displacement along each Cartesian direction
        msd_xy = np.mean((data[1:] - data[0])**2, axis=1)
        # Calculate total mean square displacement
        msd_tot = np.sum(msd_xy, axis=1)
        # Return MSDs
        return msd_xy, msd_tot
    
    def get_powerlaw(self, data):
        """
        Obtain the late-time power-law dependence of a dataset.
        
        Argument:
            data: dependent data
        
        Returns:
            a: fitted power
            b: fitted logarithm of prefactor
        """
        # Create array of measurement times
        t = np.arange(1, self.T + 1) * self.dt
        # Consider only late-time data, i.e. >> tau
        if self.T >= 1000000:
            late = t >= 100 * tau
        else:
            late = t >= 10 * tau
        y = np.log(data[late])
        x = np.log(t[late])
        # Fit to a 1st degree polynomial
        a, b = np.polyfit(x, y, 1)
        # return fitted parameters
        return a, b
    
    def get_xspeed(self, data):
        """
        Obtain the instantaneous velocity of each particle in the x direction over time.
        
        Arguments:
            data: position history
        
        Returns:
            instantaneous velocity of each particle in the x direction (2D array)
        """
        # Get x-position history
        x = data[:, :, 0]
        # Calculate instantaneous velocity
        dx = x[1:] - x[:-1]
        return dx / self.dt
    
    def get_yspeed(self, data):
        """
        Obtain the instantaneous velocity of each particle in the y direction over time.
        
        Arguments:
            data: position history
        
        Returns:
            instantaneous velocity of each particle in the y direction (T, N)
        """
        # Get x-position history
        y = data[:, :, 1]
        # Calculate instantaneous velocity
        dy = y[1:] - y[:-1]
        return dy / self.dt

    def MSD(self, data):
        """
        Calculate the mean sqaure displacement of an ensemble of particles over time.
        Plot the results on a graph.

        Arguments:
            data: position history
        """
        # Get MSD
        msd_xy, msd = self.get_MSD(data)

        # Create array of measurement times
        t = np.arange(1, self.T + 1) * self.dt
        # Theoretical mean square displacement
        msd_theory = 2 * d * self.D * t + 2 * self.Ps**2 * t - 2 * self.Ps**2 * (1 - np.exp(-t))
        # Theoretical msd for ballistic and diffusive regimes
        msd_b = self.Ps**2 * t**2 + 2 * d * self.D * t
        msd_d = 2 * t * (d * self.D + self.Ps**2)
        # Perform fit to late-time data
        a, b = self.get_powerlaw(msd)
        B = np.exp(b)
        t_fit = np.linspace(tau/self.dt, self.T, 100) * self.dt
        msd_fit = B * t_fit**a
        # Calculate MSD at t = 10*tau
        msd_fit_10 = B * (10*tau)**a

        # Plot MSD with fit and theory lines
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"MSD: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d, color='green', linestyle='--', label='diffusive')
        plt.loglog(t_fit, msd_fit, color='magenta', label=r'$\sim t^{\alpha}$')
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$t=\tau_r$')
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\langle (\Delta r)^2 \rangle/w^2$")
        if a < 1.5:
            plt.text(10*tau, 0.75*msd_fit_10, r'$\alpha$ = ' + f'{np.round(a, 2)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{np.round(B/2/d, 2)}', ha='left', va='top', fontsize=12)
        else:
            plt.text(10*tau, 0.75*msd_fit_10, r'$\alpha$ = ' + f'{np.round(a, 2)}\n' + r'$Pe_{s,\mathrm{eff}}$ = ' + f'{np.round(np.sqrt(B), 2)}', ha='left', va='top', fontsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()

        # Calculate MSD in the x-direction and line of best fit fit
        msd_x = msd_xy[:, 0]
        a_x, b_x = self.get_powerlaw(msd_x)
        B_x = np.exp(b_x)
        msd_x_fit = B_x * t_fit**a_x
        # Calculate MSD at t = 10*tau
        msd_x_fit_10 = B_x * (10*tau)**a_x

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"MSD$_x$: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.scatter(t, msd_x, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory/2, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b/2, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d/2, color='green', linestyle='--', label='diffusive')
        plt.loglog(t_fit, msd_x_fit, color='magenta', label=r'$\sim t^{\alpha}$')
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$t=\tau_r$')
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
        if a_x < 1.5:
            plt.text(10*tau, 0.75*msd_x_fit_10, r'$\alpha$ = ' + f'{np.round(a_x, 2)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{np.round(B_x/2, 2)}', ha='left', va='top', fontsize=12)
        else:
            plt.text(10*tau, 0.75*msd_x_fit_10, r'$\alpha$ = ' + f'{np.round(a_x, 2)}\n' + r'$Pe_{s,\mathrm{eff}}$ = ' + f'{np.round(np.sqrt(B_x), 2)}', ha='left', va='top', fontsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()

        # Calculate MSD in the y-direction and line of best fit fit
        msd_y = msd_xy[:, 1]
        a_y, b_y = self.get_powerlaw(msd_y)
        B_y = np.exp(b_y)
        msd_y_fit = B_y * t_fit**a_y
        # Calculate MSD at t = 10*tau
        msd_y_fit_10 = B_y * (10*tau)**a_y

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"MSD$_y$: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.scatter(t, msd_y, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory/2, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b/2, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d/2, color='green', linestyle='--', label='diffusive')
        plt.loglog(t_fit, msd_y_fit, color='magenta', label=r'$\sim t^{\alpha}$')
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$t=\tau_r$')
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\langle (\Delta y)^2 \rangle/w^2$")
        plt.text(10*tau, 0.5*msd_y_fit_10, r'$\alpha$ = ' + f'{np.round(a_y, 2)}', ha='left', va='top', fontsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()

        plt.show()

    def get_variance(self, x):
        """
        Return the variance of displacement along a single Cartesian direction.
        
        Arguments:
            x: position history in one direction
        
        Returns:
            the variance of the one-dimensional displacement over time
        """
        dx = x[1:] - x[0]
        return np.var(dx, axis=1)

    def Variance(self, data):
        """
        Compute the variance of the displacement in the x-direction. Display exponent
        of time-dependence (beta) and fitted diffusivity if beta < 1.2.
        
        Arguments:
            data: position history
        """
        # Get variance
        var = self.get_variance(data[:, :, 0])

        # Create array of measurement times
        t = np.arange(1, self.T + 1) * self.dt
        # Theoretical mean square displacement
        # Perform fit to variance
        a, b = self.get_powerlaw(var)
        # Perform fit to late-time data
        t_fit = np.linspace(tau/self.dt, self.T, 100) * self.dt
        var_fit = np.exp(b) * t_fit**a
        # Calculate diffusivity (x-direction)
        D_eff = np.round(np.exp(b) / 2, 2)
        # Calculate theoretical (diffusive) variance (divide by 2 for theory in one dimension)
        var_theory = (2 * d * (self.D + self.Ps**2 / 2) * t) / 2
        # Calculate variance at t = 10tau
        var_fit_10 = np.exp(b) * (10*tau)**a

        # Plot variance with fit, display parameters
        fig = plt.figure(figsize=[8, 6])
        plt.title(r"Var($\Delta x$): $l_p/w$ = " + f"{self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.scatter(t, var, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, var_theory, color='red', linestyle='--', label=r'$\sim t$')
        plt.loglog(t_fit, var_fit, color='magenta', label=r'$\sim t^{\beta}$')
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\langle (\Delta x - \langle \Delta x \rangle)^2 \rangle/w^2$")
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$t=\tau_r$')
        plt.text(10*tau, 0.75*var_fit_10, r'$\beta$ = ' + f'{np.round(a, 2)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{D_eff}', ha='left', va='top', fontsize=12)
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    def Trajectory(self, data, data1):
        """
        Plot the trajectory of a single particle from its position and orientation history. One
        graph will show the trajectory with the option to show traps, whilst the other includes
        arrows denoting the orientation of the particle every number of timesteps.
        
        Arguments:
            data: position history
            data1: orientation history
        """        
        # Consider only first particle
        x, y = data[:, 0, 0], data[:, 0, 1]
        start_x, start_y = data[0, 0, 0], data[0, 0, 1]
        end_x, end_y = data[-1, 0, 0], data[-1, 0, 1]
        # Calculate orientation angles
        dx, dy = data1[:, 0, 0], data1[:, 0, 1]
        theta = np.arctan2(dy, dx)

        # Plot regular trajectory with optional traps
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Trajectory: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")

        # Show start and end points of trajectory
        plt.scatter(start_x, start_y, color='lime', s=20, zorder=1)
        plt.scatter(end_x, end_y, color='red', s=20, zorder=1)

        if show_traps:
            idx_s, idx_e, _ = track_traps(y, self.dt, theta)
            # Show traps
            plt.scatter(x[idx_s], y[idx_s], color='cyan', s=15, zorder=1)
            plt.scatter(x[idx_e], y[idx_e], color='orange', s=15, zorder=1)

        #plt.scatter(x, y, color='black', marker='.', s=1, zorder=-1)
        plt.plot(x, y, color='black', zorder=-1)
        plt.xlabel(r"$x/w$")
        plt.ylabel(r"$y/w$")
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(1, color='black', linestyle='--', alpha=0.5)
        plt.tight_layout()

        # Show trajectory with orientation directions overlaid
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Trajectory: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        
        # Show start and end points
        plt.scatter(start_x, start_y, color='lime', s=20, zorder=1)
        plt.scatter(end_x, end_y, color='red', s=20, zorder=1)

        #plt.scatter(x, y, color='black', marker='.', s=1, zorder=-1)
        plt.plot(x, y, color='black', zorder=-1)
        plt.xlabel(r"$x/w$")
        plt.ylabel(r"$y/w$")
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(1, color='black', linestyle='--', alpha=0.5)
        # Create array of arrow directions
        dx, dy = dx[::arrow_spacing], dy[::arrow_spacing]
        # Create array of arrow bases
        X, Y = x[::arrow_spacing], y[::arrow_spacing]
        # Make quiver plot
        plt.quiver(X, Y, dx, dy, color='red', width=0.002, headwidth=3, headlength=4, scale=25, zorder=-1)
        
        plt.tight_layout()
        plt.show()
    
    def FPTD(self, data, target):
        """
        Obtain the first-passage time distribution of an ensemble of particles reaching a target along
        the longitudinal direction.
        
        Arguments:
            data: ABP position history
            target: distance to check for first passage time
        """
        # Get all x positions
        x = data[:, :, 0]
        # Define condition for particle crossing the target
        if target > 0:
            crossed = x >= target
        else:
            crossed = x <= target
        # Identify all particles that successfully crossed the target
        has_crossed = np.any(crossed, axis=0)
        # Obtain first crossing of each particle
        first = np.argmax(crossed, axis=0)
        # Discard all unsuccessful attempts and convert to time units
        fpt = first[has_crossed] * self.dt
        success_rate = np.round(len(fpt) / self.N * 100, 1)
        # Save data to file
        filename = f'fptd data/fptd_x{target}_N{self.N}_{self.Ps}_{np.round(self.Pf/self.Ps, 6)}_{self.G}.txt'
        with open(filename, 'w') as f:
            f.write(f"# lp/w = {self.Ps}\n")
            f.write(f"# Pf/Ps = {np.round(self.Pf/self.Ps, 6)}\n")
            f.write(f"# G = {self.G}\n")
            f.write(f"# target = {target}\n")
            f.write(f"# success rate = {success_rate}\n")
            np.savetxt(f, fpt)

    def initial_config(self):
        """
        View the initial configuration, consisting of the positions and orientations 
        of all N particles.
        """
        fig = plt.figure(figsize=[8, 6])
        plt.title("Initial Configuration")
        plt.xlabel(r"$x/w$")
        plt.ylabel(r"$y/w$")
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(1, color='black', linestyle='--', alpha=0.5)
        # Plot position and orientation of each particle
        for i in range(self.N):
            x, y = self.r[i, 0], self.r[i, 1]
            dx, dy = self.e[i, 0], self.e[i, 1]
            plt.scatter(x, y, color='black', s=20, zorder=1)
            plt.quiver(x, y, dx, dy, color='red', width=0.002, headwidth=3, headlength=4, scale=25, zorder=-1)
        plt.tight_layout()
        plt.show()

    def trapping_index(self, p, vx):
        """
        Obtain the indices for proximity to the surface during upstream swimming.
        
        Arguments:
            p: reduced particle position history
            vx: particle x-velocity history
        
        Returns:
            indices of particle data in 'trapped' state
        """
        # Extract y-position history
        y = p[:, :, 1]
        # Return trapping condition
        return (vx < 0) & ((y < 0.05) | (y > 0.95))
    
    def wall_index(self, p):
        """
        Obtain the indices for proximity to the surface.
        
        Arguments:
            p: reduced particle position history
        
        Returns:
            indices of particle data near wall
        """
        # Extract y-position history
        y = p[:, :, 1]
        # Return wall condition
        return ((y < 0.05) | (y > 0.95))

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
        ax1.set_title(f"Spatial PDF: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
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
        mean_theta, _, _ = binned_statistic(y, np.abs(theta), statistic='mean', bins=edges1)

        # Plot spatial PDF with average orientation overlaid
        fig, ax1 = plt.subplots(figsize=[8, 6])
        ax1.stairs(pdf1, edges1, color='black')
        ax1.set_title(f"Spatial PDF: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        ax1.set_xlabel("$y/w$")
        ax1.set_ylabel("$P(y/w)$")
        ax1.set_xlim(0, 1)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel(r'$\langle |\theta| \rangle$', color=color)  # we already handled the x-label with ax1
        ax2.plot(bin_centres, mean_theta, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
        plt.tight_layout()

        # Find particle orientations near the surface
        wall = self.wall_index(p)
        theta_wall = theta[wall.flatten()]
        # Construct orientational PDF near the surface 
        pdf4, edges4 = np.histogram(theta_wall, bins=num_bins, density=True)

        # Find particle orientations in the bulk
        theta_bulk = theta[~wall.flatten()]
        # Construct orientational PDF in the bulk
        pdf5, edges5 = np.histogram(theta_bulk, bins=num_bins, density=True)

        # Plot both orientational distributions together
        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf4, edges4, color='red', label='wall')
        plt.stairs(pdf5, edges5, color='blue', label='bulk')
        plt.title(f"Orientational PDF: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.xlabel(r"$\theta$")
        plt.ylabel(r"$P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.legend(loc='upper center')
        plt.tight_layout()

        # Construct instantaneous velocity PDF
        pdf2, edges2 = np.histogram(vx_independent.flatten()/self.Ps, bins=num_bins, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf2, edges2, color='black')
        plt.title(f"Velocity PDF: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.xlabel("$v_x/v_0$")
        plt.ylabel("$P(v_x/v_0)$")
        plt.tight_layout()

        plt.show()

    def TTD(self, pos, orient):
        """
        Record the trapping time distribution.
        
        Arguments:
            pos: position history
            orient: orientation history
        """
        # Separate transverse coordinate from positions
        y = pos[:, :, 1]
        # Get array of trapping times
        tt = trapping_times(y, orient, self.N, self.dt)
        # Save results to data file
        if vorticity == 1:
            filename = f'ttd data/ttd_N{self.N}_{self.Ps}_{np.round(self.Pf/self.Ps, 6)}_{self.G}.txt'
        else:
            filename = f'ttd data/ttd_N{self.N}_{self.Ps}_{np.round(self.Pf/self.Ps, 6)}_{self.G}_NV.txt'
        with open(filename, 'w') as f:
            f.write(f"# lp/w = {self.Ps}\n")
            f.write(f"# Pf/Ps = {np.round(self.Pf/self.Ps, 6)}\n")
            f.write(f"# G = {self.G}\n")
            np.savetxt(f, tt)




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('--MSD', action='store_true', help='Obtain the mean square displacement')
    parser.add_argument('--trajectory', action='store_true', help='Display the particle trajectory')
    parser.add_argument('--PDF', action='store_true', help='Obtain the probability density functions')
    parser.add_argument('--variance', action='store_true', help='Obtain the variance of longitudinal displacement')
    parser.add_argument('--FPTD', action='store_true', help='Obtain first-passage time distribution')
    parser.add_argument('-T', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-Ps', type=float, default=5, help='Swim Peclet number')
    parser.add_argument('-Pf', type=float, default=5, help='Flow Peclet number')
    parser.add_argument('-D', type=float, default=0.01, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-x', type=float, default=10, help='Distance along x-axis to check for first-passage')
    parser.add_argument('-G', type=float, default=0, help='Geometrical factor related to particle aspect ratio')
    parser.add_argument('--init', action='store_true', help='View the initial configuration of the system')
    parser.add_argument('--TTD', action='store_true', help='Obtain the trapping time distribution')
    parser.add_argument('--DX', action='store_true', help='Display the longitudinal displacements over time')
    args = parser.parse_args()

    abp = ABP(args.N, args.T, args.dt, args.Ps, args.D, args.Pf, args.G, args.FPTD)

    # Display initial configuration
    if args.init:
        abp.initial_config()
    # Run simulation
    pos, orient = abp.Run()
    # Perform select statistics
    if args.trajectory:
        abp.Trajectory(pos, orient)
    if args.DX:
        abp.Displacements(pos, orient)
    if args.MSD:
        abp.MSD(pos)
    if args.variance:
        abp.Variance(pos)
    if args.PDF:
        abp.PDF(pos, orient)
    if args.TTD:
        abp.TTD(pos, orient)
    if args.FPTD:
        abp.FPTD(pos, args.x)