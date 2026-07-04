import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scienceplots
from scipy.stats import uniform_direction
import argparse
import random

plt.style.use('science')
plt.rcParams['text.usetex'] = False

# Developer tools ;)
d = 2
noise_t = 1
noise_r = 1
use_arrows = False
centre_start = False

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
def update(N, p, e, dt, Ps, D, Pf, G):
        """
        Update the positions and orientations of each particle.
        
        Arguments:
            N: number of particles
            p: particle positions
            e: particle orientations
            dt: timestep
            Ps: swim Peclet number
            D: diffusion number
            Pf: flow Peclet number
            G: elongation factor

        Returns:
            r_new: updated positions
            e_new: updated orientations
        """
        p_new = np.zeros_like(p)
        e_new = np.zeros_like(e)
        # Iterate over every particle
        for i in range(N):
            # Compute swim velocity term
            p_swim = dt * Ps * e[i] 
            # Generate translational noise term
            p_noise = noise_t * np.sqrt(2 * dt) * D * np.random.normal(0, 1, 2) 
            # Update position via forward difference scheme
            p_new[i] = p[i] + p_swim + p_noise
            # Incorporate correction due to fluid flow
            p_new[i, 0] += 4 * dt * Pf * p[i, 1] * (1 - p[i, 1]) 
            # Impose reflection at boundaries
            if p_new[i, 1] <= 0:
                p_new[i, 1] *= -1
            elif p_new[i, 1] >= 1:
                p_new[i, 1] = 2 - p_new[i, 1]
            # Update orientation
            e_new[i] = orientation(e[i], dt, Pf, p[i, 1], G)
        # Return updated position and orientation vectors
        return p_new, e_new

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
    d_theta_omega = 2 * dt * Pf * (1 - 2 * y) * (G * np.cos(2 * theta) - 1)
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
    x_min = 0
    x_max = 0.5
    y_min = 0
    y_max = 1
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
        # Initialise orientation vector of each particle from a uniform rotationally symmetric distribution
        distribution = uniform_direction(2)
        self.e = distribution.rvs(N)
        # Initialise positions
        if centre_start:
            self.r = np.full((N, 2), [0, 0.5])
            self.e = np.full((N, 2), [1/np.sqrt(2), 1/np.sqrt(2)])
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
        msd_xy = np.mean((data - data[0])**2, axis=1)
        # Calculate total mean square displacement
        msd_tot = np.sum(msd_xy, axis=1)
        # Delete first data point to avoid conflict with logscale
        return msd_xy[1:], msd_tot[1:]
    
    def get_MSD_fit(self, msd):
        """
        Obtain the power-law dependence of the late-time mean square displacement.
        
        Argument:
            msd: mean square displacement data
        
        Returns:
            a: fitted power
            b: fitted logarithm of prefactor
        """
        # Create array of measurement times
        t = np.arange(1, self.T + 1) * self.dt
        # Calculate persistence time
        tau = 1 / (d - 1) 
        # Consider only late-time data, i.e. >> tau
        late = t >= 10 * tau
        y = np.log(msd[late])
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
        msd_xy, msd = self.get_MSD(data)

        # Create array of measurement times
        t = np.arange(1, self.T + 1) * self.dt
        # Calculate persistence time
        tau = 1 / (d - 1) 
        # Theoretical mean square displacement
        msd_theory = 2 * d * self.D**2 * t + 2 * self.Ps**2 * tau * t - 2 * self.Ps**2 * tau**2 * (1 - np.exp(-t / tau))
        # Theoretical msd for ballistic and diffusive regimes
        msd_b = self.Ps**2 * t**2 + 2 * d * self.D**2 * t
        msd_d = 2 * t * (d * self.D**2 + self.Ps**2 * tau)
        # Perform fit to late-time data
        a, b = self.get_MSD_fit(msd)
        t_fit = np.linspace(tau/self.dt, self.T, 100) * self.dt
        msd_fit = np.exp(b) * t_fit**a
        # Plot MSD with fit and theory lines
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Mean Square Displacement: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d, color='green', linestyle='--', label='diffusive')
        plt.loglog(t_fit, msd_fit, color='magenta', label=r'$\alpha$ = ' + f'{np.round(a, 2)}')
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$\tau_r$')
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\langle (r/w)^2 \rangle$")
        plt.legend()

        # Calculate MSD in the x-direction and line of best fit fit
        msd_x = msd_xy[:, 0]
        a_x, b_x = self.get_MSD_fit(msd_x)
        msd_fit_x = np.exp(b_x) * t_fit**a_x

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Longitudinal MSD: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.scatter(t, msd_x, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory/2, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b/2, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d/2, color='green', linestyle='--', label='diffusive')
        plt.loglog(t_fit, msd_fit_x, color='magenta', label=r'$\alpha$ = ' + f'{np.round(a_x, 2)}')
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$\tau_r$')
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\langle (x/w)^2 \rangle$")
        plt.legend()

        # Calculate MSD in the y-direction and line of best fit fit
        msd_y = msd_xy[:, 1]
        a_y, b_y = self.get_MSD_fit(msd_y)
        msd_fit_y = np.exp(b_y) * t_fit**a_y

        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Transverse MSD: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.scatter(t, msd_y, color='black', marker='.', s=10, label='simulation')
        plt.loglog(t, msd_theory/2, color='red', linestyle='--', label='theory')
        plt.loglog(t, msd_b/2, color='blue', linestyle='--', label='ballistic')
        plt.loglog(t, msd_d/2, color='green', linestyle='--', label='diffusive')
        plt.loglog(t_fit, msd_fit_y, color='magenta', label=r'$\alpha$ = ' + f'{np.round(a_y, 2)}')
        plt.axvline(tau, color='black', linestyle='dotted', label=r'$\tau_r$')
        plt.xlabel("$tD_r$")
        plt.ylabel(r"$\langle (y/w)^2 \rangle$")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def Trajectory(self, data, data1):
        """
        Retrieve the positions of an ensemble of particles over time, as well
        as their orientations at intervals during the first 10% of timesteps.
        Plot results on a graph.
        
        Arguments:
            data: position history
            data1: orientation history
        """
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"Particle trajectory: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        
        # Consider only first particle
        x, y = data[:, 0, 0], data[:, 0, 1]
        start_x, start_y = data[0, 0, 0], data[0, 0, 1]
        plt.scatter(start_x, start_y, color='lime', s=20, zorder=1)
        end_x, end_y = data[-1, 0, 0], data[-1, 0, 1]
        plt.scatter(end_x, end_y, color='red', s=20, zorder=1)
        plt.scatter(x, y, color='black', marker='.', s=1, zorder=-1)
        plt.xlabel(r"$x/w$")
        plt.ylabel(r"$y/w$")
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(1, color='black', linestyle='--', alpha=0.5)
        # Plot early-time orientations along trajectory
        if use_arrows:
            # Create array of arrow directions
            limit = int(self.T / 10)
            dx, dy = data1[:limit, 0, 0], data1[:limit, 0, 1]
            spacing = max(100, limit // 50)
            dx, dy = dx[::spacing], dy[::spacing]
            # Create array of arrow bases
            X, Y = x[:limit], y[:limit]
            X, Y = X[::spacing], Y[::spacing]
            # Make quiver plot
            plt.quiver(X, Y, dx, dy, color='red', width=0.002, headwidth=3, headlength=4, scale=25, zorder=-1)
        plt.tight_layout()
        plt.show()
    
    def FPTD(self, data, target):
        """
        Obtain the first-passage time distribution for an ensemble of particles
        to reach a point x along the x-axis.
        Display results as a histogram.
        
        Arguments:
            data: ABP position history
            target: distance to check for first passage
        """
        # Get all x positions
        x = data[:, :, 0]
        # Define condition for particle crossing the target
        crossed = x >= target
        # Identify all particles that successfully crossed the target
        has_crossed = np.any(crossed, axis=0)
        # Obtain first crossing of each particle
        first = np.argmax(crossed, axis=0)
        # Discard all unsuccessful attempts and convert to time units
        fpt = first[has_crossed] * self.dt
        success_rate = np.round(len(fpt) / self.N * 100, 1)
        # Build histogram of the FPTD
        fig = plt.figure(figsize=[8, 6])
        plt.hist(fpt, bins='auto')
        plt.title(f"First-Passage Time Distribution\nTarget: {target}" + r"$\sigma$, " + f"Success Rate: {success_rate}%")
        plt.xlabel("time [$1/D_r$]")
        plt.ylabel("crossings")
        plt.tight_layout()
        plt.show()

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
        # Extract y-position history and align with shape of x-velocity array
        y = p[:, :, 1]
        # Return trapping condition
        return (vx < 0) & ((y < 0.05) | (y > 0.95))

    def PDF(self, pos, orient):
        """
        Obtain the probability density functions in both positional and
        orientational space.

        Arguments:
            pos: position data
            orient: orientation data
        """
        # Decorrelate position and orientation data
        p = pos[::self.step][10:-1]
        o = orient[::self.step][10:-1]
        
        # Isolate vertical position from position data
        y = p[:, :, 1].flatten()

        # Construct positional PDF
        pdf1, edges1 = np.histogram(y, bins=50, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf1, edges1, color='black')
        plt.title(f"Spatial distribution: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.xlabel("height along channel, $y/w$")
        plt.ylabel("probability density, $P(y/w)$")
        plt.xlim(0, 1)

        # Isolate orientation angles from orientation data
        theta = np.arctan2(o[:, :, 1], o[:, :, 0]).flatten()

        # Construct orientational PDF
        pdf2, edges2 = np.histogram(theta, bins=50, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf2, edges2, color='black')
        plt.title(f"Orientational distribution: $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.xlabel(r"orientation angle, $\theta$")
        plt.ylabel(r"probability density, $P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

        # Find particle orientations near the surface
        vx = self.get_xspeed(pos)
        vx_independent = vx[::self.step][10:]
        trap = self.trapping_index(p, vx_independent)
        theta_sfc = theta[trap.flatten()]

        # Construct orientational PDF near the surface
        pdf3, edges3 = np.histogram(theta_sfc, bins=50, density=True)

        fig = plt.figure(figsize=[8, 6])
        plt.stairs(pdf3, edges3, color='black')
        plt.title(f"Orientational distribution (trapped): $l_p/w$ = {self.Ps}, $Pe_f/Pe_s$ = {np.round(self.Pf/self.Ps, 6)}, $G$ = {self.G}")
        plt.xlabel(r"orientation angle, $\theta$")
        plt.ylabel(r"probability density, $P(\theta)$")
        plt.xlim(-np.pi, np.pi)
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, default=100, help='Number of realisations of the ABP')
    parser.add_argument('-dt', type=float, default=0.001, help='Simulation timestep')
    parser.add_argument('--MSD', action='store_true', help='Obtain the mean square displacement')
    parser.add_argument('--trajectory', action='store_true', help='Display the particle trajectory')
    parser.add_argument('--PDF', action='store_true', help='Obtain the probability density functions')
    parser.add_argument('--FPTD', action='store_true', help='Obtain first-passage time distribution')
    parser.add_argument('-T', type=int, default=100000, help='Number of timesteps over which to run the simulation')
    parser.add_argument('-Ps', type=float, default=5, help='Swim Peclet number')
    parser.add_argument('-Pf', type=float, default=5, help='Flow Peclet number')
    parser.add_argument('-D', type=float, default=1, help='Dimensionless ratio of diffusion constants')
    parser.add_argument('-x', type=float, default=50, help='Distance along x-axis to check for first-passage')
    parser.add_argument('-G', type=float, default=0, help='Geometrical factor related to particle aspect ratio')
    parser.add_argument('--init', action='store_true', help='View the initial configuration of the system')
    args = parser.parse_args()

    abp = ABP(args.N, args.T, args.dt, args.Ps, args.D, args.Pf, args.G)

    if args.init:
        abp.initial_config()

    pos, orient = abp.Run()

    if args.trajectory:
        abp.Trajectory(pos, orient)
    if args.MSD:
        abp.MSD(pos)
    if args.FPTD:
        abp.FPTD(pos, args.x)
    if args.PDF:
        abp.PDF(pos, orient)