import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import argparse
from matplotlib.legend_handler import HandlerTuple

plt.style.use('science')
plt.rcParams['text.usetex'] = False

d = 2
D = 0.01

def plot_MSD(G, Ps, Pf, offset):
    """
    Plot the mean square displacement for a given set of parameters.

    Arguments:
        G: elongation factor
        Ps: swim Peclet number
        Pf: flow Peclet number
        offset: number of skipped logscale blocks in data
    """
    # Resolve filepath
    filename = f"G {G} NV results/MSD o{offset}/{Ps} {Pf}.txt"
    t, msd = np.loadtxt(filename, unpack=True)

    # Obtain theory curves
    msd_theory = 2 * d * D * t + 2 * float(Ps)**2 * t - 2 * float(Ps)**2 * (1 - np.exp(-t))
    # Theoretical msd for ballistic and diffusive regimes
    msd_b = float(Ps)**2 * t**2 + 2 * d * D * t
    msd_d = 2 * t * (d * D + float(Ps)**2)

    # Fit powerlaw to late-time data
    a, b = get_powerlaw(t, msd)
    B = np.exp(b)
    # Define x-axis data for fitted curve
    t_fit = np.linspace(100, 10000, 50)
    msd_fit = B * t_fit**a
    # Calculate MSD at t = 1000tau
    msd_fit_1000 = B * (1000)**a


    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$: $Pe_s$ = {Ps}, $Pe_f$ = {Pf}, $G$ = {G}")
    plt.scatter(t, msd, color='black', marker='.', s=10, label='simulation')
    plt.loglog(t, msd_theory, color='red', linestyle='--', label='theory (no flow)')
    plt.loglog(t, msd_b, color='blue', linestyle='--', label='ballistic limit')
    plt.loglog(t, msd_d, color='green', linestyle='--', label='diffusive limit')
    plt.axvline(1, color='black', linestyle='dotted', label=r'$t=\tau_r$')
    plt.xlabel(r"$t/\tau_r$")
    plt.ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    plt.xscale('log')
    plt.yscale('log')

    # Add fitted parameters
    plt.loglog(t_fit, msd_fit, color='magenta', label=r'$\sim t^{\alpha}$')
    if a < 1.5:
        plt.text(1000, 0.75*msd_fit_1000, r'$\alpha$ = ' + f'{np.round(a, 3)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{np.round(B/2, 3)}', ha='left', va='top', fontsize=12)
    else:
        plt.text(1000, 0.75*msd_fit_1000, r'$\alpha$ = ' + f'{np.round(a, 3)}\n' + r'$Pe_{s,\mathrm{eff}}$ = ' + f'{np.round(np.sqrt(B), 3)}', ha='left', va='top', fontsize=12)

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_variance(G, Ps, Pf, offset):
    """
    Plot the variance of displacement for a given set of parameters.

    Arguments:
        G: elongation factor
        Ps: swim Peclet number
        Pf: flow Peclet number
        offset: number of skipped logscale blocks in data
    """
    # Resolve filepath
    filename = f"G {G} results/variance o{offset}/{Ps} {Pf}.txt"
    t, var = np.loadtxt(filename, unpack=True)

    # Calculate theoretical (diffusive) variance (divide by 2 for theory in one dimension)
    var_theory = (2 * d * (D + float(Ps)**2 / 2) * t) / 2

    # Fit powerlaw to late-time data
    a, b = get_powerlaw(t, var)
    B = np.exp(b)
    # Define x-axis data for fitted curve
    t_fit = np.linspace(100, 10000, 50)
    # Perform fit to late-time data
    var_fit = B * t_fit**a
    # Calculate diffusivity (x-direction)
    D_eff = np.round(B / 2, 3)
    # Calculate variance at t = 1000tau
    var_fit_1000 = B * (1000)**a

    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(r"Var($\Delta x$): " + f"$Pe_s$ = {Ps}, $Pe_f$ = {Pf}, $G$ = {G}")
    plt.scatter(t, var, color='black', marker='.', s=10, label='simulation')
    plt.loglog(t, var_theory, color='red', linestyle='--', label=r'$\sim t$')
    plt.axvline(1, color='black', linestyle='dotted', label=r'$t=\tau_r$')
    plt.xlabel("$tD_r$")
    plt.ylabel(r"$\langle (\Delta x - \langle \Delta x \rangle)^2 \rangle/w^2$")
    plt.xscale('log')
    plt.yscale('log')

    # Add fitted parameters
    plt.loglog(t_fit, var_fit, color='magenta', label=r'$\sim t^{\beta}$')
    plt.text(1000, 0.75*var_fit_1000, r'$\beta$ = ' + f'{np.round(a, 3)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{D_eff}', ha='left', va='top', fontsize=12)
    
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_MSD3(G, filename1, filename2, filename3, offset):
    """
    Plot three MSDs along a line of constant swim Peclet number in phase-space.

    Arguments:
        G: elongation factor
        filename1: incomplete filepath to first dataset
        filename2: incomplete filepath to second dataset
        filename3: incomplete filepath to third dataset
        offset: number of skipped logscale blocks in data
    """
    # Define function to read in MSD data
    def read_data(filename):
            # Read parameters and trapping/bulk times from datafile
            Ps, string = filename.split(' ')
            Pf = string.split('.txt')[0]
            t, msd = np.loadtxt(f"G {G} results/MSD o{offset}/{filename}", dtype=float, unpack=True)
            return Ps, Pf, t, msd
    
    # Read parameters and mean square displacements from datafiles
    Ps, Pf1, t, msd1 = read_data(filename1)
    _, Pf2, _, msd2 = read_data(filename2)
    _, Pf3, _, msd3 = read_data(filename3)

    # Obtain theory curves
    msd_theory = 2 * d * D * t + 2 * float(Ps)**2 * t - 2 * float(Ps)**2 * (1 - np.exp(-t))
    # Theoretical msd for ballistic and diffusive regimes
    msd_b = float(Ps)**2 * t**2 + 2 * d * D * t
    msd_d = 2 * t * (d * D + float(Ps)**2)

    # Fit powerlaw to late-time data for each curve
    a1, b1 = get_powerlaw(t, msd1)
    a2, b2 = get_powerlaw(t, msd2)
    a3, b3 = get_powerlaw(t, msd3)
    B1 = np.exp(b1)
    B2 = np.exp(b2)
    B3 = np.exp(b3)
    # Define x-axis data for fitted curve
    t_fit = np.linspace(10, 10000, 50)
    msd_fit1 = 0.1 * B1 * t_fit**a1
    msd_fit2 = 0.1 * B2 * t_fit**a2
    msd_fit3 = 0.1 * B3 * t_fit**a3
    # Calculate locations of fitted parameter texts from lowest peak to highest peak
    text_order = np.argsort([np.max(msd_fit1), np.max(msd_fit2), np.max(msd_fit3)])
    textx = np.array([10, 100, 1000], dtype=float)
    curve_data = [(a1, B1, msd_fit1), (a2, B2, msd_fit2), (a3, B3, msd_fit3)]
    text_positions = {}
    for rank, curve_idx in enumerate(text_order):
        a, B, _ = curve_data[curve_idx]
        x = textx[rank]
        y = 0.1 * B * x**a
        text_positions[curve_idx] = (x, y)


    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$: $Pe_s$ = {Ps}, $G$ = {G}")
    p1 = plt.scatter(t, msd1, color='green', marker='.', s=10, zorder=0)
    p2 = plt.scatter(t, msd2, color='red', marker='.', s=10, zorder=0)
    p3 = plt.scatter(t, msd3, color='blue', marker='.', s=10, zorder=0)
    p4, = plt.loglog(t, msd_theory, color='black', linestyle='--', zorder=1)
    p5, = plt.loglog(t, msd_b, color='orange', linestyle='--', zorder=-1)
    p6, = plt.loglog(t, msd_d, color='dodgerblue', linestyle='--', zorder=-1)
    plt.axvline(1, color='black', linestyle='dotted')
    plt.text(1+1e-1, 1e-5, r'$t=\tau_r$', ha='left', va='bottom', fontsize=12)
    plt.xlabel(r"$t/\tau_r$")
    plt.ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    plt.xscale('log')
    plt.yscale('log')

    # Add fitted parameters
    p7, = plt.loglog(t_fit, msd_fit1, color='green')
    x1, y1 = text_positions[0]
    if a1 < 1.5:
        plt.text(x1, y1, r'$\alpha$ = ' + f'{np.round(a1, 3)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{np.round(B1/2, 3)}', ha='left', va='top', fontsize=12, color='green')
    else:
        plt.text(x1, y1, r'$\alpha$ = ' + f'{np.round(a1, 3)}\n' + r'$Pe_{s,\mathrm{eff}}$ = ' + f'{np.round(np.sqrt(B1), 3)}', ha='left', va='top', fontsize=12, color='green')
    p8, = plt.loglog(t_fit, msd_fit2, color='red')
    x2, y2 = text_positions[1]
    if a2 < 1.5:
        plt.text(x2, y2, r'$\alpha$ = ' + f'{np.round(a2, 3)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{np.round(B2/2, 3)}', ha='left', va='top', fontsize=12, color='red')
    else:
        plt.text(x2, y2, r'$\alpha$ = ' + f'{np.round(a2, 3)}\n' + r'$Pe_{s,\mathrm{eff}}$ = ' + f'{np.round(np.sqrt(B2), 3)}', ha='left', va='top', fontsize=12, color='red')
    p9, = plt.loglog(t_fit, msd_fit3, color='blue')
    x3, y3 = text_positions[2]
    if a3 < 1.5:
        plt.text(x3, y3, r'$\alpha$ = ' + f'{np.round(a3, 3)}\n' + r'$D_{\mathrm{eff}}$ = ' + f'{np.round(B3/2, 3)}', ha='left', va='top', fontsize=12, color='blue')
    else:
        plt.text(x3, y3, r'$\alpha$ = ' + f'{np.round(a3, 3)}\n' + r'$Pe_{s,\mathrm{eff}}$ = ' + f'{np.round(np.sqrt(B3), 3)}', ha='left', va='top', fontsize=12, color='blue')

    # Create legend
    l = plt.legend([p1, p2, p3, p4, p5, p6, (p7, p8, p9)], 
                   [f'$Pe_f$ = {Pf1}', f'$Pe_f$ = {Pf2}', f'$Pe_f$ = {Pf3}', 'theory (no flow)', 'ballistic limit', 'diffusive limit', r'$\sim t^{\alpha}$'],
                    handler_map={tuple: HandlerTuple(ndivide=None)})

    plt.tight_layout()
    plt.show()

def get_powerlaw(t, msd):
    """
    Obtain the late-time power-law dependence of the MSD.
    
    Argument:
        t: measurement times
        msd: mean square displacements
    
    Returns:
        a: fitted power
        b: fitted logarithm of prefactor
    """
    # Consider only very late times
    late = t >= 1000
    # Take the logarithms of time and MSD
    y = np.log(msd[late])
    x = np.log(t[late])
    # Fit to a 1st degree polynomial
    a, b = np.polyfit(x, y, 1)
    # return fitted parameters
    return a, b

def plot_all_MSD(G, Ps_list, Pf_list, offset, old):
    """
    Plot all MSDs from a list of swim and flow Peclet numbers on one graph.
    
    Arguments:
        G: elongation factor
        Ps_list: list of swim Peclet numbers
        Pf_list: list of flow Peclet numbers
        offset: skipped logscale blocks
        old: True if MSDs are to be plotted using a single origin
    """
    # Resolve folder path
    if old:
        folder = f"G {G} results/MSD old o{offset}"
    else:
        folder = f"G {G} results/MSD o{offset}"
    # Retrieve Peclet numbers to plot
    Ps = np.loadtxt(Ps_list, dtype=str)
    Pf = np.loadtxt(Pf_list, dtype=str)

    # Generate colours for plotting
    cmap = plt.get_cmap("rainbow")
    samples = np.linspace(0, 1, len(Ps))
    colours = cmap(samples)

    # Set up the figure
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$ comparison: $G$ = {G}, offset = {offset} blocks")
    plt.axvline(1, color='black', linestyle='dotted')
    plt.text(1+1e-1, 1e-5, r'$t=\tau_r$', ha='left', va='bottom', fontsize=12)
    plt.xlabel(r"$t/\tau_r$")
    plt.ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    plt.xscale('log')
    plt.yscale('log')

    # Initialise loop index
    i = -1
    # Iterate over each MSD file
    for ps, pf in zip(Ps, Pf):
        # Increment loop index
        i += 1
        # Read MSD from file
        file = f"{folder}/{ps} {pf}.txt"
        t, msd = np.loadtxt(file, unpack=True)
        # Add MSD to figure
        plt.scatter(t, msd, color=colours[i], marker='.', s=10, label=f'$Pe_s$ = {ps}, $Pe_f$ = {pf}')

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_pp_MSD(G, Ps, Pf, sample):
    """
    Plot the individual MSDs over a sample of trajectories. Plot includes both full
    MSD and zoom at long times (t > 10^3).
    
    Arguments:
        G: elongation factor
        Ps: swim Peclet number
        Pf: flow Peclet number
        sample: number of trajectories to plot
    """
    # Resolve folder filepath
    filename = f"G {G} NV results/MSD ind/{Ps} {Pf} n{sample}.npz"
    # Retrieve MSD data
    data = np.load(filename)
    t = data['time']
    msds = data['MSD']

    # Generate colours for plotting
    cmap = plt.get_cmap("rainbow")
    samples = np.linspace(0, 1, sample)
    colours = cmap(samples)

    # Set up the figure
    fig, ax = plt.subplots(1, 2, figsize=[12, 6])
    fig.suptitle(f"MSD$_x$ (per trajectory): $Pe_s$ = {Ps}, $Pe_f$ = {Pf}, $G$ = {G}")
    ax[0].set_title('full trajectory')
    ax[0].axvline(1, color='black', linestyle='dotted')
    ax[0].text(1+1e-1, 1e-5, r'$t=\tau_r$', ha='left', va='bottom', fontsize=12)
    ax[0].set_xlabel(r"$t/\tau_r$")
    ax[0].set_ylabel(r"$\langle (\Delta x)^2 \rangle/w^2$")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[1].set_title('late-time trajectory')
    ax[1].set_xlabel(r"$t/\tau_r$")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    # Iterate over each MSD, colour
    for msd, colour in zip(msds, colours):
        # Add MSD to figures
        ax[0].scatter(t, msd, color=colour, marker='.', s=10)
        ax[1].scatter(t[t > 1000], msd[t > 1000], color=colour, marker='.', s=10)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', type=str, help="Geometrical elongation factor")
    parser.add_argument('-Ps', type=str, help="Swim Peclet number")
    parser.add_argument('-Pf', type=str, help="flow Peclet number")
    parser.add_argument('-off', type=int, default=0, help="Skipped logscale blocks")
    parser.add_argument('--MSD', action='store_true', help="Plot the MSD")
    parser.add_argument('--Var', action='store_true', help="Plot the variance")
    parser.add_argument('--MSD3', action='store_true', help="Plot three MSDs")
    parser.add_argument('--ppMSD', action='store_true', help="Plot MSDs for individual trajectories")
    parser.add_argument('--all', action='store_true', help="Plot all MSDs from parameter list")
    parser.add_argument('-PsL', type=str, help="List of swim Peclet numbers")
    parser.add_argument('-PfL', type=str, help="List of flow Peclet numbers")
    parser.add_argument('-s', type=int, help="Number of trajectories to consider")
    parser.add_argument('-old', action='store_true', help="Specify whether to plot single origin MSDs")
    parser.add_argument('-f1', type=str, default=None, help='Filepath to first dataset')
    parser.add_argument('-f2', type=str, default=None, help='Filepath to second dataset, if applicable')
    parser.add_argument('-f3', type=str, default=None, help='Filepath to third dataset, if applicable')
    args = parser.parse_args()

    if args.MSD:
        plot_MSD(args.G, args.Ps, args.Pf, args.off)
    elif args.all:
        plot_all_MSD(args.G, args.PsL, args.PfL, args.off, args.old)
    elif args.ppMSD:
        plot_pp_MSD(args.G, args.Ps, args.Pf, args.s)
    elif args.MSD3:
        plot_MSD3(args.G, args.f1, args.f2, args.f3, args.off)
    elif args.Var:
        plot_variance(args.G, args.Ps, args.Pf, args.off)