import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from scipy.optimize import curve_fit
import scienceplots
from abp import ABP
from pathlib import Path

plt.style.use('science')
plt.rcParams['text.usetex'] = False

tau = 1
d = 2

def read_PD_data(filename):
    """
    Read in phase diagram data.
    
    Arguments:
        filename: filepath to stored data
    
    Returns:
        G: elongation factor
        D: dimensionless diffusion constant
        A: MSDx scaling exponent data
        B: variance scaling exponent data
        VX: mean longitudinal velocity data
        Deff: effective diffusivity data
        X: meshgrid columns
        Y: meshgrid rows
    """
    # Read parameters
    with open(filename, 'r') as f:
        line1 = f.readline().strip()
        G = float(line1.split("=")[1])
        line2 = f.readline().strip()
        D = float(line2.split("=")[1])
    # Read in and interpret data
    x, y, a, b, D_eff, mean_vx = np.loadtxt(filename, delimiter=',', skiprows=3, unpack=True)
    nx, ny = np.unique(x), np.unique(y)
    size_x = len(nx)
    size_y = len(ny)
    A = a.reshape(size_x, size_y).T
    B = b.reshape(size_x, size_y).T
    Deff = D_eff.reshape(size_x, size_y).T 
    VX = mean_vx.reshape(size_x, size_y).T #/ np.linspace(0.5, 4, 16)
    X, Y = np.meshgrid(nx, ny)
    # Return data
    return G, D, A, B, VX, Deff, X, Y
    

def phase_diagram(filename):
    """
    Plot the various phase diagrams of the ABP system in the persistence 
    length-Peclet number ratio plane.
    
    Arguments:
        filename: filepath to stored data
    """
    # Read data
    G, D, A, B, VX, Deff, X, Y = read_PD_data(filename)

    def plot(data, title, label, norm, cmap='bwr', ticks=None):
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"{title}: $D$ = {D}, $G$ = {G}")
        plt.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(label=label)
        if ticks is not None:
            cbar.set_ticks(ticks=ticks)
            cbar.minorticks_off()
        plt.xlabel("$l_p/w$")
        plt.ylabel("$Pe_f/Pe_s$")
        plt.tight_layout()
        return fig, cbar

    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=A.min(), vcenter=1.5, vmax=A.max())
    # Plot MSD scaling exponent
    plot(A, "MSD$_x$ scaling exponent", r'$\alpha$', norm_a)

    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=VX.min(), vcenter=0, vmax=VX.max())
    # Plot mean longitudinal velocity
    plot(VX, "Mean longitudinal velocity", r'$\langle v_x \rangle/v_0$', norm_vx)

    # Normalise divergent colormap for variance
    norm_var = colors.TwoSlopeNorm(vmin=np.nanmin(B), vcenter=1, vmax=np.nanmax(B))
    # Plot variance of displacement
    plot(B, r"Var($\Delta x$) scaling exponent", r"$\beta$", norm_var)

    # Plot effective diffusivity
    plot(Deff, "Effective diffusivity", r'$D_{\mathrm{eff}}$', 'log', 'rainbow')

    plt.show()


def phase_diagram_alt(filename):
    """
    Plot the various phase diagrams of the ABP system in the swim Peclet
    number-flow Peclet number plane.
    
    Arguments:
        filename: file to stored data
    """
    # Read data
    G, D, A, B, VX, Deff, X, Y = read_PD_data(filename)

    def plot(data, title, label, norm, cmap='bwr', ticks=None):
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"{title}: $D$ = {D}, $G$ = {G}")
        plt.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(label=label)
        if ticks is not None:
            cbar.set_ticks(ticks=ticks)
            cbar.minorticks_off()
        plt.xlabel("$Pe_s$")
        plt.ylabel("$Pe_f$")
        plt.tight_layout()
        return fig, cbar
    
    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=A.min(), vcenter=1.5, vmax=A.max())
    # Plot MSD scaling exponent
    plot(A, "MSD$_x$ scaling exponent", r'$\alpha$', norm_a)

    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=VX.min(), vcenter=0, vmax=VX.max())
    part1 = np.linspace(VX.min(), 0, 4)
    part2 = np.linspace(0, VX.max(), 4)[1:]
    ticks_vx = np.append(part1, part2)
    # Plot mean longitudinal velocity
    plot(VX, "Mean longitudinal velocity", r'$\langle v_x \rangle/v_0$', norm_vx, ticks=ticks_vx)

    # Normalise divergent colormap for variance
    norm_var = colors.TwoSlopeNorm(vmin=np.nanmin(B), vcenter=1, vmax=np.nanmax(B))
    # Plot variance of displacement
    plot(B, r"Var($\Delta x$) scaling exponent", r'$\beta$', norm_var)

    # Plot effective diffusivity
    plot(Deff, "Effective diffusivity", r'$D_{\mathrm{eff}}$', 'log', 'rainbow')

    plt.show()

def phase_diagram_vx(filename):
    """
    Construct a phase diagram of the mean longitudinal velocity.
    
    Arguments:
        filename: filepath to mean velocity results
    """
    # Read in and interpret data
    x, y, mean_vx = np.loadtxt(filename, skiprows=1, unpack=True)
    nx, ny = np.unique(x), np.unique(y)
    size_x = len(nx)
    size_y = len(ny)
    VX = mean_vx.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nx, ny)

    # Normalise divergent colormap
    title = "Mean longitudinal velocity"
    label = r'$\langle v_x \rangle/v_0$'
    norm_vx = colors.TwoSlopeNorm(vmin=VX.min(), vcenter=0, vmax=VX.max())
    part1 = np.linspace(VX.min(), 0, 4)
    part2 = np.linspace(0, VX.max(), 4)[1:]
    ticks_vx = np.append(part1, part2)

    # Plot mean longitudinal velocity
    fig = plt.figure(figsize=[8, 6])

    plt.title(f"{title}")
    plt.pcolormesh(X, Y, VX, cmap='bwr', norm=norm_vx, shading='auto')
    cbar = plt.colorbar(label=label)
    cbar.set_ticks(ticks=ticks_vx)
    cbar.minorticks_off()
    plt.xlabel("$Pe_s$")
    plt.ylabel("$Pe_f$")

    # Plot values as text
    #for Ps, Pf, vx in zip(x, y, mean_vx):
    #    #if Pf == (2 * Ps - 1):
    #    plt.text(Ps, Pf, f"{np.round(vx, 3)}", ha='center', va='center', fontsize=8)

    # Plot boundary between upstream/downstream swimming
    inc = (y[1] - y[0]) * 0.5
    Ps_list = np.sort(np.unique(x))
    dPs = Ps_list[1] - Ps_list[0]
    edges = np.concatenate(([Ps_list[0] - dPs/2], Ps_list + dPs/2))
    Pf_stairs = np.full(len(Ps_list), inc)
    i = -1

    for Ps in Ps_list:
        i += 1
        mvx_list = mean_vx[x == Ps]
        Pf_list = y[x == Ps]
        nmvx_list = mvx_list[mvx_list <= 0]
        if len(nmvx_list) > 0:
            max_nmvx = np.max(nmvx_list)
            Pf = Pf_list[mvx_list == max_nmvx][0]
            Pf_stairs[i] = Pf + inc

    #plt.stairs(Pf_stairs, edges, color='black', label=r'$\langle v_x \rangle \rightarrow 0$')
    plt.ylim(bottom=inc)

    plt.legend()
    plt.tight_layout()
    plt.show()

def phase_diagram_alpha(filename):
    """
    Construct a phase diagram of the MSD scaling exponent.
    
    Arguments:
        filename: filepath to alpha results
    """
    # Read in and interpret data
    x, y, alpha = np.loadtxt(filename, unpack=True)
    nx, ny = np.unique(x), np.unique(y)
    size_x = len(nx)
    size_y = len(ny)
    A = alpha.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nx, ny)

    # Generate labelling
    title = "MSD$_x$ scaling exponent"
    label = r'$\alpha$'
    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=A.min(), vcenter=1.5, vmax=A.max())

    # Plot MSD scaling exponent
    fig = plt.figure(figsize=[8, 6])

    plt.title(f"{title}")
    plt.pcolormesh(X, Y, A, cmap='bwr', norm=norm_a, shading='auto')
    cbar = plt.colorbar(label=label)
    plt.xlabel("$Pe_s$")
    plt.ylabel("$Pe_f$")

    # Plot values as text
    for Ps, Pf, a in zip(x, y, alpha):
        if Pf == (2 * Ps - 1):
            plt.text(Ps, Pf, f"{np.round(a, 3)}", ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

def phase_diagram_dx(filename1, filename2):
    """
    Construct a phase diagram of the mean longitudinal displacement in traps
    and in the bulk.
    
    Arguments:
        filename1: filepath to trapping results
        filename2: filepath to bulk results
    """
    # Read in and interpret data
    X, Y, TDX = read_data(filename1)
    X, Y, BDX = read_data(filename2)

    # Define phase diagram plotting function
    def plot(data, cmap, norm, title, label, ticks):
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"{title}")
        plt.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(label=label)
        cbar.set_ticks(ticks=ticks)
        cbar.minorticks_off()
        plt.xlabel("$Pe_s$")
        plt.ylabel("$Pe_f$")
        plt.tight_layout()
        return fig

    # Generate trap labelling
    title_T = "Mean longitudinal trap displacement"
    label_T = r'$\langle \Delta x \rangle_t$'
    norm_T = colors.TwoSlopeNorm(vmin=TDX.min(), vcenter=0, vmax=TDX.max())
    part1 = np.linspace(TDX.min(), 0, 4)
    part2 = np.linspace(0, TDX.max(), 4)[1:]
    ticks_T = np.append(part1, part2)
    # Plot trap diagram
    plot(TDX, 'bwr', norm_T, title_T, label_T, ticks_T)

    # Generate bulk labelling
    title_B = "Mean longitudinal bulk displacement"
    label_B = r'$\langle \Delta x \rangle_b$'
    norm_B = colors.TwoSlopeNorm(vmin=BDX.min(), vcenter=0, vmax=BDX.max())
    part1 = np.linspace(BDX.min(), 0, 4)
    part2 = np.linspace(0, BDX.max(), 4)[1:]
    ticks_B = np.append(part1, part2)
    # Plot bulk diagram
    plot(BDX, 'rainbow', norm_B, title_B, label_B, ticks_B)

    plt.show()

def phase_diagram_beta_Deff(filename):
    """
    Construct a phase diagram of the variance scaling exponent and the effective diffusivity.
    
    Arguments:
        filename: filepath to beta/D_eff results
    """
    # Read in and interpret data
    x, y, beta, Deff = np.loadtxt(filename, unpack=True)
    nx, ny = np.unique(x), np.unique(y)
    size_x = len(nx)
    size_y = len(ny)
    B = beta.reshape(size_x, size_y).T
    D_eff = Deff.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nx, ny)

    # Define phase diagram plotting function
    def plot(data, cmap, norm, title, label):
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"{title}")
        plt.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(label=label)
        plt.xlabel("$Pe_s$")
        plt.ylabel("$Pe_f$")
        plt.tight_layout()
        return fig
    
    # Generate labelling for beta
    title_b = r"Var($\Delta x$) scaling exponent"
    label_b = r'$\beta$'
    # Normalise divergent colormap
    norm_b = colors.TwoSlopeNorm(vmin=B.min(), vcenter=1, vmax=B.max())
    # Plot variance scaling exponent
    plot(B, 'bwr', norm_b, title_b, label_b)
    # Plot values as text
    for Ps, Pf, b in zip(x, y, beta):
        if Pf == (2 * Ps - 1):
            plt.text(Ps, Pf, f"{np.round(b, 3)}", ha='center', va='center', fontsize=8)

    # Generate labelling for Deff
    title_d = r"Effective diffusivity"
    label_d = r'$D_{\mathrm{eff}}$'
    # Plot variance scaling exponent
    plot(D_eff, 'rainbow', 'log', title_d, label_d)
    # Plot values as text
    for Ps, Pf, deff in zip(x, y, Deff):
        if Pf == (2 * Ps - 1):
            plt.text(Ps, Pf, f"{np.round(deff, 3)}", ha='center', va='center', fontsize=8)

    plt.show()

def read_data(filename):
    x, y, data = np.loadtxt(filename, unpack=True)
    nx, ny = np.unique(x), np.unique(y)
    size_x = len(nx)
    size_y = len(ny)
    DATA = data.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nx, ny)
    return X, Y, DATA
    
def phase_diagram_tau(filename):
    """
    Construct a phase diagram of tau for the trapping and bulk time distributions.
    
    Arguments:
        filename: filepath to decay constant results
    """
    # Read in and interpret data
    x, y, tau_ttd, tau_btd = np.loadtxt(filename, unpack=True)
    nx, ny = np.unique(x), np.unique(y)
    size_x = len(nx)
    size_y = len(ny)
    tauTTD = tau_ttd.reshape(size_x, size_y).T
    tauBTD = tau_btd.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nx, ny)

    # Define phase diagram plotting function
    def plot(data, cmap, norm, title, label):
        fig = plt.figure(figsize=[8, 6])
        plt.title(f"{title}")
        plt.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='auto')
        cbar = plt.colorbar(label=label)
        plt.xlabel("$Pe_s$")
        plt.ylabel("$Pe_f$")
        plt.tight_layout()
        return fig
    
    # Generate labelling for trapping tau
    title_ttd = "Characteristic trapping time"
    label_ttd = r'$\tau_t$'
    # Plot characteristic trapping time
    plot(tauTTD, 'rainbow', 'log', title_ttd, label_ttd)

    # Generate labelling for bulk tau
    title_btd = r"Characteristic bulk time"
    label_btd = r'$\tau_b$'
    # Plot characteristic bulk time
    plot(tauBTD, 'rainbow', 'log', title_btd, label_btd)

    # Generate labelling for ratio of taus
    title_ratio = "Characteristic trapping:bulk time ratio"
    label_ratio = r"$\tau_t/\tau_b$"
    # Plot ratio of characteristic trapping to bulk times
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"{title_ratio}")
    plt.pcolormesh(X, Y, tauTTD/tauBTD, cmap='rainbow', norm='log', shading='auto')
    cbar = plt.colorbar(label=label_ratio)
    plt.xlabel("$Pe_s$")
    plt.ylabel("$Pe_f$")

    # Draw Ps = Pf line and mark unit ratio phase points
    Ps_list = []
    Pf_list = []
    Ps_ratio_1 = []
    Pf_ratio_1 = []
    for i in range(len(x)):
        if x[i] == y[i]:
            Ps_list.append(x[i])
            Pf_list.append(y[i])
        if np.round(tau_ttd[i]/tau_btd[i], 0) == 1:
            Ps_ratio_1.append(x[i])
            Pf_ratio_1.append(y[i])

    plt.plot(Ps_list, Pf_list, color='black', label='$Pe_s = Pe_f$')
    plt.scatter(Ps_ratio_1, Pf_ratio_1, marker='+', color='black', s=20, label=r'$0.5 \leq \tau_t/\tau_b < 1.5$')
    plt.legend()
    plt.tight_layout()

    plt.show()

def pd3_comparison():
    """
    Plot side-by-side phase diagram comparisons for three datasets
    """
    # Define alternative data reading function
    def read_data2(filename):
        x, y, data1, data2 = np.loadtxt(filename, unpack=True)
        nx, ny = np.unique(x), np.unique(y)
        size_x = len(nx)
        size_y = len(ny)
        DATA1 = data1.reshape(size_x, size_y).T
        DATA2 = data2.reshape(size_x, size_y).T
        X, Y = np.meshgrid(nx, ny)
        return X, Y, DATA1, DATA2

    X1, Y1, A1 = read_data('alpha_G1.txt')
    X1, Y1, VX1 = read_data('mean_vx_G1.txt')
    X1, Y1, B1, Deff1 = read_data2('beta_G1.txt')

    X0, Y0, A0 = read_data('alpha_G0.txt')
    X0, Y0, VX0 = read_data('mean_vx_G0.txt')
    X0, Y0, B0, Deff0 = read_data2('beta_G0.txt')

    X0NV, Y0NV, A0NV = read_data('alpha_G0_NV.txt')
    X0NV, Y0NV, VX0NV = read_data('mean_vx_G0_NV.txt')
    X0NV, Y0NV, B0NV, Deff0NV = read_data2('beta_G0_NV.txt')

    # Find minimum/maximum values for alpha
    vmin = np.round(min(np.nanmin(A1), np.nanmin(A0), np.nanmin(A0NV)), 2)
    vmax = np.round(max(np.nanmax(A1), np.nanmax(A0), np.nanmax(A0NV)), 2)
    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=vmin, vcenter=1.5, vmax=vmax)

    # Define function for plotting MSD scaling exponents
    def plot(data1, data2, data3, title, label, norm, ticks=None):
        fig, axes = plt.subplots(1, 3, figsize=[21, 5], constrained_layout=True, sharey=True)
        mesh1 = axes[0].pcolormesh(X1, Y1, data1, cmap='bwr', norm=norm, shading='auto')
        axes[0].set_title('$G=1$')
        axes[0].set_xlabel("$Pe_s$")
        axes[0].set_ylabel("$Pe_f$")
        mesh2 = axes[1].pcolormesh(X0, Y0, data2, cmap='bwr', norm=norm, shading='auto')
        axes[1].set_title('$G = 0$')
        axes[1].set_xlabel("$Pe_s$")
        mesh3 = axes[2].pcolormesh(X0NV, Y0NV, data3, cmap='bwr', norm=norm, shading='auto')
        axes[2].set_title(r'$\Omega = 0$')
        axes[2].set_xlabel("$Pe_s$")
        fig.suptitle(f"{title}")
        cbar = fig.colorbar(mesh3, ax=axes, location='right', label=label)
        if ticks is not None:
            cbar.set_ticks(ticks=ticks)
            cbar.minorticks_off()
        return fig, cbar
    
    # Plot comparison of MSD scaling exponents
    plot(A1, A0, A0NV, 'MSD$_x$ scaling exponent', r'$\alpha$', norm_a)
    
    # Find min/max values of variance scaling exponent
    vmin = min(np.nanmin(B1), np.nanmin(B0), np.nanmin(B0NV))
    vmax = max(np.nanmax(B1), np.nanmax(B0), np.nanmax(B0NV))
    # Normalise divergent colormap
    norm_var = colors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
    # Plot comparison of variance scaling exponents
    plot(B1, B0, B0NV, r'Var($\Delta x$) scaling exponent', r'$\beta$', norm_var)

    # Find min/max values for <vx>
    vmin = min(np.nanmin(VX1), np.nanmin(VX0), np.nanmin(VX0NV))
    vmax = max(np.nanmax(VX1), np.nanmax(VX0), np.nanmax(VX0NV))
    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    part1 = np.linspace(vmin, 0, 4)
    part2 = np.linspace(0, vmax, 4)[1:]
    ticks_vx = np.append(part1, part2)
    # Plot comparison of mean longitudinal velocity
    plot(VX1, VX0, VX0NV, "Mean longitudinal velocity", r"$\langle v_x \rangle/v_0$", norm_vx, ticks_vx)

    # Plot comparison of effective diffusivity
    #plot(Deff1, Deff2, Deff3, "Effective diffusivity", r"$D_{\mathrm{eff}}$", 'log', 'rainbow')

    plt.show()

def pd3_comparison_alt(filename1, filename2, filename3):
    """
    Plot side-by-side phase diagram comparisons for three datasets, using
    alternative Peclet number plane.
    
    Arguments:
        filename1: filepath to dataset with shear
        filename2: filepath to dataset without shear
        filename3: filepath to dataset without shear or vorticity
    """
    # Read shear data
    G1, D1, A1, B1, Deff1, VX1, X1, Y1 = read_PD_data(filename1)

    # Read no shear data
    G2, D2, A2, B2, Deff2, VX2, X2, Y2 = read_PD_data(filename2)

    # Read no shear/vorticity data
    G3, D3, A3, B3, Deff3, VX3, X3, Y3 = read_PD_data(filename3)

    # Find minimum/maximum values for alpha
    vmin = np.round(min(np.nanmin(A1), np.nanmin(A2), np.nanmin(A3)), 2)
    vmax = np.round(max(np.nanmax(A1), np.nanmax(A2), np.nanmax(A3)), 2)
    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=vmin, vcenter=1.5, vmax=vmax)

    # Define function for plotting MSD scaling exponents
    def scaling_plot(data1, data2, data3, title, label, norm, cmap='bwr', ticks=None):
        fig, axes = plt.subplots(1, 3, figsize=[21, 5], constrained_layout=True, sharey=True)
        mesh1 = axes[0].pcolormesh(X1, Y1, data1, cmap=cmap, norm=norm, shading='auto')
        axes[0].set_title('vorticity, shear')
        axes[0].set_xlabel("$Pe_s$")
        axes[0].set_ylabel("$Pe_f$")
        mesh2 = axes[1].pcolormesh(X2, Y2, data2, cmap=cmap, norm=norm, shading='auto')
        axes[1].set_title('vorticity, no shear')
        axes[1].set_xlabel("$Pe_s$")
        mesh3 = axes[2].pcolormesh(X3, Y3, data3, cmap=cmap, norm=norm, shading='auto')
        axes[2].set_title('no vorticity, no shear')
        axes[2].set_xlabel("$Pe_s$")
        fig.suptitle(f"{title}: $D$ = {D1}")
        cbar = fig.colorbar(mesh3, ax=axes, location='right', label=label)
        if ticks is not None:
            cbar.set_ticks(ticks=ticks)
            cbar.minorticks_off()           
        return fig, cbar
    
    # Plot comparison of MSD scaling exponents
    scaling_plot(A1, A2, A3, 'MSD$_x$ scaling exponent', r'$\alpha$', norm_a)
    
    # Find min/max values of variance scaling exponent
    vmin = min(np.nanmin(B1), np.nanmin(B2), np.nanmin(B3))
    vmax = max(np.nanmax(B1), np.nanmax(B2), np.nanmax(B3))
    # Normalise divergent colormap
    norm_var = colors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
    # Plot comparison of variance scaling exponents
    scaling_plot(B1, B2, B3, r'Var($\Delta x$) scaling exponent', r'$\beta$', norm_var)

    # Find min/max values for <vx>
    vmin = min(np.nanmin(VX1), np.nanmin(VX2), np.nanmin(VX3))
    vmax = max(np.nanmax(VX1), np.nanmax(VX2), np.nanmax(VX3))
    part1 = np.linspace(vmin, 0, 4)
    part2 = np.linspace(0, vmax, 4)[1:]
    ticks_vx = np.append(part1, part2)
    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # Plot comparison of mean longitudinal velocity
    scaling_plot(VX1, VX2, VX3, "Mean longitudinal velocity", r"$\langle v_x \rangle/v_0$", norm_vx, ticks=ticks_vx)

    # Plot comparison of effective diffusivity
    scaling_plot(D1, D2, D3, "Effective diffusivity", r"$D_{\mathrm{eff}}$", 'log', 'rainbow')

    plt.show()

def pdx_comparison(filename1):
    """
    Plot side-by-side phase diagram of total MSD scaling exponent compared
    to MSD scaling exponent in the longitudinal direction only.
    
    Arguments:
        filename1: filepath to dataset
    """
    # Read in and interpret data
    lp_w, Pf_Ps, a, a_x = np.loadtxt(filename1, delimiter=',', skiprows=1, unpack=True)
    nlp_w, nPf_Ps = np.unique(lp_w), np.unique(Pf_Ps)
    size_x = len(nlp_w)
    size_y = len(nPf_Ps)
    A = a.reshape(size_x, size_y).T
    A_x = a_x.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nlp_w, nPf_Ps)

    vmin = min(np.nanmin(A), np.nanmin(A_x))
    vmax = max(np.nanmax(A), np.nanmax(A_x))
    fig, axes = plt.subplots(1, 2, figsize=[14, 6], constrained_layout=True)

    mesh1 = axes[0].pcolormesh(X, Y, A, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Total MSD')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")

    mesh2 = axes[1].pcolormesh(X, Y, A_x, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Longitudinal MSD')
    axes[1].set_xlabel("$l_p/w$")
    axes[1].set_ylabel("$Pe_f/Pe_s$")

    fig.colorbar(mesh2, ax=axes, location='right', label=r"MSD scaling exponent, $\alpha$")
    plt.show()

def big_histogram(folder):
    """
    Construct the three histograms from a large set of data, made up of many smaller files
    contained in folder.
    
    Arguments:
        folder: directory containing the data files
    """
    pdf1_total = 0
    pdf2_total = 0
    pdf3_total = 0

    # Read every file in folder and sum together the counts
    for file in Path(f"./{folder}").glob("data_*.npz"):
        data = np.load(file)
        pdf1_total += data['pdf1']
        pdf2_total += data['pdf2']
        pdf3_total += data['pdf3']
    
    # Save edges from most recent instance of data
    edges1 = data['edges1']
    edges2 = data['edges2']
    edges3 = data['edges3']

    # Normalise the counts
    bin_width1 = edges1[1] - edges1[0]   
    pdf1 = pdf1_total / (pdf1_total.sum() * bin_width1)
    bin_width2 = edges2[1] - edges2[0]   
    pdf2 = pdf2_total / (pdf2_total.sum() * bin_width2)
    bin_width3 = edges3[1] - edges3[0]   
    pdf3 = pdf3_total / (pdf3_total.sum() * bin_width3)

    # Extract parameters from folder name
    _, lp_w, Pf_Ps, G = folder.split("_")

    # Plot the three histograms
    fig = plt.figure(figsize=[8, 6])
    plt.stairs(pdf1, edges1, color='black')
    plt.title(f"Spatial distribution: $l_p/w$ = {lp_w}, $Pe_f/Pe_s$ = {Pf_Ps}, $G$ = {G}")
    plt.xlabel("height along channel, $y/w$")
    plt.ylabel("probability density, $P(y/w)$")
    plt.xlim(0, 1)

    fig = plt.figure(figsize=[8, 6])
    plt.stairs(pdf2, edges2, color='black')
    plt.title(f"Orientational distribution: $l_p/w$ = {lp_w}, $Pe_f/Pe_s$ = {Pf_Ps}, $G$ = {G}")
    plt.xlabel(r"orientation angle, $\theta$")
    plt.ylabel(r"probability density, $P(\theta)$")
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    
    fig = plt.figure(figsize=[8, 6])
    plt.stairs(pdf3, edges3, color='black')
    plt.title(f"Orientational distribution (trapped): $l_p/w$ = {lp_w}, $Pe_f/Pe_s$ = {Pf_Ps}, $G$ = {G}")
    plt.xlabel(r"orientation angle, $\theta$")
    plt.ylabel(r"probability density, $P(\theta)$")
    plt.xlim(-np.pi, np.pi)
    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    plt.tight_layout()
    plt.show()

def TD_logbin(data, num_bins=50):
    """
    Perform logarithmic binning on some time distribution data.

    Arguments:
        data: time distribution data
        num_bins: number of bins for histogram

    Returns:
        bin_centres: array of bin centres
        counts: normalised counts
        bins: locations of bin edges
    """
    # Perform logarithmic binning
    bins = np.logspace(np.log10(min(data)), np.log10(max(data)), num_bins)
    counts, bins = np.histogram(data, bins=bins, density=True)
    bin_centres = (bins[:-1] + bins[1:]) / 2
    return bin_centres, counts, bins

def TTD(filename):
    """
    Display the trapping time distribution.
    
    Arguments:
        filename: path to stored ttd data
    """
    # Read parameters and data
    with open(filename, 'r') as f:
        line1 = f.readline().strip()
        lp_w = float(line1.split("=")[1])
        line2 = f.readline().strip()
        Ps_Pf = float(line2.split("=")[1])
        line3 = f.readline().strip()
        G = float(line3.split("=")[1])
        tt = np.loadtxt(f)

    # Filter data 
    data = tt[tt > 0.1]
    # Construct histogram
    counts, bins = np.histogram(data, bins=200, density=True)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    # Perform curve fit
    #popt, pcov = curve_fit(func, bin_centres, counts)
    #tfit = np.logspace(np.log10(bin_centres[0]), np.log10(bin_centres[-1]), 200)
    #yfit = func(tfit, popt[0], popt[1], popt[2], popt[3])


    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.scatter(bin_centres, counts, color='black', marker='.', s=20, label='simulation')
    plt.title(f"$l_p/w$ = {lp_w}, $Pe_f/Pe_s$ = {Ps_Pf}, $G$ = {G}")
    #plt.plot(tfit, yfit, color='magenta', label=r'$Ae^{\gamma T} + Be^{-\zeta T}$')
    plt.xlabel("$tD_r$")
    plt.ylabel("probability density")
    #plt.xscale('log')
    plt.yscale('log')
    plt.axvline(tau, color='black', linestyle='dotted', label=r'$t=\tau_r$')
    #plt.text(tfit[3*len(tfit)//4], 0.75*yfit[3*len(tfit)//4], r'$\gamma$ = ' + f'{np.round(popt[2], 2)}\n' + r'$\zeta$ = ' + f'{np.round(popt[3], 2)}',  ha='left', va='bottom', fontsize=12)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def TTD3(filename1, filename2, filename3):
    """
    Plot the trapping time distributions for the upstream ballistic, downstream 
    ballistic, and diffusive regimes (or whatever else is being measured) on one graph.
    
    Arguments:
        filename1: filepath to first dataset
        filename2: filepath to second dataset
        filename3: filepath to third dataset
    """
    # Select number of bins
    num_bins = 'auto'
    # Read parameters and trapping times from first datafile
    with open(filename1, 'r') as f:
        line1 = f.readline().strip()
        lp_w1 = float(line1.split("=")[1])
        line2 = f.readline().strip()
        Ps_Pf1 = float(line2.split("=")[1])
        line3 = f.readline().strip()
        G = float(line3.split("=")[1])
        tt1 = np.loadtxt(f)

    # Filter data 
    data1 = tt1[tt1 > 0.1]
    # Construct histogram
    counts1, bins = np.histogram(data1, bins=num_bins, density=True)
    bin_centres1 = (bins[:-1] + bins[1:]) / 2

    # Get parameters and trapping times from second datafile
    with open(filename2, 'r') as f:
        line1 = f.readline().strip()
        lp_w2 = float(line1.split("=")[1])
        line2 = f.readline().strip()
        Ps_Pf2 = float(line2.split("=")[1])
        tt2 = np.loadtxt(f)
    
    # Filter data 
    data2 = tt2[tt2 > 0.1]
    # Construct histogram
    counts2, bins = np.histogram(data2, bins=num_bins, density=True)
    bin_centres2 = (bins[:-1] + bins[1:]) / 2

    # Get parameters and trapping times from third datafile
    with open(filename3, 'r') as f:
        line1 = f.readline().strip()
        lp_w3 = float(line1.split("=")[1])
        line2 = f.readline().strip()
        Ps_Pf3 = float(line2.split("=")[1])
        tt3 = np.loadtxt(f)
    
    # Filter data 
    data3 = tt3[tt3 > 0.1]
    # Construct histogram
    counts3, bins = np.histogram(data3, bins=num_bins, density=True)
    bin_centres3 = (bins[:-1] + bins[1:]) / 2

    if lp_w1 == lp_w2 == lp_w3:
        title = f"Trapping time distribution: $l_p/w$ = {lp_w1}, $G$ = {G}"
        label1 = f'$Pe_f/Pe_s$ = {Ps_Pf1}'
        label2 = f'$Pe_f/Pe_s$ = {Ps_Pf2}'
        label3 = f'$Pe_f/Pe_s$ = {Ps_Pf3}'
    elif Ps_Pf1 == Ps_Pf2 == Ps_Pf3:
        title = f"Trapping time distribution: $Pe_f/Pe_s$ = {Ps_Pf1}, $G$ = {G}"
        label1 = f'$l_p/w$ = {lp_w1}'
        label2 = f'$l_p/w$ = {lp_w2}'
        label3 = f'$l_p/w$ = {lp_w3}'
    else:
        title = f"Trapping time distribution: $G$ = {G}"
        label1 = f'$l_p/w$ = {lp_w1}, $Pe_f/Pe_s$ = {Ps_Pf1}'
        label2 = f'$l_p/w$ = {lp_w2}, $Pe_f/Pe_s$ = {Ps_Pf2}'
        label3 = f'$l_p/w$ = {lp_w3}, $Pe_f/Pe_s$ = {Ps_Pf3}'

    # Find minimum and maximum of independent data
    bin_min = np.min([np.min(bin_centres1), np.min(bin_centres2), np.min(bin_centres3)])
    bin_max = np.max([np.max(bin_centres1), np.max(bin_centres2), np.max(bin_centres3)])
    # Generate x- and y-axis data
    x = np.linspace(bin_min, bin_max, 50)
    y = np.exp(-x)

    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(title)
    plt.scatter(bin_centres1, counts1, color='green', marker='.', s=10, label=label1)
    plt.scatter(bin_centres2, counts2, color='red', marker='.', s=10, label=label2)
    plt.scatter(bin_centres3, counts3, color='blue', marker='.', s=10, label=label3)
    # Plot exponential decay
    plt.plot(x, y, color='black', label=r'$e^{-t/\tau_r}$')
    plt.xlabel("$tD_r$")
    plt.ylabel("probability density")
    plt.yscale('log')
    plt.axvline(tau, color='black', linestyle='dotted', label=r'$t=\tau_r$')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def FPTD(filename):
    """
    Display the first-passage time distribution for an ensemble of particles to reach a point x along the x-axis.
    
    Arguments:
        filename: location of FPT data
    """
    with open(filename, 'r') as f:
        line1 = f.readline().strip()
        lp_w = float(line1.split("=")[1])
        line2 = f.readline().strip()
        Ps_Pf = float(line2.split("=")[1])
        line3 = f.readline().strip()
        G = float(line3.split("=")[1])
        line4 = f.readline().strip()
        target = float(line4.split("=")[1])
        line5 = f.readline().strip()
        success_rate = float(line5.split("=")[1])
        fpt = np.loadtxt(f)

    # Construct histogram
    num_bins = 'auto'
    counts, bins = np.histogram(fpt, bins=num_bins, density=True)
    bin_centres = (bins[:-1] + bins[1:]) / 2

    # Build histogram of the FPTD
    fig = plt.figure(figsize=[8, 6])
    plt.scatter(bin_centres, counts, color='black', marker='.', s=20)
    #plt.plot(bin_centres, counts, color='black', linestyle='--')
    #plt.stairs(counts, bins, color='black')
    plt.title(f"FPTD: $l_p/w$ = {lp_w}, $Pe_f/Pe_s$ = {Ps_Pf}, $G$ = {G}, $x_T/w$ = {target}, success rate = {success_rate}%")
    plt.xlabel("$tD_r$")
    plt.ylabel("probability density")
    plt.yscale('log')
    #plt.xscale('log')

    plt.tight_layout()
    plt.show()

def effective_constants(filename):
    """
    Plot effective diffusivities and Peclet numbers along an axis of the
    Pf/Ps - lp/w phase diagram.
    
    Arguments:
        filename: filepath to stored data
    """
    # Load data and read in parameters
    data = np.load(filename)
    params = data['p']
    Ps = params[0]
    D = params[1]
    G = params[2]
    
    # Calculate effective diffusivity in the absence of flow
    D_eff_noflow = D**2 + Ps**2 / 4

    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$ effective constants: $l_p/w$ = {Ps}, $D$ = {D}, $G$ = {G}")
    plt.plot(data['PfPs_D'], data['D_eff'], '-o', color='red', markersize=4, label=r'$D_{\mathrm{eff}}$')
    plt.plot(data['PfPs_P'], data['P_eff'], '-o', color='blue', markersize=4, label=r'$Pe_{s,\mathrm{eff}}$')
    plt.xlabel("$Pe_f/Pe_s$")
    plt.ylabel("value of constant")
    plt.axhline(D_eff_noflow, color='black', linestyle='--', label='no flow', alpha=0.5)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

def Trajectory(filename):
    """
    Plot the trajectory of a single particle from its logscale trajectory file.
    
    Arguments:
        filename: logscale trajectory data
    """     
    # Read in trajectory data
    x, y, theta = np.loadtxt(filename, skiprows=1, delimiter=',', unpack=True) 
    dx = np.cos(theta)
    dy = np.sin(theta)

    # Retrieve start and end positions
    start_x = x[0]
    start_y = y[0]
    end_x = x[-1]
    end_y = y[-1]

    fig = plt.figure(figsize=[8, 6])

    # Show start and end points of trajectory
    plt.scatter(start_x, start_y, color='lime', s=100, marker='*', zorder=1)
    plt.scatter(end_x, end_y, color='red', s=100, marker='*', zorder=1)

    # Show start and end points of each logscale block
    plt.scatter(x[::17], y[::17], color='lime', s=10, zorder=0)
    plt.scatter(x[16::17], y[16::17], color='red', s=10, zorder=0)

    plt.scatter(x, y, color='black', marker='.', s=1, zorder=-1)
    plt.xlabel(r"$x/w$")
    plt.ylabel(r"$y/w$")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(1, color='black', linestyle='--', alpha=0.5)
    
    # Make quiver plot
    plt.quiver(x, y, dx, dy, color='blue', width=0.002, headwidth=3, headlength=4, scale=35, zorder=-1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', type=str, default=None, help='Filepath to first dataset')
    parser.add_argument('-f2', type=str, default=None, help='Filepath to second dataset, if applicable')
    parser.add_argument('-f3', type=str, default=None, help='Filepath to third dataset, if applicable')
    parser.add_argument('--PD', action='store_true', help='Construct the phase diagram')
    parser.add_argument('--PDalt', action='store_true', help='Construct the alternative phase diagram')
    parser.add_argument('--PD3', action='store_true', help='Compare the phase diagrams of systems with/without shear, vorticity')
    parser.add_argument('--PDX', action='store_true', help='Compare the phase diagrams of alpha for total and longitudinal displacement')
    parser.add_argument('--PDDX', action='store_true', help='Construct the phase diagram of mean trap and bulk displacements')
    parser.add_argument('--PDVX', action='store_true', help='Construct a phase diagram for the mean longitudinal velocity only')
    parser.add_argument('--PDA', action='store_true', help='Construct a phase diagram for the MSD scaling exponent only')
    parser.add_argument('--PDt', action='store_true', help='Construct a phase diagram for trapping/bulk time decay constant only')
    parser.add_argument('--PDB', action='store_true', help='Construct a phase diagram for the bariance scaling exponent and effective diffusivity only')
    parser.add_argument('-F', type=str, default=None, help='Folder containing data files')
    parser.add_argument('--trajectory', action='store_true', help="Plot trajectory from logscale data file")
    parser.add_argument('--hist', action='store_true', help='Construct histograms from saved data')
    parser.add_argument('--TTD', action='store_true', help='Display the trapping time distribution')
    parser.add_argument('--TTD3', action='store_true', help='Display the trapping time distribution for three phases')
    parser.add_argument('--FPTD', action='store_true', help='Display the first passage time distribution')
    parser.add_argument('--eff', action='store_true', help='Plot effective diffusivities/Peclet numbers along an axis')
    args = parser.parse_args()

    if args.PD:
        phase_diagram(args.f1)
    elif args.PDalt:
        phase_diagram_alt(args.f1)
    elif args.PDVX:
        phase_diagram_vx(args.f1)
    elif args.PDDX:
        phase_diagram_dx(args.f1, args.f2)
    elif args.PDA:
        phase_diagram_alpha(args.f1)
    elif args.PDB:
        phase_diagram_beta_Deff(args.f1)
    elif args.PDt:
        phase_diagram_tau(args.f1)
    if args.PD3:
        pd3_comparison()
    if args.PDX:
        pdx_comparison(args.f1)
    if args.hist:
        big_histogram(args.F)
    if args.TTD:
        TTD(args.f1)
    if args.TTD3:
        TTD3(args.f1, args.f2, args.f3)
    if args.FPTD:
        FPTD(args.f1)
    if args.eff:
        effective_constants(args.f1)
    if args.trajectory:
        Trajectory(args.f1)