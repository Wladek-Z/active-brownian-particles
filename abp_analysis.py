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

def phase_diagram(filename):
    """
    Plot the various phase diagrams of the ABP system in the persistence 
    length-Peclet number ratio plane.
    
    Arguments:
        filename: file to stored data
    """
    # Read parameters
    with open(filename, 'r') as f:
        line1 = f.readline().strip()
        G = float(line1.split("=")[1])
        line2 = f.readline().strip()
        D = float(line2.split("=")[1])
    # Read in and interpret data
    lp_w, U_v, a, b, D_eff, mean_vx = np.loadtxt(filename, delimiter=',', skiprows=3, unpack=True)
    nlp_w, nU_v = np.unique(lp_w), np.unique(U_v)
    size_x = len(nlp_w)
    size_y = len(nU_v)
    A = a.reshape(size_x, size_y).T
    B = b.reshape(size_x, size_y).T
    Deff = D_eff.reshape(size_x, size_y).T 
    VX = mean_vx.reshape(size_x, size_y).T #/ np.linspace(0.5, 4, 16)
    X, Y = np.meshgrid(nlp_w, nU_v)

    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=A.min(), vcenter=1.5, vmax=A.max())

    # Plot MSD scaling exponent
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$ scaling exponent: $D$ = {D}, $G$ = {G}")
    plt.pcolormesh(X, Y, A, cmap='bwr', norm=norm_a, shading='auto')
    plt.colorbar(label=r'$\alpha$')
    plt.xlabel("$l_p/w$")
    plt.ylabel("$U/v_0$")
    plt.tight_layout()

    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=VX.min(), vcenter=0, vmax=VX.max())

    # Plot mean longitudinal velocity
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"Mean longitudinal velocity: $G$ = {G}")
    plt.pcolormesh(X, Y, VX, cmap='bwr', shading='auto', norm=norm_vx)
    plt.colorbar(label=r'$\langle v_x \rangle/v_0$')
    plt.xlabel("$l_p/w$")
    plt.ylabel("$U/v_0$")
    plt.tight_layout()

    # Normalise divergent colormap for variance
    norm_var = colors.TwoSlopeNorm(vmin=np.nanmin(B), vcenter=1, vmax=np.nanmax(B))

    # Plot variance of displacement
    fig = plt.figure(figsize=[8, 6])
    plt.title(r"Var($\Delta x$) scaling exponent: " + f'$D$ = {D}, $G$ = {G}')
    plt.pcolormesh(X, Y, B, cmap='bwr', shading='auto', norm=norm_var)
    plt.colorbar(label=r'$\beta$')
    plt.xlabel("$l_p/w$")
    plt.ylabel("$U/v_0$")
    plt.tight_layout()

    # Plot effective diffusivity
    fig = plt.figure(figsize=[8, 6])
    plt.title(r"Effective diffusivity: " + f'$D$ = {D}, $G$ = {G}')
    plt.pcolormesh(X, Y, Deff, cmap='rainbow', shading='auto', norm='log')
    plt.colorbar(label=r'$D_{\mathrm{eff}}$')
    plt.xlabel("$l_p/w$")
    plt.ylabel("$U/v_0$")
    plt.tight_layout()

    plt.show()


def phase_diagram_alt(filename):
    """
    Plot the various phase diagrams of the ABP system in the swim Peclet
    number-flow Peclet number plane.
    
    Arguments:
        filename: file to stored data
    """
    # Read parameters
    with open(filename, 'r') as f:
        line1 = f.readline().strip()
        G = float(line1.split("=")[1])
        line2 = f.readline().strip()
        D = float(line2.split("=")[1])
    # Read in and interpret data
    Ps, Pf, a, b, D_eff, mean_vx = np.loadtxt(filename, delimiter=',', skiprows=3, unpack=True)
    nPs, nPf = np.unique(Ps), np.unique(Pf)
    size_x = len(nPs)
    size_y = len(nPf)
    A = a.reshape(size_x, size_y).T
    B = b.reshape(size_x, size_y).T
    Deff = D_eff.reshape(size_x, size_y).T 
    VX = mean_vx.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nPs, nPf)

    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=A.min(), vcenter=1.5, vmax=A.max())

    # Plot MSD scaling exponent
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"MSD$_x$ scaling exponent: $D$ = {D}, $G$ = {G}")
    plt.pcolormesh(X, Y, A, cmap='bwr', norm=norm_a, shading='auto')
    plt.colorbar(label=r'$\alpha$')
    plt.xlabel("$Pe_s$")
    plt.ylabel("$Pe_f$")
    plt.tight_layout()

    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=VX.min(), vcenter=0, vmax=VX.max())
    part1 = np.linspace(VX.min(), 0, 4)
    part2 = np.linspace(0, VX.max(), 4)[1:]
    ticks = np.append(part1, part2)

    # Plot mean longitudinal velocity
    fig = plt.figure(figsize=[8, 6])
    plt.title(f"Mean longitudinal velocity: $D$ = {D}, $G$ = {G}")
    plt.pcolormesh(X, Y, VX, cmap='bwr', shading='auto', norm=norm_vx)
    cbar = plt.colorbar(label=r'$\langle v_x \rangle/v_0$')
    cbar.set_ticks(ticks=ticks)
    cbar.minorticks_off()
    plt.xlabel("$Pe_s$")
    plt.ylabel("$Pe_f$")
    plt.tight_layout()

    # Normalise divergent colormap for variance
    norm_var = colors.TwoSlopeNorm(vmin=np.nanmin(B), vcenter=1, vmax=np.nanmax(B))

    # Plot variance of displacement
    fig = plt.figure(figsize=[8, 6])
    plt.title(r"Var($\Delta x$) scaling exponent: " + f'$D$ = {D}, $G$ = {G}')
    plt.pcolormesh(X, Y, B, cmap='bwr', shading='auto', norm=norm_var)
    plt.colorbar(label=r'$\beta$')
    plt.xlabel("$Pe_s$")
    plt.ylabel("$Pe_f$")
    plt.tight_layout()

    # Plot effective diffusivity
    fig = plt.figure(figsize=[8, 6])
    plt.title(r"Effective diffusivity: " + f'$D$ = {D}, $G$ = {G}')
    plt.pcolormesh(X, Y, Deff, cmap='rainbow', shading='auto', norm='log')
    plt.colorbar(label=r'$D_{\mathrm{eff}}$')
    plt.xlabel("$Pe_s$")
    plt.ylabel("$Pe_f$")
    plt.tight_layout()

    plt.show()


def pd3_comparison(filename1, filename2, filename3):
    """
    Plot side-by-side phase diagram comparisons for three datasets.
    
    Arguments:
        filename1: filepath to dataset with shear
        filename2: filepath to dataset without shear
        filename3: filepath to dataset without shear or vorticity
    """
    # Read in and interpret data (1)
    lp_w1, U_v1, a1, b1, D_eff1, mean_vx1 = np.loadtxt(filename1, delimiter=',', skiprows=3, unpack=True)
    nlp_w1, nU_v1 = np.unique(lp_w1), np.unique(U_v1)
    size_x1 = len(nlp_w1)
    size_y1 = len(nU_v1)
    A1 = a1.reshape(size_x1, size_y1).T
    B1 = b1.reshape(size_x1, size_y1).T
    D1 = D_eff1.reshape(size_x1, size_y1).T 
    VX1 = mean_vx1.reshape(size_x1, size_y1).T 
    X1, Y1 = np.meshgrid(nlp_w1, nU_v1)

    # Read in and interpret data (2)
    lp_w2, U_v2, a2, b2, D_eff2, mean_vx2 = np.loadtxt(filename2, delimiter=',', skiprows=3, unpack=True)
    nlp_w2, nU_v2 = np.unique(lp_w2), np.unique(U_v2)
    size_x2 = len(nlp_w2)
    size_y2 = len(nU_v2)
    A2 = a2.reshape(size_x2, size_y2).T
    B2 = b2.reshape(size_x2, size_y2).T
    D2 = D_eff2.reshape(size_x2, size_y2).T 
    VX2 = mean_vx2.reshape(size_x2, size_y2).T 
    X2, Y2 = np.meshgrid(nlp_w2, nU_v2)

    # Read in and interpret data (2)
    lp_w3, U_v3, a3, b3, D_eff3, mean_vx3 = np.loadtxt(filename3, delimiter=',', skiprows=3, unpack=True)
    nlp_w3, nU_v3= np.unique(lp_w3), np.unique(U_v3)
    size_x3 = len(nlp_w3)
    size_y3 = len(nU_v3)
    A3 = a3.reshape(size_x3, size_y3).T
    B3 = b3.reshape(size_x3, size_y3).T
    D3 = D_eff3.reshape(size_x3, size_y3).T 
    VX3 = mean_vx3.reshape(size_x3, size_y3).T
    X3, Y3 = np.meshgrid(nlp_w3, nU_v3)

    # Find minimum/maximum values for alpha
    vmin = np.round(min(np.nanmin(A1), np.nanmin(A2), np.nanmin(A3)), 2)
    vmax = np.round(max(np.nanmax(A1), np.nanmax(A2), np.nanmax(A3)), 2)
    # Normalise divergent colormap
    norm_a = colors.TwoSlopeNorm(vmin=vmin, vcenter=1.5, vmax=vmax)

    # Define function for plotting MSD scaling exponents
    def scaling_plot(data1, data2, data3, title, label, norm, cmap='bwr'):
        fig, axes = plt.subplots(1, 3, figsize=[21, 5], constrained_layout=True, sharey=True)
        mesh1 = axes[0].pcolormesh(X1, Y1, data1, cmap=cmap, norm=norm, shading='auto')
        axes[0].set_title('vorticity, shear')
        axes[0].set_xlabel("$l_p/w$")
        axes[0].set_ylabel("$U/v_0$")
        mesh2 = axes[1].pcolormesh(X2, Y2, data2, cmap=cmap, norm=norm, shading='auto')
        axes[1].set_title('vorticity, no shear')
        axes[1].set_xlabel("$l_p/w$")
        mesh3 = axes[2].pcolormesh(X3, Y3, data3, cmap=cmap, norm=norm, shading='auto')
        axes[2].set_title('no vorticity, no shear')
        axes[2].set_xlabel("$l_p/w$")
        fig.suptitle(title)
        fig.colorbar(mesh3, ax=axes, location='right', label=label)
        return fig
    
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
    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # Plot comparison of mean longitudinal velocity
    scaling_plot(VX1, VX2, VX3, "Mean longitudinal velocity", r"$\langle v_x \rangle/v_0$", norm_vx)

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

    # Plot results
    fig = plt.figure(figsize=[8, 6])
    plt.title(title)
    plt.scatter(bin_centres1, counts1, color='green', marker='.', s=10, label=label1)
    plt.scatter(bin_centres2, counts2, color='red', marker='.', s=10, label=label2)
    plt.scatter(bin_centres3, counts3, color='blue', marker='.', s=10, label=label3)
    plt.xlabel("$tD_r$")
    plt.ylabel("probability density")
    #plt.xscale('log')
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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', type=str, default=None, help='Filepath to first dataset')
    parser.add_argument('-f2', type=str, default=None, help='Filepath to second dataset, if applicable')
    parser.add_argument('-f3', type=str, default=None, help='Filepath to third dataset, if applicable')
    parser.add_argument('--PD', action='store_true', help='Construct the phase diagram')
    parser.add_argument('--PDA', action='store_true', help='Construct the alternative phase diagram')
    parser.add_argument('--PD3', action='store_true', help='Compare the phase diagrams of systems with/without shear, vorticity')
    parser.add_argument('--PDX', action='store_true', help='Compare the phase diagrams of alpha for total and longitudinal displacement')
    parser.add_argument('-F', type=str, default=None, help='Folder containing data files')
    parser.add_argument('--hist', action='store_true', help='Construct histograms from saved data')
    parser.add_argument('--TTD', action='store_true', help='Display the trapping time distribution')
    parser.add_argument('--TTD3', action='store_true', help='Display the trapping time distribution for three phases')
    parser.add_argument('--FPTD', action='store_true', help='Display the first passage time distribution')
    parser.add_argument('--eff', action='store_true', help='Plot effective diffusivities/Peclet numbers along an axis')
    args = parser.parse_args()

    if args.PD:
        phase_diagram(args.f1)
    elif args.PDA:
        phase_diagram_alt(args.f1)
    if args.PD3:
        pd3_comparison(args.f1, args.f2, args.f3)
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