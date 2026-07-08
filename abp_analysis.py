import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import scienceplots
from abp import ABP
from pathlib import Path

plt.style.use('science')
plt.rcParams['text.usetex'] = False

def phase_diagram(filename):
    """
    Plot the various phase diagrams of the ABP system in the Peclet
    number-persistence length plane.
    
    Arguments:
        filename: file to stored data
    """
    # Read in and interpret data
    lp_w, Pf_Ps, a, b, trap_frac, mean_vx = np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)
    nlp_w, nPf_Ps = np.unique(lp_w), np.unique(Pf_Ps)
    size_x = len(nlp_w)
    size_y = len(nPf_Ps)
    A = a.reshape(size_x, size_y).T
    B = b.reshape(size_x, size_y).T
    TF = trap_frac.reshape(size_x, size_y).T
    VX = mean_vx.reshape(size_x, size_y).T
    X, Y = np.meshgrid(nlp_w, nPf_Ps)

    # Define custom colormap
    colours = ['blue', 'white', 'red']
    cmap = colors.ListedColormap(colours)
    vmin = np.min(A)
    vmax = np.max(A)
    boundaries = np.array([np.round(vmin, 2), 1.2, 1.8, np.round(vmax, 2)])

    # Plot MSD scaling exponent
    fig = plt.figure(figsize=[8, 6])
    plt.title("MSD$_x$ scaling exponent")
    plt.pcolormesh(X, Y, A, cmap=cmap, shading='auto')
    cbar = plt.colorbar(label=r'$\alpha$', boundaries=boundaries, spacing='proportional')
    cbar.set_ticks(boundaries)
    cbar.set_ticklabels(boundaries)
    plt.xlabel("$l_p/w$")
    plt.ylabel("$Pe_f/Pe_s$")

    # Normalise divergent colormap
    norm_vx = colors.TwoSlopeNorm(vmin=VX.min(), vcenter=0, vmax=VX.max())

    # Plot mean longitudinal velocity
    fig = plt.figure(figsize=[8, 6])
    plt.title("Mean longitudinal velocity")
    plt.pcolormesh(X, Y, VX, cmap='bwr', shading='auto', norm=norm_vx)
    plt.colorbar(label=r'$\langle v_x \rangle/wD_r$')
    plt.xlabel("$l_p/w$")
    plt.ylabel("$Pe_f/Pe_s$")

    # Normalise divergent colormap for variance
    norm_var = colors.TwoSlopeNorm(vmin=np.nanmin(B), vcenter=1, vmax=np.nanmax(B))

    # Plot variance of displacement
    fig = plt.figure(figsize=[8, 6])
    plt.title(r"Var($\Delta x$) scaling exponent")
    plt.pcolormesh(X, Y, B, cmap='bwr', shading='auto', norm=norm_var)
    cbar = plt.colorbar(label=r'$\beta$', spacing='proportional')
    plt.xlabel("$l_p/w$")
    plt.ylabel("$Pe_f/Pe_s$")

    plt.tight_layout()
    plt.show()

def pd_comparison(filename1, filename2):
    """
    Plot side-by-side phase diagram comparisons for two datasets.
    
    Arguments:
        filename1: filepath to dataset with shear
        filename2: filepath to dataset without shear
    """
    # Read in and interpret data (1)
    lp_w1, Pf_Ps1, a1, b1, trap_frac1, mean_vx1 = np.loadtxt(filename1, delimiter=',', skiprows=1, unpack=True)
    nlp_w1, nPf_Ps1 = np.unique(lp_w1), np.unique(Pf_Ps1)
    size_x1 = len(nlp_w1)
    size_y1 = len(nPf_Ps1)
    A1 = a1.reshape(size_x1, size_y1).T
    B1 = b1.reshape(size_x1, size_y1).T
    TF1 = trap_frac1.reshape(size_x1, size_y1).T
    VX1 = mean_vx1.reshape(size_x1, size_y1).T
    X1, Y1 = np.meshgrid(nlp_w1, nPf_Ps1)

    # Read in and interpret data (2)
    lp_w2, Pf_Ps2, a2, b2, trap_frac2, mean_vx2 = np.loadtxt(filename2, delimiter=',', skiprows=1, unpack=True)
    nlp_w2, nPf_Ps2 = np.unique(lp_w2), np.unique(Pf_Ps2)
    size_x2 = len(nlp_w2)
    size_y2 = len(nPf_Ps2)
    A2 = a2.reshape(size_x2, size_y2).T
    B2 = b2.reshape(size_x2, size_y2).T
    TF2 = trap_frac2.reshape(size_x2, size_y2).T
    VX2 = mean_vx2.reshape(size_x2, size_y2).T
    X2, Y2 = np.meshgrid(nlp_w2, nPf_Ps2)

    # Define custom colormap
    colours = ['blue', 'white', 'red']
    cmap = colors.ListedColormap(colours)
    boundaries = np.array([1, 1.2, 1.8, 2])

    # Plot comparison of MSD scaling exponents
    fig, axes = plt.subplots(1, 2, figsize=[14, 6], constrained_layout=True, sharey=True)
    mesh1 = axes[0].pcolormesh(X1, Y1, A1, cmap=cmap, shading='auto')
    axes[0].set_title('shear effects')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")
    mesh2 = axes[1].pcolormesh(X2, Y2, A2, cmap=cmap, shading='auto')
    axes[1].set_title('no shear effects')
    axes[1].set_xlabel("$l_p/w$")
    fig.suptitle("ABP channel phase diagram comparison")
    cbar = fig.colorbar(mesh2, ax=axes, location='right', label=r"$\alpha$", boundaries=boundaries, spacing='proportional')
    cbar.set_ticks([1, 1.2, 1.8, 2])
    cbar.set_ticklabels([1, 1.2, 1.8, 2])

    # Plot comparison of variance scaling exponents
    fig, axes = plt.subplots(1, 2, figsize=[14, 6], constrained_layout=True, sharey=True)
    mesh1 = axes[0].pcolormesh(X1, Y1, B1, cmap=cmap, shading='auto')
    axes[0].set_title('shear effects')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")
    mesh2 = axes[1].pcolormesh(X2, Y2, B2, cmap=cmap, shading='auto')
    axes[1].set_title('no shear effects')
    axes[1].set_xlabel("$l_p/w$")
    fig.suptitle("ABP channel phase diagram comparison")
    cbar = fig.colorbar(mesh2, ax=axes, location='right', label=r"$\beta$", boundaries=boundaries, spacing='proportional')
    cbar.set_ticks([1, 1.2, 1.8, 2])
    cbar.set_ticklabels([1, 1.2, 1.8, 2])

    # Normalise divergent colormap
    vmin = min(np.nanmin(VX1), np.nanmin(VX2))
    vmax = max(np.nanmax(VX1), np.nanmax(VX2))
    norm_vx = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot comparison of mean longitudinal velocity    
    fig, axes = plt.subplots(1, 2, figsize=[14, 6], constrained_layout=True, sharey=True)
    mesh1 = axes[0].pcolormesh(X1, Y1, VX1, cmap='bwr', shading='auto', norm=norm_vx)
    axes[0].set_title('shear effects')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")
    mesh2 = axes[1].pcolormesh(X2, Y2, VX2, cmap='bwr', shading='auto', norm=norm_vx)
    axes[1].set_title('no shear effects')
    axes[1].set_xlabel("$l_p/w$")
    fig.suptitle("ABP channel phase diagram comparison")
    fig.colorbar(mesh2, ax=axes, location='right', label=r"$\langle v_x \rangle/wD_r$")

    # Plot trapping fraction comparison
    vmin = min(np.nanmin(TF1), np.nanmin(TF2))
    vmax = max(np.nanmax(TF1), np.nanmax(TF2))
    fig, axes = plt.subplots(1, 2, figsize=[14, 6], constrained_layout=True, sharey=True)
    mesh1 = axes[0].pcolormesh(X1, Y1, TF1, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('shear effects')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")
    mesh2 = axes[1].pcolormesh(X2, Y2, TF2, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('no shear effects')
    axes[1].set_xlabel("$l_p/w$")
    fig.suptitle("ABP channel phase diagram comparison")
    fig.colorbar(mesh2, ax=axes, location='right', label='trapping fraction')

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
    lp_w1, Pf_Ps1, a1, b1, trap_frac1, mean_vx1 = np.loadtxt(filename1, delimiter=',', skiprows=1, unpack=True)
    nlp_w1, nPf_Ps1 = np.unique(lp_w1), np.unique(Pf_Ps1)
    size_x1 = len(nlp_w1)
    size_y1 = len(nPf_Ps1)
    A1 = a1.reshape(size_x1, size_y1).T
    B1 = b1.reshape(size_x1, size_y1).T
    TF1 = trap_frac1.reshape(size_x1, size_y1).T
    VX1 = mean_vx1.reshape(size_x1, size_y1).T
    X1, Y1 = np.meshgrid(nlp_w1, nPf_Ps1)

    # Read in and interpret data (2)
    lp_w2, Pf_Ps2, a2, b2, trap_frac2, mean_vx2 = np.loadtxt(filename2, delimiter=',', skiprows=1, unpack=True)
    nlp_w2, nPf_Ps2 = np.unique(lp_w2), np.unique(Pf_Ps2)
    size_x2 = len(nlp_w2)
    size_y2 = len(nPf_Ps2)
    A2 = a2.reshape(size_x2, size_y2).T
    B2 = b2.reshape(size_x2, size_y2).T
    TF2 = trap_frac2.reshape(size_x2, size_y2).T
    VX2 = mean_vx2.reshape(size_x2, size_y2).T
    X2, Y2 = np.meshgrid(nlp_w2, nPf_Ps2)

    # Read in and interpret data (2)
    lp_w3, Pf_Ps3, a3, b3, trap_frac3, mean_vx3 = np.loadtxt(filename3, delimiter=',', skiprows=1, unpack=True)
    nlp_w3, nPf_Ps3= np.unique(lp_w3), np.unique(Pf_Ps3)
    size_x3 = len(nlp_w3)
    size_y3 = len(nPf_Ps3)
    A3 = a3.reshape(size_x3, size_y3).T
    B3 = b3.reshape(size_x3, size_y3).T
    TF3 = trap_frac3.reshape(size_x3, size_y3).T
    VX3 = mean_vx3.reshape(size_x3, size_y3).T
    X3, Y3 = np.meshgrid(nlp_w3, nPf_Ps3)

    # Find minimum/maximum values for alpha
    vmin = np.round(min(np.nanmin(A1), np.nanmin(A2), np.nanmin(A3)), 2)
    vmax = np.round(max(np.nanmax(A1), np.nanmax(A2), np.nanmax(A3)), 2)

    # Define custom colormap
    colours = ['blue', 'white', 'red']
    cmap = colors.ListedColormap(colours)
    boundaries = np.array([vmin, 1.2, 1.8, vmax])

    # Define function for plotting MSD/variance scaling exponents
    def scaling_plot(data1, data2, data3, title, label):
        fig, axes = plt.subplots(1, 3, figsize=[21, 5], constrained_layout=True, sharey=True)
        mesh1 = axes[0].pcolormesh(X1, Y1, data1, cmap=cmap, shading='auto')
        axes[0].set_title('vorticity, shear')
        axes[0].set_xlabel("$l_p/w$")
        axes[0].set_ylabel("$Pe_f/Pe_s$")
        mesh2 = axes[1].pcolormesh(X2, Y2, data2, cmap=cmap, shading='auto')
        axes[1].set_title('vorticity, no shear')
        axes[1].set_xlabel("$l_p/w$")
        mesh3 = axes[2].pcolormesh(X3, Y3, data3, cmap=cmap, shading='auto')
        axes[2].set_title('no vorticity, no shear')
        axes[2].set_xlabel("$l_p/w$")
        fig.suptitle(title)
        cbar = fig.colorbar(mesh3, ax=axes, location='right', label=label, boundaries=boundaries, spacing='proportional')
        cbar.set_ticks(boundaries)
        cbar.set_ticklabels(boundaries)
        return fig
    
    # Plot comparison of MSD scaling exponents
    scaling_plot(A1, A2, A3, 'MSD$_x$ scaling exponent', r'$\alpha$')
    
    # Normalise divergent colormap for variance
    vmin = min(np.nanmin(B1), np.nanmin(B2), np.nanmin(B3))
    vmax = max(np.nanmax(B1), np.nanmax(B2), np.nanmax(B3))
    norm_var = colors.TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)

    # Plot comparison of variance scaling exponents    
    fig, axes = plt.subplots(1, 3, figsize=[21, 5], constrained_layout=True, sharey=True)
    mesh1 = axes[0].pcolormesh(X1, Y1, B1, cmap='bwr', shading='auto', norm=norm_var)
    axes[0].set_title('vorticity, shear')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")
    mesh2 = axes[1].pcolormesh(X2, Y2, B2, cmap='bwr', shading='auto', norm=norm_var)
    axes[1].set_title('vorticity, no shear')
    axes[1].set_xlabel("$l_p/w$")
    mesh3 = axes[2].pcolormesh(X3, Y3, B3, cmap='bwr', shading='auto', norm=norm_var)
    axes[2].set_title('no vorticity, no shear')
    axes[2].set_xlabel("$l_p/w$")
    fig.suptitle(r'Var($\Delta x$) scaling exponent')
    fig.colorbar(mesh3, ax=axes, location='right', label=r'$\beta$')

    # Normalise divergent colormap for <vx>
    vmin = min(np.nanmin(VX1), np.nanmin(VX2), np.nanmin(VX3))
    vmax = max(np.nanmax(VX1), np.nanmax(VX2), np.nanmax(VX3))
    norm_vx = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plot comparison of mean longitudinal velocity    
    fig, axes = plt.subplots(1, 3, figsize=[21, 5], constrained_layout=True, sharey=True)
    mesh1 = axes[0].pcolormesh(X1, Y1, VX1, cmap='bwr', shading='auto', norm=norm_vx)
    axes[0].set_title('vorticity, shear')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")
    mesh2 = axes[1].pcolormesh(X2, Y2, VX2, cmap='bwr', shading='auto', norm=norm_vx)
    axes[1].set_title('vorticity, no shear')
    axes[1].set_xlabel("$l_p/w$")
    mesh3 = axes[2].pcolormesh(X3, Y3, VX3, cmap='bwr', shading='auto', norm=norm_vx)
    axes[2].set_title('no vorticity, no shear')
    axes[2].set_xlabel("$l_p/w$")
    fig.suptitle("Mean longitudinal velocity")
    fig.colorbar(mesh3, ax=axes, location='right', label=r"$\langle v_x \rangle/wD_r$")

    # Find minimum/maximum values for trapping fraction
    vmin = min(np.nanmin(TF1), np.nanmin(TF2), np.nanmin(TF3))
    vmax = max(np.nanmax(TF1), np.nanmax(TF2), np.nanmax(TF3))

    # Plot comparison of trapping fractions
    fig, axes = plt.subplots(1, 3, figsize=[21, 5], constrained_layout=True, sharey=True)
    mesh1 = axes[0].pcolormesh(X1, Y1, TF1, cmap='autumn', shading='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('vorticity, shear')
    axes[0].set_xlabel("$l_p/w$")
    axes[0].set_ylabel("$Pe_f/Pe_s$")
    mesh2 = axes[1].pcolormesh(X2, Y2, TF2, cmap='autumn', shading='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('vorticity, no shear')
    axes[1].set_xlabel("$l_p/w$")
    mesh3 = axes[2].pcolormesh(X3, Y3, TF3, cmap='autumn', shading='auto', vmin=vmin, vmax=vmax)
    axes[2].set_title('no vorticity, no shear')
    axes[2].set_xlabel("$l_p/w$")
    fig.suptitle("% time spent swimming upstream at the surface")
    fig.colorbar(mesh3, ax=axes, location='right', label="trapping fraction")
    
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



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', type=str, default=None, help='Filepath to first dataset')
    parser.add_argument('-f2', type=str, default=None, help='Filepath to second dataset, if applicable')
    parser.add_argument('-f3', type=str, default=None, help='Filepath to third dataset, if applicable')
    parser.add_argument('--PD', action='store_true', help='Construct the phase diagram')
    parser.add_argument('--PD3', action='store_true', help='Compare the phase diagrams of systems with/without shear, vorticity')
    parser.add_argument('--PDX', action='store_true', help='Compare the phase diagrams of alpha for total and longitudinal displacement')
    parser.add_argument('-F', type=str, default=None, help='Folder containing data files')
    parser.add_argument('--hist', action='store_true', help='Construct histograms from saved data')
    args = parser.parse_args()

    if args.PD:
        phase_diagram(args.f1)
    if args.PD3:
        pd3_comparison(args.f1, args.f2, args.f3)
    if args.PDX:
        pdx_comparison(args.f1)
    if args.hist:
        big_histogram(args.F)