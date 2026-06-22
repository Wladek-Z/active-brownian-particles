import matplotlib.pyplot as plt
import numpy as np

def angular_velocity():
    """
    Plot the angular velocity due to shear, due to flow velocity, and due 
    to the combined contribution of both.
    """
    colors = ['red', 'blue', 'green', 'cyan', 'yellow', 'magenta', 'orange', 'lime']

    # Define angular velocity formulas
    omega_shear = lambda yw, theta: (1 - 2 * yw) * np.cos(2 * theta)
    omega_flow = lambda yw: (2 * yw - 1)

    x = np.linspace(0, 1, 100)
    angles = np.arange(0, np.pi, np.pi/8)
    labels = ["0", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$", r"$5\pi/8$", r"$3\pi/4$", r"$7\pi/8$"]

    # Plot variation of flow angular velocity with height along channel
    O_f = omega_flow(x)
    fig = plt.figure(figsize=[8, 6])
    plt.plot(x, O_f, color='red')
    plt.title("Angular velocity due to flow effects")
    plt.xlabel(r"fractional height along channel, $y/w$")
    plt.ylabel(r"normalised angular velocity, $\Omega_{\mathrm{f}}$")
    plt.tight_layout()

    # Plot variation of shear angular velocity with height along channel for each angle
    fig = plt.figure(figsize=[8, 6])

    for i in range(len(colors)):
        O_s = omega_shear(x, angles[i])
        plt.plot(x, O_s, color=colors[i], label=labels[i])
    
    plt.title("Angular velocity due to shear effects")
    plt.xlabel(r"fractional height along channel, $y/w$")
    plt.ylabel(r"normalised angular velocity, $\Omega_{\mathrm{s}}$")
    plt.legend()
    plt.tight_layout()

    # Plot combined result
    fig = plt.figure(figsize=[8, 6])

    for i in range(len(colors)):
        O_s = omega_shear(x, angles[i])
        O_tot =  (O_s + O_f)
        plt.plot(x, O_tot, color=colors[i], label=labels[i])
    
    plt.title("Angular velocity due to flow and shear effects")
    plt.xlabel(r"fractional height along channel, $y/w$")
    plt.ylabel(r"total angular velocity, $\Omega_{\mathrm{f}} + \Omega_{\mathrm{s}}$")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    angular_velocity()
