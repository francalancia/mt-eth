import glob
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 16,            
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})

def load_and_plot_multiple_files(pattern):
    """
    Find all .npz files matching the given pattern, 
    load the arrays (x, u, alpha, e_el, e_dam) from each, 
    and create sample plots.
    """
    # Find all files matching your pattern (e.g., *_UP*.npz)
    file_list = sorted(glob.glob(pattern))

    if not file_list:
        print(f"No files found for pattern: {pattern}")
        return

    e_el_list = []
    e_dam_list = []
    x_list = np.zeros([50,1])
    u_list = np.zeros([50,1])
    alpha_list = np.zeros([50,1])
    i = 0
    for filename in file_list:
        # Load the NPZ file
        data = np.load(filename)

        # Extract arrays
        x = data["x"].reshape(-1, 1)  # Ensure x is a column vector
        u = data["u"].reshape(-1, 1)
        alpha = data["alpha"].reshape(-1, 1)
        e_el = data["e_el"]
        e_dam = data["e_dam"]

        e_el_mean = np.mean(e_el)
        e_dam_mean = np.mean(e_dam)        
        
        e_el_list.append(e_el_mean)
        e_dam_list.append(e_dam_mean)
        if i == 0:
            x_list = x
            u_list = u
            alpha_list = alpha
        else:
            x_list = np.hstack((x_list, x))
            u_list = np.hstack((u_list, u))
            alpha_list = np.hstack((alpha_list, alpha))
        
        # Parse suffix from filename (e.g., "UP0.5")
        base = os.path.basename(filename)              # e.g., outputphasefieldweak_UP0.5.npz
        suffix = os.path.splitext(base)[0].split("_")[-1]  # e.g., "UP0.5"

        # Print some info
        print(f"\nLoaded file: {filename}")
        print(f"  Suffix: {suffix}")
        #print(f"  x shape: {x.shape}")
        #print(f"  u shape: {u.shape}")
        #print(f"  alpha shape: {alpha.shape}")
        print(f"  e_el shape: {e_el.shape}")
        print(f"  e_dam shape: {e_dam.shape}")
        i = 1
    
    e_el_stack = np.stack(e_el_list, axis=0)
    e_dam_stack = np.stack(e_dam_list, axis=0)
    x2 = np.linspace(0, 1, 101)
    
    if False:
        # Plotting
        fig = plt.figure(figsize=(10, 4))
        plt.plot(x2, e_el_stack, label="elsastic energy")
        plt.plot(x2, e_dam_stack, label="damage energy")
        plt.xlabel("x")
        plt.ylabel("energy")
        plt.title("Energy")
        plt.grid()
        #plt.ylim(0.0, 0.3)
        plt.legend()
        plt.tight_layout()
    
    if True:
        #Up = [0.0, 0.125,0.25,0.375,0.5]
        wd = [1e-5, 1e-8, 0]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for i in range(u_list.shape[1]):
            #axs[0].plot(x, u_list[:, i], label=f"u (Up = {Up[i]})")
            axs[0].plot(x, u_list[:, i], label=f"u (Wd = {wd[i]})")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("u")
        axs[0].set_title("Displacement")
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].grid()
        axs[0].legend()
        for i in range(alpha_list.shape[1]):
            axs[1].plot(x, alpha_list[:, i], label=f"alpha (Wd = {wd[i]})")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("alpha")
        axs[1].set_title("Phase Field")
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    #plt.savefig(r"E:\ETH\Master\25HS_MA\Data_Phasefield\PINN_PhasefieldEnergy_weak.png", dpi=300)
    #plt.savefig(r"E:\ETH\Master\25HS_MA\Data_Phasefield\PINN_Phasefield_weak.png", dpi=300)
    #plt.savefig(r"E:\ETH\Master\25HS_MA\Data_Phasefield\PINN_Phasefield_Strongwd.png", dpi=300)
    
    
    
    plt.show()


if __name__ == "__main__":
    # Update this pattern to match the location and file naming you use
    #pattern = r"E:\ETH\Master\25HS_MA\Data_Phasefield\PINN_weak_energy\outputphasefieldweak_UP*.npz"
    pattern = r"E:\ETH\Master\25HS_MA\Data_Phasefield\PINN_weak_displacement\outputphasefieldweak_UP*.npz"
    #pattern = r"E:\ETH\Master\25HS_MA\Data_Phasefield\PINN_Strong_displacement\outputphasefieldstrong_UP*.npz"

    # Load files and make plots
    load_and_plot_multiple_files(pattern)