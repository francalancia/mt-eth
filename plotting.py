import glob
import os
import numpy as np
import matplotlib.pyplot as plt

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
    for filename in file_list:
        # Load the NPZ file
        data = np.load(filename)

        # Extract arrays
        #x = data["x"]
        #u = data["u"]
        #alpha = data["alpha"]
        e_el = data["e_el"]
        e_dam = data["e_dam"]

        e_el_mean = np.sum(e_el)
        e_dam_mean = np.sum(e_dam)        
        
        e_el_list.append(e_el_mean)
        e_dam_list.append(e_dam_mean)
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
    
    e_el_stack = np.stack(e_el_list, axis=0)
    e_dam_stack = np.stack(e_dam_list, axis=0)
    x = np.linspace(-0.5, 0.5, 21)
    
    # Plotting
    fig = plt.figure(figsize=(10, 4))
    plt.plot(x, e_el_stack, label="elsastic energy")
    plt.plot(x, e_dam_stack, label="damage energy")
    plt.xlabel("x")
    plt.ylabel("energy")
    plt.title("Energy")
    plt.grid()
    #plt.ylim(0.0, 0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Update this pattern to match the location and file naming you use
    pattern = r"C:\Users\3399n\Documents\ETH\MasterThesis\Data_phase\outputphasefieldweak_UP*.npz"

    # Load files and make plots
    load_and_plot_multiple_files(pattern)