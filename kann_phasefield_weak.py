"""Pytorch implementation of KANN."""

import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
import torch.nn as nn
import tqdm
import parameters_phasefield as param
import pandas as pd
import numpy as np
import os
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import datetime
torch.manual_seed(432)
# set the default data type for tensors to double precision
torch.set_default_dtype(torch.float64)
# set the default plotting sizes
plt.rcParams.update({
        'font.size': 12,              # Base font size
        'axes.labelsize': 12,         # Axis labels
        'xtick.labelsize': 12,        # X-axis tick labels
        'ytick.labelsize': 12,        # Y-axis tick labels
        'legend.fontsize': 12,        # Legend
        'figure.titlesize': 10        # Figure title
    })

class LagrKANNautoinner(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(LagrKANNautoinner, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max

        #self.weight = torch.nn.parameter.Parameter(
        #    torch.ones((self.n_width, self.n_nodes))
        #)
        # Define weight as a parameter
        self.weight = torch.nn.Parameter(torch.empty((self.n_width, self.n_nodes)))

        # Apply Xavier initialization (uniform)
        init.xavier_normal_(self.weight, gain=1.0)

    def lagrange(self, x, n_order):
        """Lagrange polynomials."""
        #stays the same
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)
        # needs extra dimension for parameter
        p_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        )
        for j in range(n_order + 1):
            p = 1.0
            for m in range(n_order + 1):
                if j != m:
                    p *= (x - nodes[m]) / (nodes[j] - nodes[m])
            p_list[:, :, j] = p

        return p_list

    def dlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)

        dp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        )
        for j in range(n_order + 1):
            y = 0.0
            for i in range(n_order + 1):
                if i != j:
                    k = torch.ones_like(x) / (nodes[j] - nodes[i])
                    for m in range(n_order + 1):
                        if m != i and m != j:
                            k *= (x- nodes[m]) / (nodes[j] - nodes[m])
                    y += k
            dp_list[:, :, j] = y

        return dp_list

    def ddlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)

        ddp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        )
        for j in range(n_order + 1):
            y = 0.0
            for i in range(n_order + 1):
                if i != j:
                    k_sum = 0.0
                    for m in range(n_order + 1):
                        if m != i and m != j:
                            k_prod = torch.ones_like(x) / (nodes[j] - nodes[m])
                            for n in range(n_order + 1):
                                if n != i and n != j and n != m:
                                    k_prod *= (x - nodes[n]) / (nodes[j] - nodes[n])
                            k_sum += k_prod
                    y += (1 / (nodes[j] - nodes[i])) * k_sum
            ddp_list[:, :, j] = y

        return ddp_list

    def to_ref(self, x_shift, node_l, node_r):
        return 2 * (x_shift-node_l)/(node_r-node_l) - 1

    def to_shift(self, x):
        x_shift = (self.n_nodes - 1) * (x - self.x_min) / (self.x_max - self.x_min)
        return x_shift

    def forward(self, x):
        """Forward pass for whole batch."""
        if len(x.shape) != 2:
            x = x.unsqueeze(-1)
            x = torch.repeat_interleave(x, self.n_width, -1)
        x_shift = self.to_shift(x)

        id_element_in = torch.floor(x_shift / self.n_order)
        # ensures that all elements of vector id_element_in are within the range of 0 and n_elements - 1
        id_element_in[id_element_in >= self.n_elements] = self.n_elements - 1
        id_element_in[id_element_in < 0] = 0

        nodes_in_l = (id_element_in * self.n_order).to(int)
        nodes_in_r = (nodes_in_l + self.n_order).to(int)

        x_transformed = self.to_ref(x_shift, nodes_in_l, nodes_in_r)
        delta_x = 0.5 * self.n_order * (self.x_max - self.x_min) / (self.n_nodes - 1)
        delta_x_1st = delta_x
        delta_x_2nd = delta_x**2

        phi_local_ikp = self.lagrange(x_transformed, self.n_order)
        dphi_local_ikp = self.dlagrange(x_transformed, self.n_order)
        ddphi_local_ikp = self.ddlagrange(x_transformed, self.n_order)

        phi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes))
        dphi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes))
        ddphi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes))
        for sample in range(self.n_samples):
            for layer in range(self.n_width):
                for node in range(self.n_order + 1):
                    phi_ikp[ sample, layer, nodes_in_l[sample, layer] + node
                        ] = phi_local_ikp[sample, layer, node]
                    dphi_ikp[ sample, layer, nodes_in_l[sample, layer] + node
                    ] = (dphi_local_ikp[sample, layer, node] / delta_x_1st)
                    ddphi_ikp[ sample, layer, nodes_in_l[sample, layer] + node
                    ] = (ddphi_local_ikp[sample, layer, node] / delta_x_2nd)

        t_ik = torch.einsum("kp, ikp -> ik", self.weight, phi_ikp)
        dt_ik = torch.einsum("kp, ikp -> ik", self.weight, dphi_ikp)
        ddt_ik = torch.einsum("kp, ikp -> ik", self.weight, ddphi_ikp)

        return {
            "t_ik": t_ik,
            "dt_ik": dt_ik,
            "ddt_ik": ddt_ik,
            "phi_ikp": phi_ikp,
            "dphi_ikp": dphi_ikp,
            "ddphi_ikp": ddphi_ikp,
            "delta_x": delta_x,
        }
class LagrKANNautoouter(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(LagrKANNautoouter, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max

        #self.weight = torch.nn.parameter.Parameter(
        #    torch.ones((self.n_width, self.n_nodes))
        #)
        # Define weight as a parameter
        self.weight = torch.nn.Parameter(torch.empty((self.n_width, self.n_nodes)))

        # Apply Xavier initialization (uniform)
        init.xavier_normal_(self.weight, gain=1.0)
    def lagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)

        p_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        )

        for j in range(n_order + 1):
            p = 1.0
            for m in range(n_order + 1):
                if j != m:
                    p *= (x - nodes[m]) / (nodes[j] - nodes[m])
            p_list[:, :, j] = p

        return p_list

    def dlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)

        dp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        )

        for j in range(n_order + 1):
            y = 0.0
            for i in range(n_order + 1):
                if i != j:
                    k = torch.ones_like(x) / (nodes[j] - nodes[i])
                    for m in range(n_order + 1):
                        if m != i and m != j:
                            k *= (x - nodes[m]) / (nodes[j] - nodes[m])
                    y += k
            dp_list[:, :, j] = y

        return dp_list

    def ddlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)

        ddp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
            )
        )

        for j in range(n_order + 1):
            y = 0.0
            for i in range(n_order + 1):
                if i != j:
                    k_sum = 0.0
                    for m in range(n_order + 1):
                        if m != i and m != j:
                            k_prod = torch.ones_like(x) / (nodes[j] - nodes[m])
                            for n in range(n_order + 1):
                                if n != i and n != j and n != m:
                                    k_prod *= (x - nodes[n]) / (nodes[j] - nodes[n])
                            k_sum += k_prod
                    y += (1 / (nodes[j] - nodes[i])) * k_sum
            ddp_list[:, :, j] = y

        return ddp_list

    def to_ref(self, x_shift, node_l, node_r):
        return 2 * (x_shift - node_l) / (node_r - node_l) - 1

    # unsure for the meaning of the following function (do we shift from -1 to 1 range to 0 to 49 range?)
    def to_shift(self, x):
        """Shift from real line to natural line."""
        x_shift = (self.n_nodes - 1) * (x - self.x_min) / (self.x_max - self.x_min)
        return x_shift

    def forward(self, x):
        """Forward pass for whole batch."""
        if len(x.shape) != 2:
            x = x.unsqueeze(-1)
            x = torch.repeat_interleave(x, self.n_width, -1)
        x_shift = self.to_shift(x)

        id_element_in = torch.floor(x_shift / self.n_order)
        # ensures that all elements of vector id_element_in are within the range of 0 and n_elements - 1
        id_element_in[id_element_in >= self.n_elements] = self.n_elements - 1
        id_element_in[id_element_in < 0] = 0

        # what is the meaning of the following lines?
        nodes_in_l = (id_element_in * self.n_order).to(int)
        nodes_in_r = (nodes_in_l + self.n_order).to(int)

        x_transformed = self.to_ref(x_shift, nodes_in_l, nodes_in_r)
        delta_x = 0.5 * self.n_order * (self.x_max - self.x_min) / (self.n_nodes - 1)

        delta_x_1st = delta_x
        delta_x_2nd = delta_x**2

        phi_local_ikp = self.lagrange(x_transformed, self.n_order)
        dphi_local_ikp = self.dlagrange(x_transformed, self.n_order)
        ddphi_local_ikp = self.ddlagrange(x_transformed, self.n_order)

        phi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes))
        dphi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes))
        ddphi_ikp = torch.zeros((self.n_samples, self.n_width, self.n_nodes))

        for sample in range(self.n_samples):
            for layer in range(self.n_width):
                for node in range(self.n_order + 1):
                    phi_ikp[sample, layer, nodes_in_l[sample, layer] + node] = (
                        phi_local_ikp[sample, layer, node]
                    )
                    dphi_ikp[sample, layer, nodes_in_l[sample, layer] + node] = (
                        dphi_local_ikp[sample, layer, node] / delta_x_1st
                    )
                    ddphi_ikp[sample, layer, nodes_in_l[sample, layer] + node] = (
                        ddphi_local_ikp[sample, layer, node] / delta_x_2nd
                    )

        t_ik = torch.einsum("kp, ikp -> ik", self.weight, phi_ikp)
        dt_ik = torch.einsum("kp, ikp -> ik", self.weight, dphi_ikp)
        ddt_ik = torch.einsum("kp, ikp -> ik", self.weight, ddphi_ikp)

        return {
            "t_ik": t_ik,
            "dt_ik": dt_ik,
            "ddt_ik": ddt_ik,
            "phi_ikp": phi_ikp,
            "dphi_ikp": dphi_ikp,
            "ddphi_ikp": ddphi_ikp,
            "delta_x": delta_x,
        }

class KANN(torch.nn.Module):
    """KANN class with Lagrange polynomials."""

    def __init__(
        self,
        n_width,
        n_order,
        n_elements,
        n_collocation,
        n_samples,
        x_min,
        x_max,
    ):
        """Initialize."""
        super(KANN, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_collocation = n_collocation
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max

        self.inner = LagrKANNautoinner(n_width, n_order, n_elements, n_samples, x_min, x_max)
        self.outer = LagrKANNautoouter(n_width, n_order, n_elements, n_samples, x_min, x_max)
            
        total_params = sum(p.numel() for p in self.parameters())
        nodes_per_width = n_elements * n_order + 1
        print(f"\nTotal parameters: {total_params}")
        print(f"Order: {n_order}")
        print(f"Number of elements: {n_elements}")
        print(f"Nodes per width: {nodes_per_width}")
        print(f"Samples: {n_samples}")
        return None

    def forward(self, x):
        """Forward pass for whole batch."""
        x = self.inner(x)["t_ik"]
        x = self.outer(x)["t_ik"]
        x = torch.einsum("ik -> i", x)
            
        return x

def plot_solution(saveloc,save_name,save,show,x_i, y_hat, y_i, l2, n_width, n_order, n_samples,n_epochs,y0,spacing,x_max, loss_str,jump_loc,log_name): 
    x_i = x_i.detach().view(-1,1).numpy()
    zeros = np.zeros_like(x_i)
    error_abs = np.abs(y_i - y_hat)
    ####################################################################################################################
    # Plotting the analytical and PINN solution
    ####################################################################################################################
    
    plt.figure(2,figsize=(12,7))
    ax = plt.gca()
    ax.plot(
        x_i,
        y_i,
        label="Analytical solution",
        color="black",
        alpha=0.75,
        linewidth=2,
    )
    ax.plot(
        x_i,
        y_hat,
        linestyle="--",
        label="KANN solution",
        color="tab:green",
        linewidth=2
    )
    ax.axvline(x=jump_loc, color='gray', linestyle='--', label = "Jump at x = 1.0", alpha=1.0)
    ax.scatter(x_i, zeros+0.250, color="red", s = 14, label="Collocation Points")
    ax.set_title(f"L2-error: {l2:0.4e}, Width: {n_width}, Order: {n_order}, Samples: {n_samples}, Epochs: {n_epochs}, Spacing: {spacing}, {log_name}, Training loss: {loss_str}")
    ax.set_xticks(np.arange(0, x_max+1, 1))
    #ax.set_ylim(0.2, 1.3)
    #ax.set_ylim(0.35, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="best")
    ax.grid(True)
    #axins = ax.inset_axes([0.4625, 0.25, 0.275, 0.45])
    axins = ax.inset_axes([0.75, 0.1, 0.2, 0.35])
    axins.plot(x_i, y_i, color="black", linewidth=2)
    axins.plot(x_i, y_hat, color="tab:green", linestyle="--", linewidth=2)
    axins.axvline(x=jump_loc, color='gray', linestyle='--')
    axins.set_xlim(jump_loc-0.1, jump_loc+0.1)
    axins.set_xticks([jump_loc-0.1, jump_loc, jump_loc+0.1])
    
    x_i_jumploc = int(np.where(x_i == jump_loc)[0][0])
    y_i_jumpval = y_i[x_i_jumploc].item()
    axins.set_ylim(y_i_jumpval-0.01,y_i_jumpval+0.01)
    axins.set_yticks([y_i_jumpval-0.01,y_i_jumpval,y_i_jumpval+0.01])
    axins.scatter(x_i, zeros+0.25, color="red", s = 14)
    """
    if y0 == 0.8:
        axins.set_ylim(0.2925, 0.3125)# y0 = 0.8
        axins.set_yticks([0.2925, 0.3025, 0.3125])
        axins.scatter(x_i, zeros+0.294, color="red", s = 14)
    elif y0 == 0.9:
        axins.set_ylim(0.32925, 0.34925)# y0 = 0.9
        axins.set_yticks([0.32925, 0.33925, 0.34925])
        axins.scatter(x_i, zeros+0.33, color="red", s = 14)
    elif y0 == 1.0:
        axins.set_ylim(0.3625, 0.3875)# y0 = 1.0
        axins.set_yticks([0.3625, 0.375, 0.3875])
        axins.scatter(x_i, zeros+0.364, color="red", s = 14)
    elif y0 == 1.1:
        axins.set_ylim(0.4025, 0.4225)# y0 = 1.1
        axins.set_yticks([0.4025, 0.4125, 0.4225])
        axins.scatter(x_i, zeros+0.405, color="red", s = 14)
    elif y0 == 1.2:
        axins.set_ylim(0.4385, 0.4585)# y0 = 1.2
        axins.set_yticks([0.4385, 0.4485, 0.4585])
        axins.scatter(x_i, zeros+0.44, color="red", s = 14)
    """
    #axins.set_xticklabels([])
    #axins.set_yticklabels([])

    axins.grid(True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.25")
    if save: 
        plt.savefig(os.path.join(saveloc,f"KANNODE_{save_name}.png"),dpi = 600)
    if show:
        plt.show()
    ####################################################################################################################
    # Plotting the absolute error between the analytical and PINN solution
    ####################################################################################################################
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)
    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bm = fig.add_subplot(gs[1, 1])
    ax_br = fig.add_subplot(gs[1, 2])
    ax_top.plot(x_i, error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_top.axvline(x=1.0, color='black', linestyle='--', label= "Jump location")
    ax_top.legend()
    ax_top.set_title("Absolute error between Analytical and PINN")
    ax_top.set_xlabel("x")
    ax_top.set_xticks(np.arange(0, x_max+1, 1))
    ax_top.set_xlim(0, x_max)
    ax_top.set_ylabel("Absolute error")
    ax_top.grid()
    
    ax_bl.plot(x_i, error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_bl.set_xlabel("x")
    ax_bl.set_xlim(0, 0.95)
    ax_bl.set_ylim(0, 0.01)
    ax_bl.set_ylabel("Absolute error")
    ax_bl.grid()
    
    ax_bm.plot(x_i, error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_bm.set_xlabel("x")
    ax_bm.set_xlim(0.85, 1.15)
    ax_bm.set_ylim(0, 0.02)
    #ax_bm.set_ylabel("Absolute error")
    ax_bm.grid()
    
    ax_br.plot(x_i, error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_br.set_xlabel("x")
    ax_br.set_xlim(1.05, x_max)
    #ax_br.set_ylabel("Absolute error")
    ax_br.grid()
    plt.subplots_adjust(hspace=0.2)
    if save:
        plt.savefig(os.path.join(saveloc,f"KANN_abs_{save_name}.png"),dpi = 600)
    if show:
        plt.show()
    return error_abs
def create_animation(saveloc,save_name,save,show, solutions, col_exact, f_x_exact,n_width, n_order, n_samples,n_epochs,y0,spacing,interval1, jump_loc,log_name):
    col_exact = col_exact.detach()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(col_exact.numpy(), f_x_exact, label="Analytical solution", color="black", alpha=1.0, linewidth=2)
    line, = ax.plot(col_exact.numpy(), solutions[:,0], linestyle="--", label="PINN solution", color="tab:green", linewidth=2)
    ax.axvline(x=jump_loc, color='gray', linestyle='--', label='Jump at x = 1.0')
    ax.scatter(col_exact.numpy(), np.zeros_like(col_exact.numpy())+0.250, color="red", s = 14, label="Collocation Points")
    #ax.set_xlim(0.4, 1)
    ax.set_ylim(0.2, 1.3)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)
    def animate(i):
        line.set_ydata(solutions[:,i])  # update the data
        epoch = i*interval1
        ax.set_title(f"Epoch = {epoch},L2 error = {np.linalg.norm(f_x_exact - solutions[:,i].reshape(-1, 1)):.4e},Width: {n_width}, Order: {n_order}, Samples: {n_samples}, Spacing: {spacing}, {log_name}")
        return line, ax,
    ani = FuncAnimation(fig, animate, frames=solutions.shape[1], interval=100, blit=False, repeat = False)  # Change the interval here
    if save: 
        ani.save(os.path.join(saveloc,f'KANN_animation_{save_name}.mp4'), writer='ffmpeg', fps=5, dpi = 300)  # Specify fps and writer
    if show:
        plt.show()
    return None

def main():
    """Execute main routine."""
    # Read all the parameters form the config file: parameters_phasefield.py
    
    # Model Parameters
    n_width = param.n_width
    n_order = param.n_order
    n_samples = param.n_samples
    n_epochs = param.n_epochs
    spacing = param.spacing
    
    # Calculate the number of elements
    n_elements = int((n_samples - spacing) / n_order)
    
    # Domain Extrema (Lenght of Bar = 2)
    x_min = param.x_min
    x_max = param.x_max
    
    # Constants
    l = param.l
    mat_E = param.mat_E
    Gc = param.Gc
    TOL_irr = param.TOL_irr
    
    # Displacement Loading
    U_max = param.U_max
    load_steps = param.load_steps
    
    # AT-1 or AT-2 model
    AT = param.AT
    
    # Data Saving and Plot Options
    save = param.save
    show = param.show
    enable_animation = param.enable_animation
    saveloc = param.saveloc
    anim_intvl = param.animation_interval
    
    # Calculate values dependent on AT-model
    if AT == "AT1":
        print("\nAT1 model selected\n")
        cw = 8.0/3.0
        dwdalpha = 1.0
    elif AT == "AT2":
        print("\nAT2 model selected\n")
        cw = 2.0
    else:
        raise SystemExit("\nNo valid AT model selected\n")
    
    # Calculate constant penalty value gamma
    #gamma = (Gc / l) * ((1.0 /(TOL_irr**2.0)) - 1.0)
    gamma = (27/(64*TOL_irr**2))
    L = abs(x_max - x_min)
    # Create the collocation points
    x_i = torch.linspace(x_min, x_max, n_samples).requires_grad_()
    
    # Create the models for both displacement u and phase field alpha
    model_u = KANN(
        n_width = n_width,
        n_order = n_order,
        n_elements = n_elements,
        n_collocation = n_samples,
        n_samples = 1,
        x_min = x_min,
        x_max = x_max,
    )
    model_alpha = KANN(
        n_width = n_width,
        n_order = n_order,
        n_elements = n_elements,
        n_collocation = n_samples,
        n_samples = 1,
        x_min = x_min,
        x_max = x_max,
    )
    alpha_prev = torch.zeros_like(x_i)
    # Currently only using a singular loading step for the problem fixed at a small value
    Ut = 0.6
    
    #raise SystemExit("\nNothing Implemented yet that would work\n")
    optimizer = torch.optim.Rprop(list(model_u.parameters()) + list(model_alpha.parameters()), lr=1e-2, step_sizes=(1e-10, 50))

    #optimizer = torch.optim.LBFGS(
    #    list(model_u.parameters()) + list(model_alpha.parameters()), lr = 1e-4
    #    )
    def closure():
        optimizer.zero_grad()
        u_hat = []
        alpha_hat = []
        for sample in range(n_samples):
                # Get the collocation point 
                x = x_i[sample].unsqueeze(-1)
                # Calculate the model prediction for the displacement u and alpha with HBC   
                u = ((x + 0.5)*(x - 0.5)*model_u(x) + (x + 0.5)) * Ut
                alpha = (x + 0.5) * (x - 0.5) * model_alpha(x)
                u_hat.append(u)
                alpha_hat.append(alpha)
                
        u_hat = torch.stack(u_hat)
        alpha_hat = torch.stack(alpha_hat)    
        # compute the derivatives
        dudx = torch.autograd.grad(u_hat.sum(), x_i, create_graph=True)[0]
        dalphadx = torch.autograd.grad(alpha_hat.sum(), x_i,  create_graph=True)[0]
        
        #compute energies
        energy_elastic = 0.5 * ((1.0 - alpha_hat) ** 2) * (dudx ** 2) 
        energy_damage = (1/cw) * (alpha_hat + (l**2) * ((dalphadx) ** 2)) 

        dAlpha = alpha_hat - torch.zeros_like(alpha_hat)
        hist_penalty = nn.ReLU()(-dAlpha) 
        E_hist_penalty = 0.5 * gamma * (hist_penalty**2) 
    
        energy_tot = torch.sum(energy_elastic) + torch.sum(energy_damage) + torch.sum(E_hist_penalty)
        
        loss_energy = torch.log10(energy_tot)
        # Weight regularization: L2 penalty over all parameters
        l2u_reg = 0.0
        for paramu in model_u.parameters():
            l2u_reg += torch.sum(paramu ** 2)
        l2a_reg = 0.0
        for parama in model_alpha.parameters():
            l2a_reg += torch.sum(parama ** 2)
    
        loss = loss_energy + l2u_reg * 1e-5 + l2a_reg* 1e-5
        loss.backward()
        return loss
    
    with tqdm.trange(n_epochs) as pbar1:
        for epoch_idx in pbar1:
            loss = optimizer.step(closure)

            #loss_mean = torch.mean(loss_epoch)
            loss_str = f"{loss.item():0.4e}"

            pbar1.set_postfix(loss=loss_str)
            """
            if enable_animation:
                    if epoch_idx % anim_intvl == 0:
                        with torch.no_grad():
                            sampleeval = 0
                            vec = torch.zeros(n_samples)
                            const = torch.zeros_like(x_i)
                            model_input = torch.stack([x_i, const], dim=1)
                            for sampleeval in range(n_samples):
                                x = x_i[sampleeval].unsqueeze(-1)
                                model_input_i = model_input[sampleeval].unsqueeze(0)
                                y_hat_intermediary_1 = 1.0+x*model_1(model_input_i,epoch_idx,sampleeval)
                                y_hat_intermediary_2 = 1.0+x*model_1(model_input_i,epoch_idx,sampleeval)
                                y_hat_intermediary = ((y_hat_intermediary_1*0.5) + (y_hat_intermediary_2*0.5))
                                vec[sampleeval] = y_hat_intermediary
                            vec = vec.detach().numpy().reshape(-1,1)
                            if epoch_idx == 0:
                                solutions[:,counter] = vec[:,0].flatten()
                            else:
                                solutions[:,counter] = vec[:,0].flatten()
                            counter += 1
            """
    
    print(f"\nTotal Elapsed Time: {pbar1.format_dict['elapsed']:.2f} seconds")
    #model_u.eval()
    #model_alpha.eval()
    x_hat = torch.linspace(x_min, x_max, n_samples, requires_grad=True)
    u_hat = torch.zeros_like(x_hat)
    alpha_hat = torch.zeros_like(x_hat)
    for sample in range(n_samples):
        x = x_hat[sample].unsqueeze(-1)
        u_hati = ((x + 0.5)*(x - 0.5)*model_u(x) + (x + 0.5))*Ut
        alpha_hati = (x + 0.5)*(x - 0.5)*model_alpha(x)
        u_hat[sample] = u_hati
        alpha_hat[sample]  = alpha_hati
    # compute the derivatives
    dudx = torch.autograd.grad(u_hat.sum(), x_hat, create_graph=True)[0]
    dalphadx = torch.autograd.grad(alpha_hat.sum(), x_hat,  create_graph=True)[0]
    
    #compute energies
    energy_elastic = 0.5 * ((1.0 - alpha_hat) ** 2) * (dudx ** 2) 
    energy_damage = (1/cw) * (alpha_hat + (l**2) * ((dalphadx) ** 2))
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(x_i.detach().numpy(), u_hat.detach().numpy(), label="u")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("u")
    axs[0].set_title("Displacement")
    axs[0].set_ylim(-0.1, 1.1)
    axs[0].grid()
    
    axs[1].plot(x_i.detach().numpy(), alpha_hat.detach().numpy(), label="alpha")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("alpha")
    axs[1].set_title("Phase Field")
    axs[1].grid()
    axs[1].set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.show()
    x_np = x_i.detach().cpu().numpy()
    u_np = u_hat.detach().cpu().numpy()
    alpha_np = alpha_hat.detach().cpu().numpy()
    e_el_np = energy_elastic.detach().cpu().numpy()
    e_dam_np = energy_damage.detach().cpu().numpy()

    # Use f-strings to incorporate 'my_var' into the filenames
    npz_path = fr"E:\ETH\Master\25HS_MA\Data_Phasefield\KANNoutputphasefieldweak_UP{Ut}.npz"
    csv_path = fr"E:\ETH\Master\25HS_MA\Data_Phasefield\KANNoutputphasefieldweak_UP{Ut}.csv"

    # Save to NPZ
    np.savez(npz_path, x=x_np, u=u_np, alpha=alpha_np, e_el=e_el_np, e_dam=e_dam_np)

    # Optionally also save as CSV
    data_to_save = np.hstack([x_np, u_np, alpha_np, e_el_np, e_dam_np])
    np.savetxt(csv_path, data_to_save, delimiter=",", header="x,u,alpha,el, dam", comments="")

    print(f"Saved NPZ to: {npz_path}")
    print(f"Saved CSV to: {csv_path}")
    return None

if __name__ == "__main__":
    main()