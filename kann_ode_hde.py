"""Pytorch implementation of KANN."""

import matplotlib.pyplot as plt
import torch
import tqdm
import parameters_ode
import pandas as pd
import numpy as np
from scipy.integrate import odeint


# set the default data type for tensors to double precision
torch.set_default_dtype(torch.float64)

class LagrangeKANN(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(LagrangeKANN, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max

        self.weight = torch.nn.parameter.Parameter(
            torch.zeros((self.n_width, self.n_nodes))
        )

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

    def to_ref(self, x, node_l, node_r):
        """Transform to reference base."""
        return (x - (0.5 * (node_l + node_r))) / (0.5 * (node_r - node_l))

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
        n_samples,
        x_min,
        x_max,
        autodiff,
        speedup
    ):
        """Initialize."""
        super(KANN, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max
        self.autodiff = autodiff
        self.speedup = speedup

        self.inner = LagrangeKANN(n_width, n_order, n_elements, n_samples, x_min, x_max)
        self.outer = LagrangeKANN(n_width, n_order, n_elements, n_samples, x_min, x_max)
        
        total_params = sum(p.numel() for p in self.parameters())
        nodes_per_width = n_elements * n_order + 1
        print(f"\nTotal parameters: {total_params}")
        print(f"Order: {n_order}")
        print(f"Number of elements: {n_elements}")
        print(f"Nodes per width: {nodes_per_width}")
        print(f"Samples: {n_samples}")
        print(f"Autodiff: {autodiff}\n")
        return None

    def forward(self, x):
        """Forward pass for whole batch."""
        x = self.inner(x)["t_ik"]
        x = self.outer(x)["t_ik"]

        x = torch.einsum("ik -> i", x)

        return x

    def residual(self, x, epoch,sample):
        """Calculate residual."""

        y = 1 + x * self.forward(x,epoch,sample)

        inner_dict = self.inner(x)
        outer_dict = self.outer(inner_dict["t_ik"])

        inner_dt_ik = inner_dict["dt_ik"]

        outer_t_ik = outer_dict["t_ik"]
        outer_dt_ik = outer_dict["dt_ik"]

        dy = torch.einsum(
            "i, ik, ik -> i", x, outer_dt_ik, inner_dt_ik
        ) + torch.einsum("ik -> i", outer_t_ik)

        residual = dy - y
        return residual

    def linear_system(self, x, y_true,_,sample):
        """Compute Ax=b."""
        if self.speedup:
            inner_dict = self.inner(x,_,sample)
            outer_dict = self.outer(inner_dict["t_ik"])

            inner_dt_ik = inner_dict["dt_ik"]
            #inner_phi_ikp = inner_dict["phi_ikp"]
            inner_phi_ikp = self.inner.phi_ikp_inner[sample:(sample+1), :, :]
            #inner_dphi_ikp = inner_dict["dphi_ikp"]
            inner_dphi_ikp = self.inner.dphi_ikp_inner[sample:(sample+1), :, :]

            outer_t_ik = outer_dict["t_ik"]
            outer_dt_ik = outer_dict["dt_ik"]
            outer_ddt_ik = outer_dict["ddt_ik"]
            outer_phi_ikl = outer_dict["phi_ikp"]
            outer_dphi_ikl = outer_dict["dphi_ikp"]

            A_outer = (
                torch.einsum("i, ikl, ik -> ikl", x, outer_dphi_ikl, inner_dt_ik)
                - torch.einsum("i, ikl -> ikl", x, outer_phi_ikl)
                + torch.einsum("ikl -> ikl", outer_phi_ikl)
            )

            A_inner = (
                torch.einsum(
                    "i, ik, ik, ikp -> kp",
                    x,
                    outer_ddt_ik,
                    inner_dt_ik,
                    inner_phi_ikp,
                )
                + torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_dphi_ikp)
                + torch.einsum("ik, ikp -> ikp", outer_dt_ik, inner_phi_ikp)
                - torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_phi_ikp)
            )
            
            y1 = torch.einsum("ik -> i", outer_t_ik)
            y = 1 + x * y1

            dy = torch.einsum(
                "i, ik, ik -> i", x, outer_dt_ik, inner_dt_ik
            ) + torch.einsum("ik -> i", outer_t_ik)

            b = dy - y
            
        else:
            b = self.residual(x, y_true,_,sample)
            inner_dict = self.inner(x)
            outer_dict = self.outer(inner_dict["t_ik"])

            inner_dt_ik = inner_dict["dt_ik"]
            inner_phi_ikp = inner_dict["phi_ikp"]
            inner_dphi_ikp = inner_dict["dphi_ikp"]

            outer_dt_ik = outer_dict["dt_ik"]
            outer_ddt_ik = outer_dict["ddt_ik"]
            outer_phi_ikl = outer_dict["phi_ikp"]
            outer_dphi_ikl = outer_dict["dphi_ikp"]

            A_outer = (
                torch.einsum("i, ikl, ik -> ikl", x, outer_dphi_ikl, inner_dt_ik)
                - torch.einsum("i, ikl -> ikl", x, outer_phi_ikl)
                + torch.einsum("ikl -> ikl", outer_phi_ikl)
            )

            A_inner = (
                torch.einsum(
                    "i, ik, ik, ikp -> kp",
                    x,
                    outer_ddt_ik,
                    inner_dt_ik,
                    inner_phi_ikp,
                )
                + torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_dphi_ikp)
                + torch.einsum("ik, ikp -> ikp", outer_dt_ik, inner_phi_ikp)
                - torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_phi_ikp)
            )

        return {"A_inner": A_inner, "A_outer": A_outer, "b": b}
def ode_hde(y0,x):
    
    def heaviside(x):
        return 1 if x >= 1 else 0
    
    def system_heaviside(y,x):
        H = heaviside(x)
        dydx = H - y
        return dydx
    
    y_heaviside = odeint(system_heaviside, y0, x)
    
    return y_heaviside
def heaviside_fct(x):
    tensor = torch.where(x >= 1, torch.ones_like(x), torch.zeros_like(x))
    return tensor

def save_excel(values, autodiff, regression, speedup, prestop):
    
    np_array = values.detach().numpy()
    df = pd.DataFrame(np_array)
    if values.shape[1] == 9:
        if autodiff:
            exc_file_n1 = "auto_"
        else:
            exc_file_n1 = "man_"
        if regression:
            exc_file_n2 = "reg.xlsx"
        else:
            exc_file_n2 = "ode.xlsx"
        if speedup:
            exc_file_n3 = "speedup_"
        else:
            exc_file_n3 = "no_speedup_"
        exc_file = "values_" + exc_file_n1 +exc_file_n3 + exc_file_n2
        df.to_excel(
            exc_file,
            index=False,
            header=[
                "n_samples",
                "n_width",
                "n_order",
                "tol",
                "loss_mean",
                "l2-error",
                "epoch",
                "runtime",
                "tickrate",
            ],
        )
    else:
        if prestop:
            exc_file = "values_losstracking_ps.xlsx" 
            df.to_excel(
                exc_file,
                index=False,
                header=["epochs", "loss_mean"],
            )
        else:
            exc_file = "values_losstracking.xlsx" 
            df.to_excel(
                exc_file,
                index=False,
                header=["epochs", "loss_mean"],
            )
    print(f"Values saved to excel file: {exc_file}")

    return None
def plot_solution(x_i, y_hat, y_i, l2): 
    x_i = x_i.detach().view(-1,1).numpy()
    # Plotting
    error_abs = np.abs(y_i - y_hat)
    plt.figure(2,figsize=(8, 4))
    plt.rcParams.update({
    'font.size': 10,              # Base font size
    'axes.labelsize': 11,         # Axis labels
    'xtick.labelsize': 10,        # X-axis tick labels
    'ytick.labelsize': 10,        # Y-axis tick labels
    'legend.fontsize': 10,        # Legend
    'figure.titlesize': 12        # Figure title
    })
    plt.plot(
        x_i,
        y_i,
        label="Exact solution",
        color="black",
        alpha=1.0,
        linewidth=2,
    )
    plt.plot(
        x_i,
        y_hat,
        linestyle="--",
        label="KANN solution",
        color="tab:green",
        linewidth=2
    )
    plt.axvline(x=1, color='gray', linestyle='--')
    plt.title(f"L2-error: {l2:0.4e}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.savefig("KANNODE2.png")
    plt.figure(3,figsize=(8, 4))
    plt.plot(
        x_i,
        error_abs,
        label="Absolute error",
        color="red",
        alpha=1.0,
        linewidth=2,
    )
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("Absolute error")
    plt.title("Absolute error between Analytical and PINN")
    plt.savefig("KANNODE2_abs.png")
    plt.show()
#
    return None
def collocationpoints(total_values):
    nval1 = total_values // 5
    nval2 = total_values - nval1
    log_values = torch.logspace(0, torch.log10(torch.tensor(5.0)), steps=nval2, base=10)
    log_values = torch.linspace(1, 5, steps=nval2)
    # Second example: Logarithmic spacing between 1 and 0
    log_values2 = torch.logspace(0, -2, steps=nval1, base=10)
    log_values2 = 1 - log_values2  # Flip to go from 1 to 0
    combined = torch.cat((log_values2, log_values))
    combined = combined.detach().numpy()
    return combined
def main():
    """Execute main routine."""
    n_width = parameters_ode.n_width
    n_order = parameters_ode.n_order
    n_samples = parameters_ode.n_samples
    n_epochs = parameters_ode.n_epochs
    tol = parameters_ode.tol
    autodiff = parameters_ode.autodiff
    runs = parameters_ode.runs
    speedup = parameters_ode.speedup
    prestop = parameters_ode.prestop
    save = parameters_ode.save
    track_values = parameters_ode.track_values
    n_elements = int((n_samples - 2) / n_order)
    
    # initialize tensor to store values
    if track_values:
        values = torch.zeros((runs, 9))
        loss_tracking = torch.zeros((int(n_epochs / 10 + 2),2))
        rval = 0
    
    for run in range(runs):
        print(f"\nrun at iteration {run+1}")
        same_loss_counter = 0
        previous_loss = 0
        
        # define range and initial value for the ODE
        x_min = 0.0
        x_max = 5.0
        y0 = 1
        logpoints = False
        # vector of n_samples from x_min to x_max
        #x_i = collocationpoints(n_samples).requires_grad_(True)
        #x_i = torch.linspace(x_min, x_max, n_samples).requires_grad_(True)
        #x_i = torch.linspace(x_min, x_max, n_samples)
        #x_i_real = x_i.detach()
        # create sample Data
        #y_i = ode_hde(y0,x_i)
        
        if logpoints:
            col_points_log = collocationpoints(n_samples)
            f_x_exact = ode_hde(y0, col_points_log)
            if False:
                plt.figure(figsize=(8, 4))
                plt.plot(col_points_log, f_x_exact, label="Exact solution", color="blue", alpha=1.0, linewidth=2)
                plt.scatter(col_points_log, np.zeros_like(col_points_log)+0.1, label="Initial condition", color="blue", alpha=1.0, linewidth=2)
                plt.grid()
                plt.show()
            col_points = torch.from_numpy(col_points_log).requires_grad_(True)
        else:
            col_points = np.linspace(x_min, x_max, n_samples)
            f_x_exact = ode_hde(y0, col_points)
            if False:
                plt.figure(figsize=(8, 4))
                plt.plot(col_points, f_x_exact, label="Exact solution", color="blue", alpha=1.0, linewidth=2)
                plt.scatter(col_points, np.zeros_like(col_points)+0.1, label="Initial condition", color="blue", alpha=1.0, linewidth=2)
                plt.grid()
                plt.show()
            col_points = torch.from_numpy(col_points).requires_grad_(True)
        
        x_i = col_points
        y_i = f_x_exact
        # create heaviside function for ODE HBC
        with torch.no_grad():
            heaviside_tensor = heaviside_fct(x_i)
            #heavyside_tensor = heaviside_fct(col_points)
            if False:
                plt.figure(figsize=(8, 4))
                plt.plot(col_points, heaviside_tensor, label="Heaviside function", color="black", alpha=1.0, linewidth=2)
                plt.scatter(col_points, np.zeros_like(col_points)+0.1, label="Initial condition", color="blue", alpha=1.0, linewidth=2)
                plt.grid()
                plt.show()
            
        sample = 0
        _ = 0
        loss_mean = 0
        # initialize the model
        model = KANN(
            n_width=n_width,
            n_order=n_order,
            n_elements=n_elements,
            n_samples=1,
            x_min=x_min,
            x_max=x_max,
            autodiff=autodiff,
            speedup=speedup
        )

        with tqdm.trange(n_epochs) as pbar1:
            for _ in pbar1:
                lr_epoch = torch.zeros((n_samples,))
                loss_epoch = torch.zeros((n_samples,))
                # start looping over each training sample (50 times (0-49))
                for sample in range(n_samples):    
                    loss = 0  
                    x = x_i[sample].unsqueeze(-1)
                    h = heaviside_tensor[sample].unsqueeze(-1)
                    if autodiff is True:
                        y = 1 + x * model(x)
                        dydx = torch.autograd.grad(
                            y, x, torch.ones_like(x), create_graph=True, materialize_grads=True
                        )[0]
                        residual = (dydx + y - h)
                    else:# manual differentiation
                        print("Manual differentiation not implemented")

                    loss = torch.mean(torch.square(residual))

                    if autodiff is True:
                        g_lst = torch.autograd.grad(
                            outputs=residual,
                            inputs=model.parameters(),
                        )
                        norm = torch.linalg.norm(torch.hstack(g_lst)) ** 2
                    else:
                        print("Manual differentiation not implemented")
                        
                    # Kaczmarz update
                    for p, g in zip(model.parameters(), g_lst):
                        update = (residual / norm) * torch.squeeze(g)
                        p.data -= update

                    loss_epoch[sample] = loss

                loss_mean = torch.mean(loss_epoch)
                loss_str = f"{loss_mean.item():0.4e}"

                pbar1.set_postfix(loss=loss_str)
                if loss.item() < 1e-3:
                    break
                Tickrate = pbar1.format_dict['rate']
            
        print(f"\nTotal Elapsed Time: {pbar1.format_dict['elapsed']:.2f} seconds")
        if same_loss_counter > 20:
            print(f"Same loss counter: {same_loss_counter}")

        # create the final output of the model
        with torch.no_grad():
            y_hat = torch.zeros_like(x_i)
            for sample in range(n_samples):
                x = x_i[sample].unsqueeze(-1)
                y_hat[sample] = 1 + x * model(x)
        y_hat = y_hat.view(-1,1).numpy()
        l2 = np.linalg.norm(y_i - y_hat)
        print(f"L2-error: {l2.item():0.4e}")

    plot_solution(x_i,y_hat, y_i, l2)

    return None

if __name__ == "__main__":
    main()