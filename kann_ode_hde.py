"""Pytorch implementation of KANN."""

import matplotlib.pyplot as plt
import torch
import torch.nn.init as init
import tqdm
import parameters_ode
import pandas as pd
import numpy as np
import os
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import datetime

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

class LagrangeKANNmaninner(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_collocation,n_samples, x_min, x_max):
        """Initialize."""
        super(LagrangeKANNmaninner, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_collocation = n_collocation
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max

        self.weight = torch.nn.parameter.Parameter(
            torch.zeros((self.n_width, self.n_nodes))
        )
        #init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        self.phi_ikp_inner = torch.zeros((self.n_collocation, self.n_width, self.n_nodes))
        self.dphi_ikp_inner = torch.zeros((self.n_collocation, self.n_width, self.n_nodes))
        self.ddphi_ikp_inner = torch.zeros((self.n_collocation, self.n_width, self.n_nodes))

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

    def to_shift(self, x):
        """Shift from real line to natural line."""
        x_shift = (self.n_nodes - 1) * (x - self.x_min) / (self.x_max - self.x_min)
        return x_shift

    def forward(self, x, _, sample):
        if _ == 0:
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
            self.delta_x_inner = 0.5 * self.n_order * (self.x_max - self.x_min) / (self.n_nodes - 1)

            delta_x_1st = self.delta_x_inner
            delta_x_2nd = self.delta_x_inner**2

            phi_local_ikp = self.lagrange(x_transformed, self.n_order)
            dphi_local_ikp = self.dlagrange(x_transformed, self.n_order)
            ddphi_local_ikp = self.ddlagrange(x_transformed, self.n_order)

            for layer in range(self.n_width):
                for node in range(self.n_order + 1):
                    self.phi_ikp_inner[sample, layer, nodes_in_l[0, layer] + node] = (
                        phi_local_ikp[0, layer, node]
                    )
                    self.dphi_ikp_inner[sample, layer, nodes_in_l[0, layer] + node] = (
                        dphi_local_ikp[0, layer, node] / delta_x_1st
                    )
                    self.ddphi_ikp_inner[sample, layer, nodes_in_l[0, layer] + node] = (
                        ddphi_local_ikp[0, layer, node] / delta_x_2nd
                    )
         
        with torch.no_grad():
            phi_cut = self.phi_ikp_inner[sample:(sample+1), :, :]
            dphi_cut = self.dphi_ikp_inner[sample:(sample+1), :, :]
            ddphi_cut = self.ddphi_ikp_inner[sample:(sample+1), :, :]

        t_ik = torch.einsum("kp, ikp -> ik", self.weight, phi_cut)
        dt_ik = torch.einsum("kp, ikp -> ik", self.weight, dphi_cut)
        ddt_ik = torch.einsum("kp, ikp -> ik", self.weight, ddphi_cut)

        return {
            "t_ik": t_ik,
            "dt_ik": dt_ik,
            "ddt_ik": ddt_ik,
            "phi_ikp": self.phi_ikp_inner,
            "dphi_ikp": self.dphi_ikp_inner,
            "ddphi_ikp": self.ddphi_ikp_inner,
            "delta_x": self.delta_x_inner,
        }       
class LagrangeKANNmanouter(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(LagrangeKANNmanouter, self).__init__()
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
        #init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
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

class LagrKANNautoinner(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max, ndim_in):
        """Initialize."""
        super(LagrKANNautoinner, self).__init__()
        self.n_width = n_width
        self.n_order = n_order
        self.n_elements = n_elements
        self.n_nodes = n_elements * n_order + 1
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max
        self.ndim_in = ndim_in

        self.weight = torch.nn.parameter.Parameter(
            torch.zeros((self.n_width, self.n_nodes,self.ndim_in))
        )

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
                x.shape[2],
            )
        )
        for i in range(self.ndim_in):
            for j in range(n_order + 1):
                p = 1.0
                for m in range(n_order + 1):
                    if j != m:
                        p *= (x[:,:,i] - nodes[m]) / (nodes[j] - nodes[m])
                p_list[:, :, j,i] = p

        return p_list

    def dlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)

        dp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
                x.shape[2],
            )
        )
        for h in range(self.ndim_in):
            for j in range(n_order + 1):
                y = 0.0
                for i in range(n_order + 1):
                    if i != j:
                        k = torch.ones_like(x[:,:,h]) / (nodes[j] - nodes[i])
                        for m in range(n_order + 1):
                            if m != i and m != j:
                                k *= (x[:,:,h] - nodes[m]) / (nodes[j] - nodes[m])
                        y += k
                dp_list[:, :, j,h] = y

        return dp_list

    def ddlagrange(self, x, n_order):
        """Lagrange polynomials."""
        nodes = torch.linspace(-1.0, 1.0, n_order + 1)

        ddp_list = torch.zeros(
            (
                x.shape[0],
                x.shape[1],
                n_order + 1,
                x.shape[2],
            )
        )
        for h in range(self.ndim_in):
            for j in range(n_order + 1):
                y = 0.0
                for i in range(n_order + 1):
                    if i != j:
                        k_sum = 0.0
                        for m in range(n_order + 1):
                            if m != i and m != j:
                                k_prod = torch.ones_like(x[:,:,h]) / (nodes[j] - nodes[m])
                                for n in range(n_order + 1):
                                    if n != i and n != j and n != m:
                                        k_prod *= (x[:,:,h] - nodes[n]) / (nodes[j] - nodes[n])
                                k_sum += k_prod
                        y += (1 / (nodes[j] - nodes[i])) * k_sum
                ddp_list[:, :, j,h] = y

        return ddp_list

    def to_ref(self, x_shift, node_l, node_r):
        return 2 * (x_shift-node_l)/(node_r-node_l) - 1

    # unsure for the meaning of the following function (do we shift from -1 to 1 range to 0 to 49 range?)
    def to_shift(self, x):
        x_shift = (self.n_nodes - 1) * (x - self.x_min) / (self.x_max - self.x_min)
        return x_shift

    def forward(self, x):
        """Forward pass for whole batch."""
        if len(x.shape) != self.ndim_in:
            raise SystemExit("Check the dimensions of the input")
        
        #if len(x.shape) != 2:
        #    x = x.unsqueeze(-1)
        # Change Shape of x and repeat it for the number of widths [n_batch, n_width, n_dim]
        x = x.unsqueeze(1).repeat(1, self.n_width, 1)
        #x = torch.repeat_interleave(x, self.n_width, 0)
        x_shift = self.to_shift(x)

        id_element_in = torch.floor(x_shift / self.n_order)
        # ensures that all elements of vector id_element_in are within the range of 0 and n_elements - 1
        id_element_in[id_element_in >= self.n_elements] = self.n_elements - 1
        id_element_in[id_element_in < 0] = 0

        #
        nodes_in_l = (id_element_in * self.n_order).to(int)
        nodes_in_r = (nodes_in_l + self.n_order).to(int)

        x_transformed = self.to_ref(x_shift, nodes_in_l, nodes_in_r)
        delta_x = 0.5 * self.n_order * (self.x_max - self.x_min) / (self.n_nodes - 1)
        delta_x_1st = delta_x
        delta_x_2nd = delta_x**2

        phi_local_ikpj = self.lagrange(x_transformed, self.n_order)
        dphi_local_ikpj = self.dlagrange(x_transformed, self.n_order)
        ddphi_local_ikpj = self.ddlagrange(x_transformed, self.n_order)

        phi_ikpj = torch.zeros((self.n_samples, self.n_width, self.n_nodes, self.ndim_in))
        dphi_ikpj = torch.zeros((self.n_samples, self.n_width, self.n_nodes, self.ndim_in))
        ddphi_ikpj = torch.zeros((self.n_samples, self.n_width, self.n_nodes, self.ndim_in))
        for sample in range(self.n_samples):
            for dim in range(self.ndim_in):
                for layer in range(self.n_width):
                    for node in range(self.n_order + 1):
                        phi_ikpj[sample, layer, nodes_in_l[sample, layer,dim] + node, dim] = (
                            phi_local_ikpj[sample, layer, node, dim]
                        )
                        dphi_ikpj[sample, layer, nodes_in_l[sample, layer,dim] + node, dim] = (
                            dphi_local_ikpj[sample, layer, node, dim] / delta_x_1st
                        )
                        ddphi_ikpj[sample, layer, nodes_in_l[sample, layer,dim] + node, dim] = (
                            ddphi_local_ikpj[sample, layer, node, dim] / delta_x_2nd
                        )

        t_ik = torch.einsum("kpj, ikpj -> ik", self.weight, phi_ikpj)
        dt_ik = torch.einsum("kpj, ikpj -> ik", self.weight, dphi_ikpj)
        ddt_ik = torch.einsum("kpj, ikpj -> ik", self.weight, ddphi_ikpj)

        return {
            "t_ik": t_ik,
            "dt_ik": dt_ik,
            "ddt_ik": ddt_ik,
            "phi_ikp": phi_ikpj,
            "dphi_ikp": dphi_ikpj,
            "ddphi_ikp": ddphi_ikpj,
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
        ndim_in,
        autodiff,
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
        self.ndim_in = ndim_in
        self.autodiff = autodiff

        if autodiff:
            self.inner = LagrKANNautoinner(n_width, n_order, n_elements, n_samples, x_min, x_max, ndim_in)
            self.outer = LagrKANNautoouter(n_width, n_order, n_elements, n_samples, x_min, x_max)
        else: 
            self.inner = LagrangeKANNmaninner(n_width, n_order, n_elements, n_collocation, n_samples, x_min, x_max)
            self.outer = LagrangeKANNmanouter(n_width, n_order, n_elements, n_samples, x_min, x_max)
        
            
        total_params = sum(p.numel() for p in self.parameters())
        nodes_per_width = n_elements * n_order + 1
        print(f"\nTotal parameters: {total_params}")
        print(f"Order: {n_order}")
        print(f"Number of elements: {n_elements}")
        print(f"Nodes per width: {nodes_per_width}")
        print(f"Samples: {n_samples}")
        print(f"Autodiff: {autodiff}\n")
        return None

    def forward(self, x, epoch, sample):
        """Forward pass for whole batch."""
        if self.autodiff is True:
            x = self.inner(x)["t_ik"]
            x = self.outer(x)["t_ik"]
            x = torch.einsum("ik -> i", x)
        else:
            x = self.inner(x,epoch,sample)["t_ik"]
            x = self.outer(x)["t_ik"]
            x = torch.einsum("ik -> i", x)
            
        return x

    def linear_system(self, x, epoch,sample,h,y0):
        """Calculate linear system."""
        inner_dict = self.inner(x,epoch,sample)
        outer_dict = self.outer(inner_dict["t_ik"])

        inner_dt_ik = inner_dict["dt_ik"]
        inner_phi_ikp = self.inner.phi_ikp_inner[sample:(sample+1), :, :]
        inner_dphi_ikp = self.inner.dphi_ikp_inner[sample:(sample+1), :, :]

        outer_t_ik = outer_dict["t_ik"]
        outer_dt_ik = outer_dict["dt_ik"]
        outer_ddt_ik = outer_dict["ddt_ik"]
        outer_phi_ikl = outer_dict["phi_ikp"]
        outer_dphi_ikl = outer_dict["dphi_ikp"]

        A_outer = (
            torch.einsum("i, ikl, ik -> ikl", x, outer_dphi_ikl, inner_dt_ik)
            + torch.einsum("i, ikl -> ikl", x, outer_phi_ikl)
            + torch.einsum("ikl -> ikl", outer_phi_ikl)
        )

        A_inner = (
            torch.einsum("i, ik, ik, ikp -> kp",x,outer_ddt_ik,inner_dt_ik,inner_phi_ikp,)
            + torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_dphi_ikp)
            + torch.einsum("ik, ikp -> ikp", outer_dt_ik, inner_phi_ikp)
            + torch.einsum("i, ik, ikp -> ikp", x, outer_dt_ik, inner_phi_ikp)
        )
        
        y1 = torch.einsum("ik -> i", outer_t_ik)
        y = y0 + x * y1
        dy = torch.einsum("i, ik, ik -> i", x, outer_dt_ik, inner_dt_ik) + y1
        b = dy + y - h
        return {"A_inner": A_inner, "A_outer": A_outer, "b": b, "y": y, "dy": dy}

def ode_hde(y0,x, jump_loc):
    
    def heaviside(x,jump_loc):
        return 1 if x >= jump_loc else 0
    
    def system_heaviside(y,x):
        H = heaviside(x,jump_loc)
        dydx = H - y
        return dydx
    
    y_heaviside = odeint(system_heaviside, y0, x)
    
    return y_heaviside
def heaviside_fct(x,jump_loc):
    tensor = torch.where(x >= jump_loc, torch.ones_like(x), torch.zeros_like(x))
    return tensor
def preprocess_data(x,n_order,n_elements,x_min,x_max):
    n_nodes = n_elements * n_order + 1
    x_shift = (n_nodes - 1) * (x - x_min) / (x_max - x_min)
    id_element_in = torch.floor(x_shift / n_order)
    id_element_in[id_element_in >= n_elements] = n_elements - 1
    id_element_in[id_element_in < 0] = 0
    nodes_in_l = (id_element_in * n_order).to(int)
    nodes_in_r = (nodes_in_l + n_order).to(int)
    nl_min = torch.min(nodes_in_l)
    nr_max = torch.max(nodes_in_r)
    edge_nodes = torch.tensor([nl_min, nr_max])
    return edge_nodes

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
def plot_solution(saveloc,save,show,x_i, y_hat, y_i, l2, n_width, n_order, n_samples,n_epochs,y0,spacing,x_max, loss_str): 
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
    ax.axvline(x=1.00, color='gray', linestyle='--', label = "Jump at x = 1.0", alpha=1.0)
    ax.scatter(x_i, zeros+0.250, color="red", s = 14, label="Collocation Points")
    ax.set_title(f"L2-error: {l2:0.4e}, Width: {n_width}, Order: {n_order}, Samples: {n_samples}, Epochs: {n_epochs}, Spacing: {spacing}, training loss: {loss_str}")
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
    axins.axvline(x=1.0, color='gray', linestyle='--')
    axins.set_xlim(0.9, 1.1)
    axins.set_xticks([0.9, 1.0, 1.1])
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

    #axins.set_xticklabels([])
    #axins.set_yticklabels([])

    axins.grid(True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.25")
    if save: 
        plt.savefig(os.path.join(saveloc,f"KANNODE_w{n_width}o{n_order}s{n_samples}sp{spacing}e{n_epochs}y{y0}.png"),dpi = 600)
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
        plt.savefig(os.path.join(saveloc,f"KANN_abs_w{n_width}o{n_order}s{n_samples}sp{spacing}e{n_epochs}y{y0}.png"),dpi = 600)
    if show:
        plt.show()
    return error_abs
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
def create_animation(saveloc,save,show, solutions, col_exact, f_x_exact,n_width, n_order, n_samples,n_epochs,y0,spacing,x_max,interval1):
    col_exact = col_exact.detach()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(col_exact.numpy(), f_x_exact, label="Analytical solution", color="black", alpha=1.0, linewidth=2)
    line, = ax.plot(col_exact.numpy(), solutions[:,0], linestyle="--", label="PINN solution", color="tab:green", linewidth=2)
    ax.axvline(x=1.0, color='gray', linestyle='--', label='Jump at x = 1.0')
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
        ax.set_title(f"Epoch = {epoch}, L2 error = {np.linalg.norm(f_x_exact - solutions[:,i].reshape(-1, 1)):.4e},Width: {n_width}, Order: {n_order}, Samples: {n_samples}, Epochs: {n_epochs}")
        return line, ax,

    ani = FuncAnimation(fig, animate, frames=solutions.shape[1], interval=100, blit=False, repeat = False)  # Change the interval here
    if save: 
        ani.save(os.path.join(saveloc,f'KANN_animation_w{n_width}o{n_order}s{n_samples}sp{spacing}e{n_epochs}y{y0}.mp4'), writer='ffmpeg', fps=5, dpi = 600)  # Specify fps and writer
    if show:
        plt.show()
    return None
def compute_nonuniform_points(x_min=0, x_max=10, cluster1=1, cluster2=2, 
                                n_points_A=34, n_points_B=34, n_points_C=65):
    """
    Compute nonuniformly spaced points between x_min and x_max with
    higher density near cluster1 and between cluster1 and cluster2.
    
    Parameters:
      x_min         : Lower bound of the overall interval.
      x_max         : Upper bound of the overall interval.
      cluster1      : A first cluster point (should satisfy x_min < cluster1 < cluster2).
      cluster2      : A second cluster point (should satisfy cluster1 < cluster2 < x_max).
      n_points_A    : Number of points in the interval [x_min, cluster1].
      n_points_B    : Number of points in the interval [cluster1, cluster2].
      n_points_C    : Number of points in the interval [cluster2, x_max].
      
    Returns:
      A sorted NumPy array of unique points.
    """
    # Ensure proper ordering of parameters.
    assert x_min < cluster1 < cluster2 < x_max, "Ensure x_min < cluster1 < cluster2 < x_max"
    
    # Segment A: [x_min, cluster1]
    # We use a quadratic mapping that clusters points toward cluster1.
    s = np.linspace(0, 1, n_points_A)
    # Mapping: when s=0, x = x_min; when s=1, x = cluster1.
    x_A = cluster1 - (cluster1 - x_min) * (1 - s)**2

    # Segment B: [cluster1, cluster2]
    # Here we cluster points near cluster1 by mapping quadratically.
    y = np.linspace(0, 1, n_points_B)
    # Mapping: when y=0, x = cluster1; when y=1, x = cluster2.
    x_B = cluster1 + (cluster2 - cluster1) * y**2
    #x_B = cluster1 + (x_max - cluster1) * y**2
    #x_B = np.linspace(cluster1, x_max, n_points_B)

    # Segment C: [cluster2, x_max]
    # A quadratic mapping for a more spread-out distribution in this segment.
    z = np.linspace(0, 1, n_points_C)
    # Mapping: when z=0, x = cluster2; when z=1, x = x_max.
    #x_C = cluster2 + (x_max - cluster2) * z**2
    x_C = np.linspace(cluster2, x_max, n_points_C)
    
    # Combine segments and remove duplicate endpoints (cluster1 and cluster2).
    
    x_all = np.unique(np.concatenate((x_A, x_B, x_C)))
    
    return x_all

def main():
    """Execute main routine."""
    # Read all the parameters form the config file
    n_width = parameters_ode.n_width
    n_order = parameters_ode.n_order
    n_samples = parameters_ode.n_samples
    n_epochs = parameters_ode.n_epochs
    autodiff = parameters_ode.autodiff
    save = parameters_ode.save
    show = parameters_ode.show
    saveloc = parameters_ode.saveloc
    interval1 = parameters_ode.interval
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    # Define range and initial value for the ODE [could be put in the config file]
    x_min = 0.0
    x_max = 10.0
    y0 = 1.0
    jump_loc = 1.0
    
    # Prametrized ODE needs input dimension [could be put in the config file]
    ndim_in = 2
    
    # Define Spacing of the elements
    spacing = 2
    n_elements = int((n_samples - spacing) / n_order)
    
    # Define if collocation points are uniformly or non-uniformly distributed [Move True/Flase in conifg file]
    logpoints = False
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
        const_points = np.zeros_like(col_points)
        #const_points = np.ones_like(col_points)*0.5
        #col_points = compute_nonuniform_points()
        f_x_exact = ode_hde(y0, col_points,jump_loc)
        if False:
            plt.figure(figsize=(8, 4))
            plt.plot(col_points, f_x_exact, label="Exact solution", color="blue", alpha=1.0, linewidth=2)
            plt.scatter(col_points, np.zeros_like(col_points)+0.1, label="Initial condition", color="blue", alpha=1.0, linewidth=2)
            plt.grid()
            plt.show()
        col_points = torch.from_numpy(col_points).requires_grad_(True)
        const_points = torch.from_numpy(const_points).requires_grad_(True)
    
    # Rename all variables for the ODE
    x_i = col_points
    y_i = f_x_exact
    const_i = const_points

    # Create heaviside function for ODE HBC
    heaviside_tensor = heaviside_fct(x_i,jump_loc)

    model = KANN(
        n_width=n_width,
        n_order=n_order,
        n_elements=n_elements,
        n_collocation = n_samples,
        n_samples=1,
        x_min=x_min,
        x_max=x_max,
        ndim_in=ndim_in,
        autodiff=autodiff,
    )
    
    # Containers to save the solutions for the animation and the final plot, respectively [adjustment for higher ouput dim necessary]
    solutions = np.empty((n_samples,1))
    with torch.no_grad():
        data = torch.zeros((n_samples,n_epochs+3))
        data[:,0] = x_i.view(-1)
        y_i_torch = torch.from_numpy(y_i)
        data[:,1] = y_i_torch.view(-1)
        dataGrad = torch.zeros((n_samples,n_epochs+2))
        dataGrad[:,0] = x_i.view(-1)
    k = 1000
    residual = 0 
    _ = 0
    sample = 0
    
    with tqdm.trange(n_epochs) as pbar1:
        for _ in pbar1:
            loss_epoch = torch.zeros((n_samples,))
            for sample in range(n_samples):    
                loss = 0  
                x = x_i[sample].unsqueeze(-1)
                const = const_i[sample].unsqueeze(-1)
                h = heaviside_tensor[sample].unsqueeze(-1)
                j = 1/(1 + torch.exp(-k*(x - 1.0)))
                if autodiff is True:
                    #model_input = torch.cat([x, const], dim=0).unsqueeze(0)
                    model_input = torch.cat([x, h], dim=0).unsqueeze(0)
                    #model_input = x
                    #print(model_input.shape)
                    #exit()
                    y = y0 + x*(model(model_input,_,sample))
                    dydx = torch.autograd.grad(
                        y, x, torch.ones_like(x), create_graph=True, materialize_grads=True
                    )[0]
                    residual = (dydx + y)
                else:
                    with torch.no_grad():
                        system = model.linear_system(x,_,sample,h,y0)
                        A_inner = system["A_inner"]
                        A_outer = system["A_outer"]
                        residual = system["b"]
                        y = system["y"]
                        dydx = system["dy"]

                # Fill data into tensors for saving
                data[sample,_+2] = y.view(-1)
                dataGrad[sample,_+1] = dydx.view(-1)
                
                loss = torch.mean(torch.square(residual))
                #loss2 = torch.norm(residual)
                #loss = residual
                
                if autodiff is True:
                    g_lst = torch.autograd.grad(
                        outputs=residual,
                        inputs=model.parameters(),
                    )
                    if ndim_in > 1:
                        g_lst0_hstack = torch.cat(torch.unbind(g_lst[0], dim=-1), dim=-1)
                        g_hstack = torch.hstack([g_lst0_hstack, g_lst[1]])
                        norm = torch.linalg.norm(g_hstack)**2
                    else:
                        norm = torch.linalg.norm(torch.hstack(g_lst))**2
                else:
                    g_lst = [A_inner, A_outer]

                    norm = (
                        torch.linalg.norm(A_inner) ** 2
                        + torch.linalg.norm(A_outer) ** 2
                    )
                
                # Kaczmarz update
                for p, g in zip(model.parameters(), g_lst):
                    update = (residual / norm) * torch.squeeze(g)
                    p.data -= update

                loss_epoch[sample] = loss

            loss_mean = torch.mean(loss_epoch)
            loss_str = f"{loss_mean.item():0.4e}"

            pbar1.set_postfix(loss=loss_str)

            Tickrate = pbar1.format_dict['rate']
            if _ % interval1 == 0:
                with torch.no_grad():
                    sampleeval = 0
                    vec = torch.zeros(n_samples)
                    for sampleeval in range(n_samples):
                        x = x_i[sampleeval].unsqueeze(-1)
                        const = const_i[sampleeval].unsqueeze(-1)
                        #model_input = torch.cat([x, const], dim=0).unsqueeze(0)
                        model_input = torch.cat([x, h], dim=0).unsqueeze(0)
                        vec[sampleeval] = y0+x*model(model_input,_,sampleeval)
                    vec = vec.detach().numpy().reshape(-1,1)
                    if _ == 0:
                        solutions = vec
                    else:
                        solutions = np.hstack([solutions, vec])
    

    print(f"\nTotal Elapsed Time: {pbar1.format_dict['elapsed']:.2f} seconds")
    # Create the Final Ouput of your Model
    y_hatvec = torch.zeros_like(x_i)
    dydy_hatvec = torch.zeros_like(x_i)
    for sample in range(n_samples):
        x = x_i[sample].unsqueeze(-1)
        h = heaviside_tensor[sample].unsqueeze(-1)
        const = const_i[sample].unsqueeze(-1)
        #model_input = torch.cat([x, const], dim=0).unsqueeze(0)
        model_input = torch.cat([x, h], dim=0).unsqueeze(0)
        y_hat = y0 + x * model(model_input,_,sample)
        dydy_hat= torch.autograd.grad(
                    y_hat, x, torch.ones_like(x), create_graph=True, materialize_grads=True
                )[0]
        y_hatvec[sample] = y_hat
        dydy_hatvec[sample] = dydy_hat
            
    y_hatvec = y_hatvec.detach()
    dydy_hatvec = dydy_hatvec.detach()
    data[:,n_epochs+2] = y_hatvec
    dataGrad[:,n_epochs+1] = dydy_hatvec
    y_hatvec = y_hatvec.view(-1,1).numpy()
    dydy_hatvec = dydy_hatvec.view(-1,1).numpy()
    l2 = np.linalg.norm(y_i - y_hatvec)
    print(f"L2-error: {l2.item():0.4e}")
    solutions = np.hstack([solutions, y_hatvec])
    
    create_animation(saveloc,save,show,solutions, x_i, y_i,n_width, n_order, n_samples,n_epochs,y0,spacing,x_max,interval1)
    error_abs = plot_solution(saveloc,save,show,x_i,y_hatvec, y_i, l2, n_width, n_order, n_samples,n_epochs,y0,spacing,x_max, loss_str)
    plt.close('all')
    print(timestamp, f"{loss_mean.item():.4e}",f"{l2.item():.4e}")
    
    data = data.detach().numpy()
    dataGrad = dataGrad.detach().numpy()
    if save: 
        np.savetxt(os.path.join(saveloc,f"data{n_width}o{n_order}s{n_samples}sp{spacing}e{n_epochs}y{y0}.csv"), data, delimiter=",",fmt='%1.3f')
        np.savetxt(os.path.join(saveloc,f"datagrad{n_width}o{n_order}s{n_samples}sp{spacing}e{n_epochs}y{y0}.csv"), dataGrad, delimiter=",",fmt='%1.3f')
        #np.savetxt(os.path.join(saveloc,f"data{n_width}o{n_order}s{n_samples}sp{spacing}e{n_epochs}y{y0}.csv"), solutions, delimiter=",")
        #np.savetxt(os.path.join(saveloc,f"dataABS{n_width}o{n_order}s{n_samples}sp{spacing}e{n_epochs}y{y0}.csv"), error_abs, delimiter=",")
    print("Data saved to csv file.")
    
    return None

if __name__ == "__main__":
    main()