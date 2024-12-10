"""Pytorch implementation of KANN."""

import matplotlib.pyplot as plt
import torch
import tqdm
import parameters
import pandas as pd

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

class LagrangeKANNinner(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(LagrangeKANNinner, self).__init__()
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
        
        self.phi_ikp_inner = torch.zeros((self.n_nodes+1, self.n_width, self.n_nodes))
        self.dphi_ikp_inner = torch.zeros((self.n_nodes+1, self.n_width, self.n_nodes))
        self.ddphi_ikp_inner = torch.zeros((self.n_nodes+1, self.n_width, self.n_nodes))

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

        # for n_order = 1 we do linear interpolation l0 and l1
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

class LagrangeKANNouter(torch.nn.Module):
    """A KANN layer using n-th order Lagrange polynomials."""

    def __init__(self, n_width, n_order, n_elements, n_samples, x_min, x_max):
        """Initialize."""
        super(LagrangeKANNouter, self).__init__()
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
        regression,
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
        self.regression = regression
        self.autodiff = autodiff
        self.speedup = speedup

        if speedup:
            self.inner = LagrangeKANNinner(n_width, n_order, n_elements, n_samples, x_min, x_max)
            self.outer = LagrangeKANNouter(n_width, n_order, n_elements, n_samples, x_min, x_max)
        else:
            self.inner = LagrangeKANN(n_width, n_order, n_elements, n_samples, x_min, x_max)
            self.outer = LagrangeKANN(n_width, n_order, n_elements, n_samples, x_min, x_max)
        
        total_params = sum(p.numel() for p in self.parameters())
        nodes_per_width = n_elements * n_order + 1
        print(f"\nTotal parameters: {total_params}")
        print(f"Order: {n_order}")
        print(f"Number of elements: {n_elements}")
        print(f"Nodes per width: {nodes_per_width}")
        print(f"Samples: {n_samples}")
        print(f"Regression: {regression}")
        print(f"Autodiff: {autodiff}\n")
        return None

    def forward(self, x, epoch, sample):
        """Forward pass for whole batch."""
        if self.speedup:
            x = self.inner(x,epoch,sample)["t_ik"]
        else:
            x = self.inner(x)["t_ik"]
        x = self.outer(x)["t_ik"]

        x = torch.einsum("ik -> i", x)

        return x

    def residual(self, x, y_true,epoch,sample):
        """Calculate residual."""
        if self.regression is True:
            y = self.forward(x,epoch,sample)
            residual = y - y_true

        else:
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
        if self.regression is True:
            if self.speedup:
                inner_dict = self.inner(x,_,sample)
                outer_dict = self.outer(inner_dict["t_ik"])
                
                #inner_phi_ikp = inner_dict["phi_ikp"]
                inner_phi_ikp = self.inner.phi_ikp_inner[sample:(sample+1), :, :]

                outer_dt_ik = outer_dict["dt_ik"]
                outer_t_ik = outer_dict["t_ik"]
                outer_phi_ikl = outer_dict["phi_ikp"]
                
                y = torch.einsum("ik -> i", outer_t_ik)
                b = y - y_true
                
                A_outer = torch.einsum("ikl -> ikl", outer_phi_ikl)

                A_inner = torch.einsum("ik, ikp -> ikp", outer_dt_ik, inner_phi_ikp)
            else:
                b = self.residual(x, y_true,_,sample)
                inner_dict = self.inner(x)
                outer_dict = self.outer(inner_dict["t_ik"])

                inner_phi_ikp = inner_dict["phi_ikp"]

                outer_dt_ik = outer_dict["dt_ik"]
                outer_phi_ikl = outer_dict["phi_ikp"]

                A_outer = torch.einsum("ikl -> ikl", outer_phi_ikl)

                A_inner = torch.einsum("ik, ikp -> ikp", outer_dt_ik, inner_phi_ikp)
                

        else:
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

def diff(x, y, order):
    """Return derivative dx/dy."""
    dxdy = torch.autograd.grad(
        x,
        y,
        torch.ones_like(x),
        create_graph=True,
    )[0]
    """
    for i in range(1, order):
        dxdy = torch.autograd.grad(
            dxdy,
            y,
            torch.ones_like(x),
            create_graph=True,
        )[0]
    """
    return dxdy

def disc_osc(x):
    """Use regression."""
    if x < 0.0:
        y = 5.0
        for k in range(1, 5):
            y += torch.sin(k * x)
    else:
        y = torch.cos(10.0 * x)

    return y

def main():
    """Execute main routine."""
    n_width = parameters.n_width
    n_order = parameters.n_order
    n_samples = parameters.n_samples
    n_epochs = parameters.n_epochs
    tol = parameters.tol
    autodiff = parameters.autodiff
    regression = parameters.regression
    runs = parameters.runs
    speedup = parameters.speedup
    """
    n_width = 5
    n_order = 1
    n_samples = 20
    n_elements = int((n_samples - 2) / n_order)
    n_epochs = 4000
    tol = 1e-30
    autodiff = True
    regression = True
    """
    values = torch.zeros((runs, 9))

    for run in range(runs):
        print(f"\nrun at iteration {run+1}")
        same_loss_counter = 0
        previous_loss = 0

        n_elements = int((n_samples - 2) / n_order)
        if regression is True:
            x_min = -1.0
        else:
            x_min = 0.0
        x_max = 1.0

        # vector of n_samples from x_min to x_max
        x_i = torch.linspace(x_min, x_max, n_samples).requires_grad_(True)

        # create sample data
        if regression is True:
            # y - vec with n_samples entries
            y_i = torch.zeros_like(x_i)
            for sample in range(n_samples):
                y_i[sample] = disc_osc(x_i[sample])
        else:
            y_i = torch.exp(x_i)

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
            regression=regression,
            autodiff=autodiff,
            speedup=speedup
        )

        with tqdm.trange(n_epochs) as pbar1:
            for _ in pbar1:
                lr_epoch = torch.zeros((n_samples,))
                loss_epoch = torch.zeros((n_samples,))
                # start looping over each training sample (50 times (0-49))
                for sample in range(n_samples):

                    x = x_i[sample].unsqueeze(-1)

                    if regression is True:
                        if autodiff is True:
                            y = model(x,_,sample)
                            residual = y - y_i[sample].unsqueeze(-1)
                        else:  # manual differentiation
                            system = model.linear_system(x, y_i[sample],_,sample)
                            A_inner, A_outer, residual = system["A_inner"], system["A_outer"], system["b"]
                    else:
                        if autodiff is True:
                            y = (1 + x * model(x,_,sample))
                            dydx = torch.autograd.grad(
                                y, x, torch.ones_like(x), create_graph=True
                            )[0]
                            residual = dydx - y
                        else:# manual differentiation
                            system = model.linear_system(x, y_i[sample],_,sample)
                            A_inner = system["A_inner"]
                            A_outer = system["A_outer"]
                            residual = system["b"]

                    loss = torch.mean(torch.square(residual))

                    if autodiff is True:
                        g_lst = torch.autograd.grad(
                            outputs=residual,
                            inputs=model.parameters(),
                        )
                        norm = torch.linalg.norm(torch.hstack(g_lst)) ** 2
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

                    # lr_epoch[sample] = lr
                    loss_epoch[sample] = loss

                loss_mean = torch.mean(loss_epoch)
                loss_str = f"{loss_mean.item():0.4e}"

                pbar1.set_postfix(loss=loss_str)
                if loss_mean.item() >= previous_loss:
                    same_loss_counter += 1
                else:
                    same_loss_counter = 0

                previous_loss = loss_mean.item()
                if same_loss_counter >= 40:
                    values[run, 6] = pbar1.format_dict["elapsed"]
                    break
                if loss_mean.item() < tol:
                    values[run, 6] = pbar1.format_dict["elapsed"]
                    break
                Tickrate = pbar1.format_dict['rate']
            
        print(f"\nTotal Elapsed Time: {pbar1.format_dict['elapsed']:.2f} seconds")
        if same_loss_counter > 20:
            print(f"Same loss counter: {same_loss_counter}")

        # calculate final result of the model and the plot
        y_hat = torch.zeros_like(x_i)
        for sample in range(n_samples):
            x = x_i[sample].unsqueeze(-1)

            if regression is True:
                y_hat[sample] = model(x,_,sample)

            else:
                y_hat[sample] = 1 + x * model(x,_,sample)
        
        l2 = torch.linalg.norm(y_i - y_hat)
        print(f"L2-error: {l2.item():0.4e}")
        # how many samples, width, order, tol, l2-error,  which epoch, runtinme
        values[run, 0] = n_samples
        values[run, 1] = n_width
        values[run, 2] = n_order
        values[run, 3] = tol
        values[run, 4] = loss_mean
        values[run, 5] = l2
        values[run, 6] = _
        values[run, 7] = pbar1.format_dict["elapsed"]
        values[run, 8] = Tickrate
        #n_samples = n_samples + 5

    # print(values[:,0])
    # print(values[:,1])
    # print(values[:,2])
    # print(values[:,3])

    plt.figure(0)
    plt.plot(y_hat.detach().numpy(), label="K(x)", c="red", linestyle="-")
    plt.plot(y_i.detach().numpy(), label="f(x)", c="black", linestyle="--")
    plt.title(f"L2-error: {l2.detach().numpy():0.4e}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ode.pdf")

    np_array = values.detach().numpy()
    df = pd.DataFrame(np_array)
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
    print(f"Values saved to excel file: {exc_file}")

    return None

if __name__ == "__main__":
    main()
