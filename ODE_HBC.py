import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm

torch.manual_seed(123)
torch.set_default_dtype(torch.float64)

def exact_solution(x):
    y = np.exp(x)
    return y


def plot_solution(pinn, col_points, col_exact, f_x_exact, i):
    f_x = pinn(col_exact).detach()
    plt.figure(figsize=(8, 4))
    plt.scatter(
        col_points.detach()[:, 0],
        torch.zeros_like(col_points)[:, 0],
        s=20,
        lw=0,
        color="tab:green",
        alpha=0.6,
    )
    plt.plot(
        col_exact[:, 0],
        f_x_exact[:, 0],
        label="Exact solution",
        color="tab:gray",
        alpha=0.6,
    )
    plt.plot(col_exact[:, 0], f_x[:, 0], label="PINN solution", color="tab:green")
    plt.title(f"Training step {i}")
    plt.legend()
    plt.show()

    return None


class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[nn.Linear(N_INPUT, N_HIDDEN), activation()])
        # creates identical hidden layers N_layers-1, fully connected with activation function
        self.fch = nn.Sequential(
            *[
                nn.Sequential(*[nn.Linear(N_HIDDEN, N_HIDDEN), activation()])
                for _ in range(N_LAYERS - 1)
            ]
        )
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        coord = x
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)

        """
        # we calculate the networks output for x = 0 and then use that to subract it from the real networks output
        x0 = torch.zeros_like(x)
        x0 = self.fcs(x0)
        x0 = self.fch(x0)
        x0 = self.fce(x0)
        """
        return 1 + (coord * (x))


def main():
    # define neural network to train
    n_input = 1
    n_output = 1
    n_hidden = 32
    n_layers = 2
    n_epochs = 5000

    pinn = FCN(n_input, n_output, n_hidden, n_layers)

    # define collocation points
    col_points = torch.linspace(0, 1, 20).view(-1, 1).requires_grad_(True)

    # exact solution
    col_exact = torch.linspace(0, 1, 200).view(-1, 1)
    f_x_exact = exact_solution(col_exact)

    #optimiser = torch.optim.AdamW(pinn.parameters(), lr=1e-3)
    optimiser = torch.optim.LBFGS(pinn.parameters(), lr=1e-2)

    def closure():
        # zero the gradients
        optimiser.zero_grad()
        # compute model output for the collocation points
        f_x = pinn(col_points)
        # compute the grad of the output w.r.t. to the collocation points
        df_xdx = torch.autograd.grad(
            f_x, col_points, torch.ones_like(f_x), create_graph=True
        )[0]
        # compute the loss mean squared error
        loss = torch.mean((df_xdx - f_x) ** 2)
        # backpropagate the loss
        loss.backward()
        # return the loss for the optimiser
        return loss
    
    with tqdm.trange(n_epochs) as pbar:
        for _ in pbar:
            loss = optimiser.step(closure)
            """
            # optimiser.zero_grad()
            # compute loss%
            f_x = pinn(col_points)
            df_xdx = torch.autograd.grad(
                f_x, col_points, torch.ones_like(f_x), create_graph=True
            )[
                0
            ]  # (30, 1)
            loss = torch.mean((df_xdx - f_x) ** 2)

            # backpropagate loss, take optimiser step
            loss.backward()
            optimiser.step()
            """
            pbar.set_postfix(loss=f"{loss.item():.4e}")

    plot_solution(pinn, col_points, col_exact, f_x_exact, n_epochs)
    return None


if __name__ == "__main__":
    main()
