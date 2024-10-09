import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x):
    y = np.exp(x)
    return y


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
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)

        # we calculate the networks output for x = 0 and then use that to subract it from the real networks output
        x0 = torch.zeros_like(x)
        x0 = self.fcs(x0)
        x0 = self.fch(x0)
        x0 = self.fce(x0)

        return 1 + (x * (x - x0))

    torch.manual_seed(123)


# define neural network to train
pinn = FCN(1, 1, 32, 2)

# define points for the physics loss
t_physics = torch.linspace(0, 2, 20).view(-1, 1).requires_grad_(True)

# exact solution
t_test = torch.linspace(0, 2, 200).view(-1, 1)
f_x_exact = exact_solution(t_test)


optimiser = torch.optim.Adam(pinn.parameters(), lr=7e-4)
for i in range(5001):
    optimiser.zero_grad()
    # compute loss
    f_x = pinn(t_physics)
    df_xdx = torch.autograd.grad(
        f_x, t_physics, torch.ones_like(f_x), create_graph=True
    )[
        0
    ]  # (30, 1)
    loss = torch.mean((df_xdx - f_x) ** 2)

    # backpropagate loss, take optimiser step
    loss.backward()
    optimiser.step()
    # plot the result as training progresses
    if i % 100 == 0:
        print(loss)
        if i % 5000 == 0:
            # print(u.abs().mean().item(), dudt.abs().mean().item(), d2udt2.abs().mean().item())
            f_x = pinn(t_test).detach()
            plt.figure(figsize=(6, 2.5))
            plt.scatter(
                t_physics.detach()[:, 0],
                torch.zeros_like(t_physics)[:, 0],
                s=20,
                lw=0,
                color="tab:green",
                alpha=0.6,
            )
            # plt.scatter(t_boundary.detach()[:,0], torch.zeros_like(t_boundary)[:,0], s=20, lw=0, color="tab:red", alpha=0.6)
            plt.plot(
                t_test[:, 0],
                f_x_exact[:, 0],
                label="Exact solution",
                color="tab:grey",
                alpha=0.6,
            )
            plt.plot(t_test[:, 0], f_x[:, 0], label="PINN solution", color="tab:green")
            plt.title(f"Training step {i}")
            plt.legend()
            plt.show()
