import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.animation import FuncAnimation
torch.manual_seed(123)
torch.set_default_dtype(torch.float64)

def exact_solution(x):
    y = np.exp(x)
    return y


def plot_solution(pinn, col_points, col_exact, f_x_exact, i):
    f_x = pinn(col_exact).detach()
    plt.rcParams.update({
    'font.size': 16,              # Base font size
    'axes.labelsize': 14,         # Axis labels
    'xtick.labelsize': 14,        # X-axis tick labels
    'ytick.labelsize': 14,        # Y-axis tick labels
    'legend.fontsize': 14,        # Legend
    'figure.titlesize': 14        # Figure title
    })
    plt.figure(figsize=(10, 5))
    plt.plot(
        col_exact[:, 0],
        f_x_exact[:, 0],
        label="Analytical solution",
        color="black",
        alpha=1.0,
        linewidth=2,
    )
    l2 = torch.linalg.norm(f_x_exact - f_x)
    plt.plot(col_exact[:, 0], f_x[:, 0], linestyle = "--" ,label="PINN solution", color="tab:green", linewidth=2)
    plt.title(f"Solution of $f(x) = e^x$, B.C. $f(0) = 1$")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(0,3)
    plt.legend(loc = "upper left")
    plt.grid()
    plt.savefig("ODE_analytical.png")
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
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

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
def create_animation(solutions, col_exact, f_x_exact, interval = 5):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(col_exact.numpy(), f_x_exact, label="Analytical solution", color="black", alpha=1.0, linewidth=2)
    line, = ax.plot(col_exact.numpy(), solutions[0], linestyle="--", label="PINN solution", color="tab:green", linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left")
    ax.grid(True)
    
    def animate(i):
        line.set_ydata(solutions[i])  # update the data
        epoch = i*interval
        ax.set_title(f"Solution of $f(x) = e^x$, B.C. $f(0) = 1$, epoch = {epoch}")
        return line, ax,

    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=100, blit=False)  # Change the interval here
    ani.save('PINN_training_animation.mp4', writer='ffmpeg', fps=10, dpi = 300)  # Specify fps and writer
    plt.show()
    return None
def main():
    # define neural network to train
    n_input = 1
    n_output = 1
    n_hidden = 32
    n_layers = 4
    n_epochs = 1000

    pinn = FCN(n_input, n_output, n_hidden, n_layers)

    # define collocation points
    col_points = torch.linspace(0, 1, 400).view(-1, 1).requires_grad_(True)

    # exact solution
    col_exact = torch.linspace(0, 1, 400).view(-1, 1)
    f_x_exact = exact_solution(col_exact)

    optimiser = torch.optim.AdamW(pinn.parameters(), lr=1e-3)
    solutions = []
    with tqdm.trange(n_epochs) as pbar:
        for _ in pbar:
            optimiser.zero_grad()
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
            
            pbar.set_postfix(loss=f"{loss.item():.4e}")
            if _ % 5 == 0:
                with torch.no_grad():
                    solutions.append(f_x.detach().numpy())

    create_animation(solutions, col_exact, f_x_exact)
    #plot_solution(pinn, col_points, col_exact, f_x_exact, n_epochs)
    return None


if __name__ == "__main__":
    main()
