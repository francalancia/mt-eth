import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from matplotlib.animation import FuncAnimation
torch.manual_seed(123)
torch.set_default_dtype(torch.float64)
plt.rcParams.update({
    'font.size': 16,              # Base font size
    'axes.labelsize': 14,         # Axis labels
    'xtick.labelsize': 14,        # X-axis tick labels
    'ytick.labelsize': 14,        # Y-axis tick labels
    'legend.fontsize': 14,        # Legend
    'figure.titlesize': 14        # Figure title
    })
def exact_solution(x):
    y = np.exp(x)
    return y


def plot_solution(pinn, col_points, col_exact, f_x_exact, i,save,show):
    f_x_exact = f_x_exact.detach().numpy()
    with torch.no_grad():
        f_x = pinn(col_exact).detach().numpy()
    error_abs = np.abs(f_x_exact - f_x)
    col_exact = col_exact.detach().numpy()
    col_points = col_points.detach().numpy()
    plt.figure(figsize=(10, 5))
    plt.plot(
        col_exact[:, 0],
        f_x_exact[:, 0],
        label="Analytical solution",
        color="black",
        alpha=1.0,
        linewidth=2,
    )
    l2 = np.linalg.norm(f_x_exact - f_x)
    plt.plot(col_exact[:, 0], f_x[:, 0], linestyle = "--" ,label="PINN solution", color="tab:green", linewidth=2)
    plt.title(f"Solution of $f(x) = e^x$, B.C. $f(0) = 1$, l2 = {l2:.4e}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(0,3)
    plt.legend(loc = "upper left")
    plt.grid()
    if save:
        plt.savefig(f'E:/ETH/Master/25HS_MA/Data_ODE1/ODE1.png', dpi = 600)
    if show:
        plt.show()
    plt.close()
    plt.figure(2,figsize=(10,5))
    plt.plot(col_points[:,0], error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    plt.title("Absolute error between Analytical and PINN")
    plt.xlabel("x")
    plt.ylabel("Absolute error")
    plt.grid()
    if save:
        plt.savefig(f'E:/ETH/Master/25HS_MA/Data_ODE1/ODE1_abs.png', dpi = 600)
    if show:
        plt.show()

    return None
def create_animation(save,show,solutions, col_exact, f_x_exact, interval = 10):
    col_exact = col_exact.detach()
    f_x_exact = f_x_exact.detach().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(col_exact.numpy(), f_x_exact, label="Analytical solution", color="black", alpha=1.0, linewidth=2)
    line, = ax.plot(col_exact.numpy(), solutions[0], linestyle="--", label="PINN solution", color="tab:green", linewidth=2)
    solutions = np.array(solutions).squeeze()
    f_x_exact = f_x_exact.transpose()
    #ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper left")
    ax.grid(True)
    i = 0
    def animate(i):
        line.set_ydata(solutions[i])  # update the data
        epoch = i*interval
        ax.set_title(f"Epoch = {epoch}, L2 error = {np.linalg.norm(f_x_exact - solutions[i]):.4e}")
        return line, ax,

    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=100, blit=False, repeat = False)  # Change the interval here
    if save:
        ani.save(f'E:/ETH/Master/25HS_MA/Data_ODE1/ODE1_animation.mp4', writer='ffmpeg', fps=10, dpi = 300)  # Specify fps and writer
    if show:
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

def main():
    save = True
    show = False
    # define neural network to train
    n_input = 1
    n_output = 1
    n_hidden = 24
    n_layers = 2
    n_epochs = 1000

    pinn = FCN(n_input, n_output, n_hidden, n_layers)

    # define collocation points
    col_points = torch.linspace(0, 1, 60).view(-1, 1).requires_grad_(True)

    # exact solution
    col_exact = torch.linspace(0, 1, 60).view(-1, 1)
    f_x_exact = exact_solution(col_exact)

    #optimiser = torch.optim.AdamW(pinn.parameters(), lr=1e-3)
    optimiser = torch.optim.LBFGS(
    pinn.parameters(),
    lr=1e-01,            
    max_iter=1000,        
    history_size=150,    
    tolerance_grad=1e-14,
    tolerance_change=1e-16,
    line_search_fn='strong_wolfe'
    )
    solutions = []
    def closure():
        # zero the gradients
        optimiser.zero_grad()
        # compute model output for the collocation points
        f_x =pinn(col_points)
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
            """
            #optimiser.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4e}")
            if _ % 10 == 0:
                with torch.no_grad():
                    f_x = pinn(col_points)
                    solutions.append(f_x.detach().numpy())
    pinn.eval()
    with torch.no_grad():
        f_x = pinn(col_points)
        solutions.append(f_x.detach().numpy())

    create_animation(save,show, solutions, col_points, f_x_exact)
    plot_solution(pinn, col_points, col_exact, f_x_exact, n_epochs,save,show)
    
    
    
    
    return None


if __name__ == "__main__":
    main()
