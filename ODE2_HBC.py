import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.integrate import odeint
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

def exact_solution(y0,x):
    
    def heaviside(x):
        return 1 if x >= 0.1 else 0
    
    def system_heaviside(y,x):
        H = heaviside(x)
        dydx = 10*(H - y)
        return dydx
    
    y_heaviside = odeint(system_heaviside, y0, x)
    
    return y_heaviside
def f(x):
    if x < 0.1:
        return np.exp(-10 * x)
    else:
        return 1 + (np.exp(-1) - 1) * np.exp(-10 * (x - 0.1))

def plot_solution(pinn, col_points, f_x_exact, i):
    with torch.no_grad():
        f_x = pinn(col_points).detach()
    col_points = col_points.detach()
    f_x_exact = torch.from_numpy(f_x_exact).view(-1,1)
    plt.figure(1,figsize=(10, 5))
    error_abs = np.abs(f_x_exact - f_x)

    """
    plt.scatter(
        col_points.detach()[:, 0],
        torch.zeros_like(col_points)[:, 0],
        s=20,
        lw=0,
        color="tab:green",
        alpha=1.0,
    )
    """
    plt.plot(
        col_points[:, 0],
        f_x_exact[:, 0],
        label="Analytical solution",
        color="black",
        alpha=1.0,
        linewidth=2,
    )
    plt.plot(
        col_points[:, 0],
        f_x[:, 0], 
        linestyle = "--" ,
        label="PINN solution",
        color="tab:green",
        linewidth=2)
    plt.axvline(x=0.1, color='gray', linestyle='--')
    l2 = torch.linalg.norm(f_x_exact - f_x)
    plt.title(f"L2 error: {l2:.4e}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.savefig("ODE2_lin2.png")
    plt.figure(2,figsize=(10,5))
    plt.plot(col_points[:,0], error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    plt.title("Absolute error between Analytical and PINN")
    plt.xlabel("x")
    plt.ylabel("Absolute error")
    plt.grid()
    plt.savefig("ODE2_abs2.png")
    plt.show()
    
    
    return None
def create_animation(pinn,solutions, col_exact, f_x_exact, interval = 100):
    col_exact = col_exact.detach()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(col_exact.numpy(), f_x_exact, label="Analytical solution", color="black", alpha=1.0, linewidth=2)
    ax.axvline(x=0.1, color='r', linestyle='--', label='x = 0.1')
    line, = ax.plot(col_exact.numpy(), solutions[0], linestyle="--", label="PINN solution", color="tab:green", linewidth=2)
    solutions = np.array(solutions).squeeze()
    f_x_exact = f_x_exact.transpose()
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    #ax.legend(loc="upper left")
    ax.grid(True)
    i = 0
    def animate(i):
        line.set_ydata(solutions[i])  # update the data
        epoch = i*interval
        ax.set_title(f"Epoch = {epoch}, L2 error = {np.linalg.norm(f_x_exact - solutions[i]):.4e}")
        return line, ax,

    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=100, blit=False, repeat = False)  # Change the interval here
    ani.save('PINN_training_animation.mp4', writer='ffmpeg', fps=100, dpi = 300)  # Specify fps and writer
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
        self.apply(self._init_weights)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
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
        return 1 + coord*x

def heaviside(x):
        tensor = torch.where(x >= 0.1, torch.ones_like(x), torch.zeros_like(x))
        return tensor
def collocationpoints(total_values):
    nval1 = total_values // 5
    nval2 = total_values - nval1
    log_values = torch.logspace(0, torch.log10(torch.tensor(1.0)), steps=nval2, base=10)

    # Second example: Logarithmic spacing between 1 and 0
    log_values2 = torch.logspace(0, -0.1, steps=nval1, base=10)
    log_values2 = 1 - log_values2  # Flip to go from 1 to 0
    combined = torch.cat((log_values2, log_values))
    combined = combined.detach().numpy()
    return combined
def main():
    learning = True
    # define neural network to train
    n_input = 1
    n_output = 1
    n_hidden = 32
    n_layers = 4
    n_epochs = 50000
    k = 1000

    pinn = FCN(n_input, n_output, n_hidden, n_layers)
    tot_val_log = 1000
    tot_val = 2001
    # define collocation points
    col_points2 = collocationpoints(tot_val_log)
    col_points = np.linspace(0, 1, tot_val)
    # exact solution
    y0 = 1
    f_x_exact = exact_solution(y0, col_points)
    #f_x_exact = exact_solution(y0, col_points2)
    #col_points = col_points2
    #plt.figure(figsize=(10, 5))
    #plt.plot(col_points, f_x_exact, label="Exact solution", color="red", alpha=1.0, linewidth=2)
    if False:
        col_exact = torch.linspace(0,5,tot_val)
        col_exact2 = collocationpoints(tot_val_log)
        f_x_exact = exact_solution(y0,col_exact)
        f_x_exact2 = exact_solution(y0,col_exact2)
        col_exact = col_exact.view(-1,1)
        col_exact2 = col_exact2.view(-1,1)
        f_x_exact = torch.from_numpy(f_x_exact)
        f_x_exact2 = torch.from_numpy(f_x_exact2)
    if False:
        plt.figure(figsize=(8, 4))
        plt.plot(col_exact, f_x_exact, label="Exact solution", color="blue", alpha=1.0, linewidth=2)
        plt.scatter(col_exact, torch.zeros_like(col_exact)+0.1, label="Initial condition", color="blue", alpha=1.0, linewidth=2)
        plt.plot(col_exact2, f_x_exact2, label="Exact solution", color="red", alpha=1.0, linewidth=2)
        plt.scatter(col_exact2, torch.zeros_like(col_exact2)-0.1, label="Initial condition", color="red", alpha=1.0, linewidth=2)
        plt.grid()
        plt.show()
    col_points = torch.from_numpy(col_points).view(-1,1).requires_grad_(True)
    with torch.no_grad():
        heavyside = heaviside(col_points)
        jump = 1/(1 + torch.exp(-k*(col_points - 0.1)))
    
    solutions = []
    optimiser = torch.optim.AdamW(pinn.parameters(), lr=1e-3, weight_decay=3e-2)
    #optimiser = torch.optim.LBFGS(pinn.parameters(), lr=1e-2)
    if learning:
        def closure():
            # zero the gradients
            optimiser.zero_grad()
            # compute model output for the collocation points
            f_x = 1+col_points*pinn(col_points)
            # compute the grad of the output w.r.t. to the collocation points
            df_xdx = torch.autograd.grad(
                f_x, col_points, torch.ones_like(f_x), create_graph=True
            )[0]
            # compute the loss mean squared error
            loss = torch.mean((df_xdx - heavyside + f_x) ** 2)
            # backpropagate the loss
            loss.backward()
            # return the loss for the optimiser
            return loss
        with tqdm.trange(n_epochs) as pbar:
            for _ in pbar:
                #loss = optimiser.step(closure)
                optimiser.zero_grad()
                # compute loss%
                f_x = pinn(col_points)
                df_xdx = torch.autograd.grad(
                    f_x, col_points, torch.ones_like(f_x), create_graph=True
                )[
                    0
                ]  # (30, 1)
                loss = torch.mean((df_xdx - 10*(-f_x+heavyside)) ** 2)

                # backpropagate loss, take optimiser step
                loss.backward()
                optimiser.step()
                
                pbar.set_postfix(loss=f"{loss.item():.4e}")
                if _ % 100 == 0:
                    with torch.no_grad():
                        solutions.append(f_x.detach().numpy())
    pinn.eval()
    with torch.no_grad():
        f_x = pinn(col_points)
        solutions.append(f_x.detach().numpy())

    
    create_animation(pinn, solutions, col_points, f_x_exact)

    plot_solution(pinn, col_points, f_x_exact, n_epochs)
    if False:
        x_values = np.linspace(0, 1, 4001)
        y_values = np.vectorize(f)(x_values)
        plt.plot(x_values, y_values, label='f(x)', color='blue')
        plt.axvline(x=0.1, color='gray',label= 'step', linestyle='--')
        plt.title(r'Solution of $ \frac{df}{dx} = 10*(H(x-0.1)-f(x))$, B.C. $f(0) = 1$')  
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()
        plt.savefig('ODE2_HBCanalytical.png')
        plt.show()
    return None


if __name__ == "__main__":
    main()
