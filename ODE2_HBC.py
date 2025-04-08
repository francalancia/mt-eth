import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        return 1 if x >= 1.0 else 0
    
    def system_heaviside(y,x):
        H = heaviside(x)
        dydx = (H - y)
        return dydx
    
    y_heaviside = odeint(system_heaviside, y0, x)
    
    return y_heaviside
def f(x):
    if x < 0.1:
        return np.exp(-10 * x)
    else:
        return 1 + (np.exp(-1) - 1) * np.exp(-10 * (x - 0.1))

def plot_solution(save,show,pinn, col_points, f_x_exact, i):
    with torch.no_grad():
        f_x = pinn(col_points).detach()
    col_points = col_points.detach()
    f_x_exact = torch.from_numpy(f_x_exact).view(-1,1)
    error_abs = np.abs(f_x_exact - f_x)
    ####################################################################################################################
    # Plotting the analytical and PINN solution
    ####################################################################################################################
    
    plt.figure(1,figsize=(10, 5))
    ax = plt.gca()
    plt.rcParams.update({
        'font.size': 14,              # Base font size
        'axes.labelsize': 12,         # Axis labels
        'xtick.labelsize': 12,        # X-axis tick labels
        'ytick.labelsize': 12,        # Y-axis tick labels
        'legend.fontsize': 12,        # Legend
        'figure.titlesize': 12        # Figure title
    })
    ax.plot(
        col_points[:, 0],
        f_x_exact[:, 0],
        label="Analytical solution",
        color="black",
        alpha=1.0,
        linewidth=2,
    )
    ax.plot(
        col_points[:, 0],
        f_x[:, 0], 
        linestyle = "--" ,
        label="PINN solution",
        color="tab:green",
        linewidth=2)
    ax.axvline(x=1.0, color='gray', linestyle='--', label='Jump at x = 1.0')
    l2 = torch.linalg.norm(f_x_exact - f_x)
    ax.set_title(f"L2 error: {l2:.4e}")
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_ylim(0.35, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)
    
    axins = ax.inset_axes([0.68, 0.35, 0.3, 0.5])
    axins.plot(col_points[:, 0], f_x_exact[:, 0], color="black", linewidth=2)
    axins.plot(col_points[:, 0], f_x[:, 0], color="tab:green", linestyle="--", linewidth=2)
    axins.axvline(x=1.0, color='gray', linestyle='--')
    axins.set_xlim(0.9, 1.1)
    axins.set_ylim(0.3625, 0.3875)
    axins.set_xticks([0.9, 1.0, 1.1])
    axins.set_yticks([0.3625, 0.375, 0.3875])
    #axins.set_xticklabels([])
    #axins.set_yticklabels([])

    axins.grid(True)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.25")
    if save:
        plt.savefig(f"E:/ETH/Master/25HS_MA/Data_ODE2/ODE2_{timestamp}.png")
    if show:
        plt.show()
    plt.close()
    
    ####################################################################################################################
    # Plotting the absolute error between the analytical and PINN solution
    ####################################################################################################################
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)
    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_bm = fig.add_subplot(gs[1, 1])
    ax_br = fig.add_subplot(gs[1, 2])
    ax_top.plot(col_points[:,0], error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_top.axvline(x=1.0, color='black', linestyle='--', label= "Jump location")
    ax_top.legend()
    ax_top.set_title("Absolute error between Analytical and PINN")
    ax_top.set_xlabel("x")
    ax_top.set_xticks(np.arange(0, 10, 1))
    ax_top.set_xlim(0, 10)
    ax_top.set_ylabel("Absolute error")
    ax_top.grid()
    
    ax_bl.plot(col_points[:,0], error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_bl.set_xlabel("x")
    ax_bl.set_xlim(0, 0.95)
    ax_bl.set_ylabel("Absolute error")
    ax_bl.grid()
    
    ax_bm.plot(col_points[:,0], error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_bm.set_xlabel("x")
    ax_bm.set_xlim(0.85, 1.15)
    #ax_bm.set_ylabel("Absolute error")
    ax_bm.grid()
    
    ax_br.plot(col_points[:,0], error_abs, label="Absolute error between Analytical and PINN", color="red", alpha=1.0, linewidth=2)
    ax_br.set_xlabel("x")
    ax_br.set_xlim(1.05, 10)
    #ax_br.set_ylabel("Absolute error")
    ax_br.grid()
    plt.subplots_adjust(hspace=0.2)
    if save:
        plt.savefig(f"E:/ETH/Master/25HS_MA/Data_ODE2/ODE2_abs_{timestamp}.png")
    if show:
        plt.show()
    return l2
def create_animation(save,show,solutions, col_exact, f_x_exact, interval = 10):
    col_exact = col_exact.detach().numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(col_exact, f_x_exact, label="Analytical solution", color="black", alpha=1.0, linewidth=2)
    line, = ax.plot(col_exact, solutions[0], linestyle="--", label="PINN solution", color="tab:green", linewidth=2)
    ax.axvline(x=1.0, color='gray', linestyle='--', label='Jump at x = 1.0')
    solutions = np.array(solutions).squeeze()
    f_x_exact = f_x_exact.transpose()
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_ylim(0.35, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)
    i = 0
    def animate(i):
        line.set_ydata(solutions[i])
        epoch = i*interval
        ax.set_title(f"Epoch = {epoch}, L2 error = {np.linalg.norm(f_x_exact - solutions[i]):.4e}")
        return line, ax,

    ani = FuncAnimation(fig, animate, frames=len(solutions), interval=100, blit=False, repeat = False)
    if save: 
        ani.save(f'E:/ETH/Master/25HS_MA/Data_ODE2/PINN_animation_{timestamp}.mp4', writer='ffmpeg', fps=10, dpi = 300)
    if show:
        plt.show()
    
    
    return None

class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Sigmoid
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
        tensor = torch.where(x >= 1.0, torch.ones_like(x), torch.zeros_like(x))
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
    learning = False
    save = False
    show = True
    heaviside_bool = True
    # define neural network to train
    n_input = 1
    n_output = 1
    n_hidden = 32
    n_layers = 2
    n_epochs = 100
    k = 1000

    pinn = FCN(n_input, n_output, n_hidden, n_layers)
    tot_val_log = 701
    tot_val = 501
    # define collocation points
    col_points2 = collocationpoints(tot_val_log)
    col_points = np.linspace(0, 10, tot_val)
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
        if heaviside_bool:
            heavyside = heaviside(col_points)
        else:
            jump = 1/(1 + torch.exp(-k*(col_points - 1.0)))
    
    #solutions = []
    #optimiser = torch.optim.AdamW(pinn.parameters(), lr=1e-3, weight_decay=3e-2)
    #optimiser = torch.optim.AdamW(pinn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    
    optimiser = torch.optim.LBFGS(
    pinn.parameters(),
    lr=0.1,            
    max_iter=400, #prev 1000        
    history_size=150,    
    tolerance_grad=1e-14,
    tolerance_change=1e-16,
    line_search_fn='strong_wolfe'
    )
    
    if learning:
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
            loss = torch.mean((df_xdx -(jump - f_x)) ** 2)
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
                loss = torch.mean((df_xdx - 10*(-f_x+heavyside)) ** 2)
                # backpropagate loss, take optimiser step
                loss.backward()
                
                optimiser.step()
                
                pbar.set_postfix(loss=f"{loss.item():.4e}")
                """
                #if _ % 10 == 0:
                #    with torch.no_grad():
                #        f_x = pinn(col_points)
                #        solutions.append(f_x.detach().numpy())
                
                #optimiser.step()
                pbar.set_postfix(loss=f"{loss.item():.4e}")
    pinn.eval()
    with torch.no_grad():
        f_x = pinn(col_points)
        #solutions.append(f_x.detach().numpy())
    f_x_np = f_x.detach().numpy()
    
    l2_error = np.linalg.norm(f_x_exact - f_x_np)
    # Plotting the analytical and PINN solution
    fig = plt.figure(figsize=(10, 5))
    plt.plot(col_points.detach().numpy(), f_x_exact, label="Exact solution", color="black", alpha=1.0, linewidth=2)
    plt.plot(col_points.detach().numpy(), f_x_np, linestyle="--", label="PINN solution", color="tab:green", linewidth=2)
    plt.title(f"L2 error: {l2_error:.4e}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()
    
    x_np = col_points.detach().numpy()
    n_hidden_np = np.full(x_np.shape, n_hidden)
    n_layers_np = np.full(x_np.shape, n_layers)
    n_epochs_np = np.full(x_np.shape, n_epochs)
    k_np = np.full(x_np.shape, k)
    tot_val_np = np.full(x_np.shape, tot_val)
    
    npz_path = fr"E:\ETH\Master\25HS_MA\Final_Results_Report\PINN_ODE_JUMP{heaviside_bool}_l2{l2_error}.npz"
    csv_path = fr"E:\ETH\Master\25HS_MA\Final_Results_Report\PINN_ODE_JUMP{heaviside_bool}_l2{l2_error}.csv"
    
    if heaviside_bool:
        data_to_save = np.hstack([x_np, f_x_np, n_hidden_np, n_layers_np, n_epochs_np, tot_val_np])
        np.savez(npz_path, x=x_np, f_x=f_x_np, n_hidden=n_hidden_np, n_layers=n_layers_np, n_epochs=n_epochs_np, tot_val=tot_val_np)
        np.savetxt(csv_path, data_to_save, delimiter=",", header="x,fx,n_h,n_l,n_e,tot_val", comments="")
    else:
        data_to_save = np.hstack([x_np, f_x_np, n_hidden_np, n_layers_np, n_epochs_np, k_np, tot_val_np])
        np.savez(npz_path, x=x_np, f_x=f_x_np, n_hidden=n_hidden_np, n_layers=n_layers_np, n_epochs=n_epochs_np, k =k_np, tot_val=tot_val_np)
        np.savetxt(csv_path, data_to_save, delimiter=",", header="x,fx,n_h,n_l,n_e,k,tot_val", comments="")

    print(f"Saved NPZ to: {npz_path}")
    print(f"Saved CSV to: {csv_path}")

    #create_animation(save, show,solutions, col_points, f_x_exact)

    #l2 = plot_solution(save,show,pinn, col_points, f_x_exact, n_epochs)
    #print(timestamp, f"{loss.item():.4e}",f"{l2.item():.4e}")
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
    if False:
        with open(f"E:/ETH/Master/25HS_MA/Data_ODE2/hyperparam_{timestamp}.txt","w") as file:
            file.write(f"Hidden neurons: {n_hidden}")
            file.write(f"Hidden layers: {n_layers}")
            file.write(f"Collocation points: {tot_val}")
            file.write(f"Training loss: {loss.item():.4e}")
            file.write(f"L2 error: {l2}")
        
    return None


if __name__ == "__main__":
    main()
