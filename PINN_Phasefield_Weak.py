import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import datetime
from NN_gitlab import NeuralNet, init_xavier
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.manual_seed(432)
torch.set_default_dtype(torch.float32)

class ScaledReLU(nn.Module):
    def __init__(self, m=1.0):
        super(ScaledReLU, self).__init__()
        self.m = m  

    def forward(self, x):
        return torch.relu(self.m * x)

class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch using the custom activation"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, m=1.0):
        super().__init__()
        # Use a lambda to ensure each layer gets its own activation instance (with the same fixed m)
        activation = lambda: ScaledReLU(m)
        #activation = lambda: nn.ReLU()
        #activation = lambda: nn.Tanh()
        # Input layer: Linear transformation followed by activation
        self.fcs = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        )

        # Hidden layers: Each hidden layer applies a linear transformation and then the activation
        self.fch = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(N_HIDDEN, N_HIDDEN),
                    activation()
                )
                for _ in range(N_LAYERS - 1)
            ]
        )

        # Output layer: A final linear transformation
        self.fce = nn.Sequential(
                nn.Linear(N_HIDDEN, N_OUTPUT),
                #activation()
        )           
        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of parameters: {total_params}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain = 1.0)
            #nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def piecewise_alpha(alpha, beta=1e-3):
    alpha = alpha.clone().requires_grad_()
    result = torch.empty_like(alpha)

    # Masks
    mask_neg = alpha < -2
    mask_pos = alpha > 2
    mask_mid = (~mask_neg) & (~mask_pos)

    # Assign pieces
    result[mask_neg] = beta * (alpha[mask_neg] + 2)
    result[mask_pos] = beta * (alpha[mask_pos] - 2) + 1
    result[mask_mid] = alpha[mask_mid] / 4 + 0.5

    return result


def main():
    # define neural network to train
    n_input = 1
    n_output = 2
    n_hidden = 50
    n_layers = 4
    n_epochs = 3000
    n_samples = 50
    x = torch.linspace(-0.5, 0.5, n_samples).reshape(-1, 1).requires_grad_()
    #x = torch.linspace(0, 1.0, n_samples).reshape(-1, 1).requires_grad_()
    U_p = 0.6
    
    for i in range(1):
        #pinn = FCN(n_input, n_output, n_hidden, n_layers)
        pinn = NeuralNet(n_input, n_output, n_layers, n_hidden, 'SteepReLU', init_coeff=1.0)
        init_xavier(pinn)
        total_params = sum(p.numel() for p in pinn.parameters())
        print(f"Total number of parameters: {total_params}")
        #optimiser = torch.optim.AdamW(pinn.parameters(), lr=1e-3, weight_decay=3e-2)
        #optimiser = torch.optim.AdamW(pinn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        """
        optimiser = torch.optim.LBFGS(
        pinn.parameters(),
        lr=float(0.5),          
        max_iter=1000, #prev 1000 
        max_eval=10000, #prev 1000       
        history_size=25,    
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
        #line_search_fn='strong_wolfe'
        )
        """
        
        """
        optimiser = torch.optim.LBFGS(
            pinn.parameters(), lr=float(1e-4), max_iter=20000, max_eval=20000000, history_size=250,
            #line_search_fn="strong_wolfe",
            tolerance_change=1.0*np.finfo(float).eps, tolerance_grad=1.0*np.finfo(float).eps
        )
        """
        #optimiser = torch.optim.LBFGS(pinn.parameters(), lr = 1e-4)
        optimiser = torch.optim.Rprop(pinn.parameters(), lr=1e-8, step_sizes=(1e-10, 50))

        l = 0.05
        cw = 8.0/3.0
        weight_decay = 1e-5
        tol_ir = 5e-3
        penalty = (27/(64*tol_ir**2))
        
        def closure():
            # zero the gradients
            optimiser.zero_grad()
            
            # compute model output for the collocation points
            output = pinn(x)
            u_hat = output[:, 0].reshape(-1, 1)
            alpha_hat = output[:, 1].reshape(-1, 1)
            #alpha = piecewise_alpha(alpha_hat)
            
            # enforce hard boundary conditions
            u = (((x + 0.5) * (x - 0.5) * u_hat) + (x + 0.5)) * U_p
            alpha = (x + 0.5) * (x - 0.5) * alpha_hat
            #u = ((x* (x - 1.0) * u_hat) + (x)) * (U_p/1.0)
            #alpha = x*(x - 1.0) * alpha_hat
            
            # compute the derivatives
            dudx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            #dudx_org = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            dalphadx = torch.autograd.grad(alpha.sum(), x,  create_graph=True)[0]
            
            energy_elastic = 0.5 * ((1.0 - alpha) ** 2) * (dudx ** 2) 

            energy_damage = (1/cw) * (alpha + (l**2) * ((dalphadx) ** 2)) 

            dAlpha = alpha - torch.zeros_like(alpha)
            hist_penalty = nn.ReLU()(-dAlpha) 
            E_hist_penalty = 0.5 * 1 * penalty * (hist_penalty**2) 
            
            energy_tot = torch.sum(energy_elastic) + torch.sum(energy_damage) + torch.sum(E_hist_penalty)

            loss_energy = torch.log10(energy_tot)
            # Weight regularization: L2 penalty over all parameters
            
            l2_reg = 0.0
            for param in pinn.parameters():
                l2_reg += torch.sum(param ** 2)
                
            loss = loss_energy + weight_decay * l2_reg
            
            loss.backward()
            # return the loss for the optimiser
            return loss
        
        with tqdm.trange(n_epochs) as pbar:
            for _ in pbar:
                loss = optimiser.step(closure)
                pbar.set_postfix(loss=f"{loss.item():.4e}")
                    
        pinn.eval()
        output = pinn(x)
        u_sol = output[:, 0].reshape(-1, 1)
        alpha_sol = output[:, 1].reshape(-1, 1)
        u = (((x + 0.5) * (x - 0.5) * u_sol) + (x + 0.5)) * U_p
        alpha = (x + 0.5) * (x - 0.5) * alpha_sol
        #u = ((x* (x - 1.0) * u_sol) + (x)) * (U_p/1.0)
        #alpha = x*(x-1.0) * alpha_sol
        # compute the derivatives
        dudx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        #dudx_org = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dalphadx = torch.autograd.grad(alpha.sum(), x,  create_graph=True)[0]
        
        energy_elastic = 0.5 * ((1.0 - alpha) ** 2) * (dudx ** 2) 

        energy_damage = (1/cw) * (alpha + (l**2) * ((dalphadx) ** 2))
        
        e_el = torch.mean(energy_elastic)
        e_dam = torch.mean(energy_damage)
            

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        axs[0].plot(x.detach().numpy(), u.detach().numpy(), label="u")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("u")
        axs[0].set_title("Displacement")
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].grid()
        
        axs[1].plot(x.detach().numpy(), alpha.detach().numpy(), label="alpha")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("alpha")
        axs[1].set_title("Phase Field")
        axs[1].grid()
        axs[1].set_ylim(-0.1, 1.1)
        
        axs[2].plot(x.detach().numpy(), energy_elastic.detach().numpy(), label="elsastic energy")
        axs[2].plot(x.detach().numpy(), energy_damage.detach().numpy(), label="damage energy")
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("energy")
        axs[2].set_title("Energy")
        axs[2].grid()
        axs[2].set_ylim(-0.1, 1.1)
        axs[2].legend()
        print(f"Elastic Energy: {e_el.item()}")
        print(f"Damage Energy: {e_dam.item()}")
        plt.tight_layout()
        plt.show()
        
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        alpha_np = alpha.detach().cpu().numpy()
        e_el_np = energy_elastic.detach().cpu().numpy()
        e_dam_np = energy_damage.detach().cpu().numpy()

        # Use f-strings to incorporate 'my_var' into the filenames
        npz_path = fr"E:\ETH\Master\25HS_MA\Data_Phasefield\outputphasefieldweak_UP{U_p}.npz"
        csv_path = fr"E:\ETH\Master\25HS_MA\Data_Phasefield\outputphasefieldweak_UP{U_p}.csv"

        # Save to NPZ
        np.savez(npz_path, x=x_np, u=u_np, alpha=alpha_np, e_el=e_el_np, e_dam=e_dam_np)

        # Optionally also save as CSV
        data_to_save = np.hstack([x_np, u_np, alpha_np, e_el_np, e_dam_np])
        np.savetxt(csv_path, data_to_save, delimiter=",", header="x,u,alpha", comments="")

        print(f"Saved NPZ to: {npz_path}")
        print(f"Saved CSV to: {csv_path}")
        U_p = U_p + 0.01
    return None


if __name__ == "__main__":
    main()
