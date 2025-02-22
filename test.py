import torch
import numpy as np
import matplotlib.pyplot as plt
x_min = 0 
x_max = 5
n_samples = 90
def collocationpoints(total_values):
    nval1 = total_values // 3
    nval2 = total_values - nval1
    log_values = torch.logspace(0, torch.log10(torch.tensor(5.0)), steps=nval2, base=10)
    log_values = torch.linspace(1, 5, steps=nval2)
    # Second example: Logarithmic spacing between 1 and 0
    log_values2 = torch.logspace(0, -2, steps=nval1, base=10)
    log_values2 = 1 - log_values2  # Flip to go from 1 to 0
    combined = torch.cat((log_values2, log_values))
    combined = combined.detach().numpy()
    return combined

col_points = np.linspace(x_min, x_max, n_samples)
col2 = collocationpoints(n_samples)
plt.figure(figsize=(8, 3))
plt.scatter(col_points, np.zeros_like(col_points), color='red')
plt.scatter(col2, np.zeros_like(col2)+0.01, color='blue')
plt.scatter(np.logspace(0, np.log10(5), num=11, base=10), np.zeros(11)+0.02, color='green')
plt.grid()
plt.show()