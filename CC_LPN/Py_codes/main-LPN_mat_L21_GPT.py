import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import seaborn as sns
from network import LPN_tensor  # Assuming LPN_tensor is defined in network.py
import Funcs  # Assuming Funcs.py contains L21_data_gen_update and other functions

sns.set()

MODEL_DIR = "experiments/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set seed
np.random.seed(1)
torch.manual_seed(1)

# Model parameters
p = 10  # Dimension of data [2,4,8,16,32....] 50 -> 20
N = 20 # Number of matrices [64,128,256] 500 -> 200
m = 40  # For incidence matrix, adjust as needed 1000 -> 400
hidden = 50  # Number of hidden units
layers = 4  # Number of layers
beta = 10  # Beta for softplus

# Generate incidence matrix Dw
num_edges = m  # Example value, adjust as needed
num_vertices = N  # Example value, adjust as needed
incidence_matrix = np.zeros((num_edges, num_vertices), dtype=int)
for i in range(num_edges):
    a, b = np.random.choice(num_vertices, size=2, replace=False)
    if a < b:
        incidence_matrix[i, a] = 1
        incidence_matrix[i, b] = -1
    else:
        incidence_matrix[i, b] = 1
        incidence_matrix[i, a] = -1
Dw = incidence_matrix

# Generate training data
gen_dt = 4000  # Total number of points
train_dt = torch.empty(gen_dt, p, N).uniform_(-1, 1)
mask = torch.rand(train_dt.shape) < 0.1
replacement_values = torch.where(torch.rand_like(train_dt[mask]) < 0.5,
                                 -torch.ones_like(train_dt[mask]),
                                 torch.ones_like(train_dt[mask]))
train_dt[mask] = replacement_values
true_f = Funcs.L21_data_gen_update(Dw, train_dt)  # Assuming this function is defined

# Create DataLoader
bsize = 500
dataset = TensorDataset(train_dt, true_f)
dataloader = DataLoader(dataset, batch_size=bsize, shuffle=True)

# Define the LPN model
tr_dim = p * N
lpn_model = LPN_tensor(in_dim=tr_dim, hidden=hidden, layers=layers, beta=beta).to(device)
optimizer = optim.Adam(lpn_model.parameters(), lr=1e-4)
iter_thres = 1e-6

# Training function
def single_iteration(lpn_model, train_xx, true_f, i, bsize, optimizer, p, loss_type=2, gamma_loss=None):
    f_xx, xx = true_f.to(device), train_xx.to(device)
    f_out = lpn_model(xx)
    cvx_out = lpn_model.scalar(xx)

    if loss_type == 2:
        loss = (f_out - f_xx).pow(2).mean()  # MSE over all elements in batch
    elif loss_type == 1:
        loss = (f_out - f_xx).abs().mean()  # MAE over batch
    else:
        raise ValueError("loss_type must be -1, 0, 1, or 2")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print(f"Epoch {i}, mse: {loss.item()}")

    lpn_model.wclip()
    return loss, None

# Training loop
loss_records = {"mse": []}
for epoch in range(20000):
    for batch_xx, batch_f_xx in dataloader:
        batch_xx = batch_xx.to(device)
        batch_f_xx = batch_f_xx.to(device)
        loss_i, _ = single_iteration(lpn_model, batch_xx, batch_f_xx, epoch, bsize, optimizer, p)
        loss_records["mse"].append(loss_i.item())
        if loss_i <= iter_thres:
            print(f"Converged at epoch {epoch}")
            break
    else:
        continue
    break

# Save the model
torch.save(lpn_model.state_dict(), os.path.join(MODEL_DIR, "l2.pth"))

# Function to plot learned prox, convex function, original function
def plot_L21(plot_data, model):
    xi = plot_data['xx']  # dimension amount by p by N
    true_y = plot_data['f_xx']  # dimension amount by p by N
    amount_dt, p, N = xi.shape

    y = model(xi)  # dimension amount by p by N
    c = model.scalar(xi)  # dimension amount by N

    if p != 2:
        # raise ValueError("Dimension not equal to 2")
        # return 0
        y_2dim = y[:, :2, :]
        true_y_2dim = true_y[:, :2, :]
        vis_y = y_2dim.permute(1, 0, 2).reshape(2, amount_dt * N)
        vis_true_y = true_y_2dim.permute(1, 0, 2).reshape(2, amount_dt * N)
    if p == 2:
        vis_y = y.permute(1, 0, 2).reshape(p, amount_dt * N)
        vis_true_y = true_y.permute(1, 0, 2).reshape(p, amount_dt * N)


    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(vis_y[0, :].detach().numpy(), vis_y[1, :].detach().numpy(), '*', color='blue', label='LPN')
    plt.plot(vis_true_y[0, :].detach().numpy(), vis_true_y[1, :].detach().numpy(), 'o', markerfacecolor='none', markeredgecolor='red', markersize=8, label='ADMM (true)')
    plt.title("Learned prox by LPN")
    plt.legend()
    plt.show()
    loss = (y - true_y).pow(2).sum() / (amount_dt * N * p)  # MSE loss
    print("mse for the plot:", loss.detach().numpy())

# Generate plot data
vis_amount_data = 8
rand_id = torch.randint(0, gen_dt, (vis_amount_data,))
plt_data = {'xx': train_dt[rand_id], 'f_xx': true_f[rand_id]}
plot_L21(plt_data, lpn_model)

#Create testing data
gen_test = 2000  # Total number of points
test_dt = torch.empty(gen_test, p, N).uniform_(-1, 1)
mask = torch.rand(test_dt.shape) < 0.1
replacement_values = torch.where(torch.rand_like(test_dt[mask]) < 0.5,
                                 -torch.ones_like(test_dt[mask]),
                                 torch.ones_like(test_dt[mask]))
test_dt[mask] = replacement_values
test_true_f = Funcs.L21_data_gen_update(Dw, test_dt)
# Create testing data
# gen_test_dt = 2000
# K_test = int(gen_test_dt ** (1 / (p * N)))
# test_dt, test_true_f = Funcs.L21_data_gen(Dw, p, N, K_test, -1, 1)
amount_test_dt = test_dt.shape[0]
vis_amount_test = 20
rand_id = torch.randint(0, amount_test_dt, (vis_amount_test,))
plt_test = {'xx': test_dt[rand_id], 'f_xx': test_true_f[rand_id]}
plot_L21(plt_test, lpn_model)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(loss_records["mse"][500:], label="mse")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence of Losses")
plt.legend()
plt.grid(True)
plt.show()