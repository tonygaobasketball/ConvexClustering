import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import seaborn as sns

sns.set()

# Directory to save models
MODEL_DIR = "experiments/models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Model hyperparameters
p = 15         # Dimension of data (e.g., 2, 3, 4, 8, 16, 32, ...)
N = 30          # Number of matrices (e.g., 64, 128, 256)
m = 60        # Number of edges for incidence matrix
hidden = 50    # Number of hidden units in the neural network
layers = 4     # Number of layers in the neural network
beta = 10      # Beta parameter for softplus activation

# Generate incidence matrix Dw
num_edges = m
num_vertices = N
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
gen_dt = 4000  # Total number of training data points
train_dt = torch.empty(gen_dt, p, N).uniform_(-1, 1)
mask = torch.rand(train_dt.shape) < 0.1
replacement_values = torch.where(torch.rand_like(train_dt[mask]) < 0.5,
                                 -torch.ones_like(train_dt[mask]),
                                 torch.ones_like(train_dt[mask]))
train_dt[mask] = replacement_values

# Placeholder for L21_data_gen_update function
# Replace this with your actual implementation from Funcs.py
def L21_data_gen_update(Dw, data):
    # Dummy implementation: returns data as is
    # Actual function should compute the L21 norm or similar based on Dw
    return data.clone()

true_f = L21_data_gen_update(Dw, train_dt)

# Create DataLoader for batching
bsize = 500
dataset = TensorDataset(train_dt, true_f)
dataloader = DataLoader(dataset, batch_size=bsize, shuffle=True)

# Define the LPN_tensor model
class LPN_tensor(nn.Module):
    def __init__(self, in_dim, hidden, layers, beta, p, N):
        super(LPN_tensor, self).__init__()
        self.p = p
        self.N = N
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.beta = beta

        # Define the network layers
        self.input_layer = nn.Linear(in_dim, hidden)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers - 1)])
        self.output_layer = nn.Linear(hidden, p * N)
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten to (batch_size, p * N)
        out = self.softplus(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.softplus(layer(out))
        out = self.output_layer(out)
        return out.view(batch_size, self.p, self.N)  # Reshape to (batch_size, p, N)

    def scalar(self, x):
        # Placeholder for scalar function if needed
        # For now, returns the forward output (modify as per your needs)
        return self.forward(x)

    def wclip(self):
        # Placeholder for weight clipping if needed
        pass

# Initialize the model
tr_dim = p * N
lpn_model = LPN_tensor(in_dim=tr_dim, hidden=hidden, layers=layers, beta=beta, p=p, N=N).to(device)
optimizer = optim.Adam(lpn_model.parameters(), lr=1e-4)
iter_thres = 1e-6

# Training function for a single iteration
def single_iteration(lpn_model, train_xx, true_f, i, bsize, optimizer, p, loss_type=2, gamma_loss=None):
    f_xx, xx = true_f.to(device), train_xx.to(device)
    f_out = lpn_model(xx)

    if loss_type == 2:
        loss = (f_out - f_xx).pow(2).mean()  # Mean Squared Error
    elif loss_type == 1:
        loss = (f_out - f_xx).abs().mean()  # Mean Absolute Error
    else:
        raise ValueError("loss_type must be 1 or 2")

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

# Save the trained model
torch.save(lpn_model.state_dict(), os.path.join(MODEL_DIR, "l2.pth"))

# Function to plot learned proximal operator vs true function
def plot_L21(plot_data, model):
    xi = plot_data['xx']  # Shape: (amount, p, N)
    true_y = plot_data['f_xx']  # Shape: (amount, p, N)
    amount_dt, p, N = xi.shape

    y = model(xi)  # Shape: (amount, p, N)
    c = model.scalar(xi)  # Shape depends on scalar implementation

    # Adjust for visualization (plot first 2 dimensions if p > 2)
    if p != 2:
        y_2dim = y[:, :2, :]
        true_y_2dim = true_y[:, :2, :]
        vis_y = y_2dim.permute(1, 0, 2).reshape(2, amount_dt * N)
        vis_true_y = true_y_2dim.permute(1, 0, 2).reshape(2, amount_dt * N)
    else:
        vis_y = y.permute(1, 0, 2).reshape(p, amount_dt * N)
        vis_true_y = true_y.permute(1, 0, 2).reshape(p, amount_dt * N)

    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot(vis_y[0, :].detach().numpy(), vis_y[1, :].detach().numpy(), '*', color='blue', label='LPN')
    plt.plot(vis_true_y[0, :].detach().numpy(), vis_true_y[1, :].detach().numpy(), 'o',
             markerfacecolor='none', markeredgecolor='red', markersize=8, label='True')
    plt.title("Learned Proximal Operator by LPN")
    plt.legend()
    plt.show()

    loss = (y - true_y).pow(2).sum() / (amount_dt * N * p)  # MSE
    print("MSE for the plot:", loss.detach().numpy())

# Generate visualization data for training set
vis_amount_data = 8
rand_id = torch.randint(0, gen_dt, (vis_amount_data,))
plt_data = {'xx': train_dt[rand_id], 'f_xx': true_f[rand_id]}
plot_L21(plt_data, lpn_model)

# Generate testing data
gen_test = 2000
test_dt = torch.empty(gen_test, p, N).uniform_(-1, 1)
mask = torch.rand(test_dt.shape) < 0.1
replacement_values = torch.where(torch.rand_like(test_dt[mask]) < 0.5,
                                 -torch.ones_like(test_dt[mask]),
                                 torch.ones_like(test_dt[mask]))
test_dt[mask] = replacement_values
test_true_f = L21_data_gen_update(Dw, test_dt)

# Visualize testing data
amount_test_dt = test_dt.shape[0]
vis_amount_test = 20
rand_id = torch.randint(0, amount_test_dt, (vis_amount_test,))
plt_test = {'xx': test_dt[rand_id], 'f_xx': test_true_f[rand_id]}
plot_L21(plt_test, lpn_model)

# Plot convergence of loss
plt.figure(figsize=(10, 6))
plt.plot(loss_records["mse"][500:], label="MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence of Losses")
plt.legend()
plt.grid(True)
plt.show()