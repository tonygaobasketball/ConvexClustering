"""1-D LPN"""
import torch
import torch.nn as nn


class LPN(nn.Module):
    def __init__(self, in_dim, hidden, layers=1, beta=1):
        super().__init__()

        self.hidden = hidden
        self.lin = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden, bias=False),
                *[nn.Linear(hidden, hidden, bias=False) for _ in range(layers)],
                nn.Linear(hidden, 1, bias=False),
            ]
        )

        self.res = nn.ModuleList(
            [*[nn.Linear(in_dim, hidden) for _ in range(layers)], nn.Linear(in_dim, 1)]
        )
        self.act = nn.Softplus(beta=beta)

    def scalar(self, x):
        y = x.clone()
        y = self.act(self.lin[0](y))
        for core, res in zip(self.lin[1:-1], self.res[:-1]):
            y = self.act(core(y) + res(x))

        y = self.lin[-1](y) + self.res[-1](x)
        return y

    def init_weights(self, mean, std):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            y = self.scalar(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]

        return grad
    
    
"""
p-dim input
"""
import torch
import torch.nn as nn


class LPN_vec(nn.Module):
    def __init__(self, in_dim, hidden, layers=1, beta=1):
        super().__init__()

        self.hidden = hidden
        self.lin = nn.ModuleList(
            [
                nn.Linear(in_dim, hidden, bias=False, dtype=torch.float64),
                *[nn.Linear(hidden, hidden, bias=False, dtype=torch.float64) for _ in range(layers)],
                nn.Linear(hidden, 1, bias=False, dtype=torch.float64),
            ]
        )

        self.res = nn.ModuleList(
            [*[nn.Linear(in_dim, hidden, dtype=torch.float64) for _ in range(layers)],
             nn.Linear(in_dim, 1, dtype=torch.float64)]
        )
        self.act = nn.Softplus(beta=beta)

    def scalar(self, x):
        y = x.clone().to(torch.float64)  # Ensure input tensor is of type torch.float64
        y = self.act(self.lin[0](y))
        for core, res in zip(self.lin[1:-1], self.res[:-1]):
            y = self.act(core(y) + res(x))

        y = self.lin[-1](y) + self.res[-1](x)
        return y

    def init_weights(self, mean, std):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x):
        x = x.to(torch.float64)  # Ensure input tensor is of type torch.float64
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            y = self.scalar(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]

        return grad
    
    
"""
p by n dim tensor input
"""
import torch
import torch.nn as nn

class LPN_tensor(nn.Module):
    """
    Learned Proximal Network (LPN) for tensor inputs.

    This network approximates a convex function and computes its gradient for given inputs.
    Designed to handle 3D tensors of shape (batch_size, p, n), where:
        - batch_size: Number of samples in the batch.
        - p: Dimensionality of each sample.
        - n: Number of features per dimension.

    Parameters:
    - in_dim: Input dimension (p * n for flattened input).
    - hidden: Number of hidden neurons in each layer.
    - layers: Number of hidden layers.
    - beta: Softplus activation parameter (controls smoothness of activation).
    """
    def __init__(self, in_dim, hidden, layers=1, beta=1):
        super().__init__()

        # Hidden dimension and activation
        self.hidden = hidden

        # Linear layers for the main feedforward path
        self.lin = nn.ModuleList(
            [
                # Input layer
                nn.Linear(in_dim, hidden, bias=False, dtype=torch.float64),
                # Hidden layers
                *[nn.Linear(hidden, hidden, bias=False, dtype=torch.float64) for _ in range(layers)],
                # Output layer
                nn.Linear(hidden, 1, bias=False, dtype=torch.float64),
            ]
        )

        # Residual layers (to add input contributions directly)
        self.res = nn.ModuleList(
            [*[nn.Linear(in_dim, hidden, dtype=torch.float64) for _ in range(layers)],
             nn.Linear(in_dim, 1, dtype=torch.float64)]
        )

        # Activation function (Softplus ensures smoothness and convexity properties)
        self.act = nn.Softplus(beta=beta)

    def scalar(self, x):
        """
        Compute the scalar output (convex function approximation) for the input tensor.

        Parameters:
        - x: Input tensor of shape (batch_size, p, n).

        Returns:
        - y: Scalar output tensor of shape (batch_size, 1).
        """
        # Ensure the input is in float64 for precision
        y = x.clone().to(torch.float64)  # Clone the input to avoid modifying the original tensor
        x = x.to(torch.float64)
        # Flatten the tensor: (batch_size, p, n) -> (batch_size, p * n)
        batch_size, p, n = y.shape
        y = y.view(batch_size, p * n)

        # Apply the first linear layer with activation
        y = self.act(self.lin[0](y))

        # Process through hidden layers with residual connections
        for core, res in zip(self.lin[1:-1], self.res[:-1]):
            # core(y): Transform the hidden state
            # res(x): Transform the original input
            # Add residual connection to maintain convexity
            y = self.act(core(y) + res(x.view(batch_size, p * n)))

        # Apply the final output layer with residual
        y = self.lin[-1](y) + self.res[-1](x.view(batch_size, p * n))
        return y

    def init_weights(self, mean, std):
        """
        Initialize the weights of the hidden layers with a normal distribution.

        Parameters:
        - mean: Mean of the normal distribution.
        - std: Standard deviation of the normal distribution.
        """
        with torch.no_grad():
            for core in self.lin[1:]:
                # Initialize weights with exp(N(mean, std)) for convexity
                core.weight.data.normal_(mean, std).exp_()

    def wclip(self):
        """
        Clip the weights of the hidden layers to ensure non-negativity.
        This maintains the convexity property of the network.
        """
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x):
        """
        Compute the gradient of the scalar output with respect to the input.
        """
        # Ensure input tensor is of type torch.float64
        x = x.to(torch.float64)
        
        # Enable gradient tracking
        if not x.requires_grad:
            x.requires_grad_(True)
    
        # Compute the scalar output
        y = self.scalar(x)
    
        # Compute gradients
        
        grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]
        return grad
    
        # grad = []
        # for i in range(x.shape[0]):
        #     grad_i = torch.autograd.grad(
        #         outputs=y[i].sum(), 
        #         inputs=x[i],
        #         retain_graph=True,
        #         create_graph=True,
        #         allow_unused = True,  # Raise an error if x[i] is not used
        #     )[0]
        #     if grad_i is None:
        #             print(f"Gradient is None for index {i}")

        #     grad.append(grad_i)
    
        # return torch.stack(grad)
        
# # Initialize the LPN model
# lpn_model = LPN_tensor(p * n, hidden, layers=layers, beta=beta)
