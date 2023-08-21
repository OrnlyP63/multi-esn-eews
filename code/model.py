import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy import sparse

torch.manual_seed(42)
torch.cuda.manual_seed(42)

class ESNLayer(nn.Module):
    # Initialize the ESN layer.
    # n_internal_units: the number of recurrent units in the ESN
    # connectivity: the sparsity of the internal weights of the ESN (0 < connectivity <= 1)
    # noise_level: the amount of noise to add to the internal state at each time step
    # n_drop: the number of initial time steps to drop from the state matrix (used for training)

    def __init__(self, n_internal_units = 10, connectivity = 0.3, noise_level = 0.01, n_drop = 0):
        super(ESNLayer, self).__init__()
         # store hyperparameters
        self.n_internal_units = n_internal_units
        self.noise_level = noise_level
        self.n_drop = n_drop
        
        # initialize internal weights using the given connectivity and spectral radius
        internal_weights = self.init_internal_weight(n_internal_units, connectivity, circle=False)
        self.internal_weights = nn.Parameter(internal_weights)

        # initialize input weights randomly
        _array = torch.empty(self.n_internal_units, 3).uniform_(0, 1)
        input_weights = (2.0 * torch.bernoulli(_array) - 1.0) * 0.2
        self._input_weights = nn.Parameter(input_weights)
    
    # Initialize the internal weights of the ESN.
    # circle: if True, the internal weights will be initialized as a circle matrix.
    #         if False, the internal weights will be initialized as a random sparse matrix with the given connectivity.
    def init_internal_weight(self, n_internal_units, connectivity, circle=True, spectral_radius = 0.99):
        if circle:
            I = torch.eye(n_internal_units - 1)
            border1 = torch.zeros(n_internal_units - 1)
            border2 = torch.zeros(n_internal_units, 1)
            border2[0] = 1
            internal_weights = torch.vstack([border1, I])
            internal_weights = torch.hstack([internal_weights, border2])
            internal_weights *= spectral_radius
            
            return internal_weights
        
        else:
            internal_weights = sparse.rand(n_internal_units, n_internal_units, density=connectivity).todense()
            internal_weights = torch.FloatTensor(internal_weights)
            internal_weights[np.where(internal_weights > 0)] -= 0.5
            E, _ = torch.linalg.eig(internal_weights)
            e_max = torch.max(torch.abs(E))
            internal_weights /= torch.abs(e_max) / spectral_radius
            
            return internal_weights
            
    def forward(self, X):
        N, T, V = X.shape # Get the shape of the input tensor
        # Determine if the model is being run on GPU or CPU
        device = 'cuda' if self._input_weights.is_cuda else 'cpu'

        # Initialize the matrix that will hold the states of the internal units
        state_matrix = torch.empty((N, T - self.n_drop, self.n_internal_units)).to(device)
        # Initialize the state of the internal units to small random values
        previous_state = torch.ones((N, self.n_internal_units)).to(device) / 100
        # Calculate the input weights multiplied by the input data
        inputs = torch.einsum('bij,kj->bki', X, self._input_weights)

        # Loop through the input data at each time step
        for t in range(T):
            # Calculate the state of the internal units before applying the tanh activation function
            state_before_tanh = previous_state @ self.internal_weights + inputs[:, :, t]
            # Add noise to the state before the tanh activation
            state_before_tanh += torch.rand(N, self.n_internal_units).to(device) * self.noise_level
            # Calculate the state of the internal units by applying the tanh activation function
            previous_state = torch.tanh(state_before_tanh)
            # If the current time step is after the specified number of time steps to drop, add the current state to the state matrix
            if (t > self.n_drop - 1):
                state_matrix[:, t - self.n_drop, :] = previous_state

        return state_matrix
    
class RidgeEmbedingLayer(nn.Module):
    def __init__(self, alpha = 0, fit_intercept = True,):
        super(RidgeEmbedingLayer, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
    
    def forward(self, red_states):
        W = []
        n = 5
        device = 'cuda' if red_states.is_cuda else 'cpu'

        # Add ones to the input matrix if fit_intercept is True
        if self.fit_intercept:
            ones = torch.ones(red_states.shape[0], red_states.shape[1], 1).to(device)
            red_states = torch.cat([ones, red_states], dim = 2)

        # Calculate ridge regularization term
        ridge = self.alpha * torch.eye(red_states.shape[-1]).to(device) 

        
        # Calculate left hand side and right hand side for least square solution
        X = red_states[:, 0:-n, :]
        y = red_states[:, n:, :]
        lhs = torch.einsum('bik,bij->bkj',X, X)
        rhs = torch.einsum('bik,bij->bkj',X, y)

        # Calculate least square solution
        res = torch.linalg.lstsq(lhs + ridge, rhs).solution
        # Flatten solution into 1D tensor
        W = torch.flatten(res, start_dim=1)

        return W
    
class RCModel(nn.Module):
    def __init__(self, n=50):
        super().__init__()
        self.n = n
        self.drop = 50
        self.Time = 500
        self.L1 = ESNLayer(n_internal_units = self.n, connectivity = 0.3, noise_level = 0.5, n_drop = self.drop)
        self.L2 = RidgeEmbedingLayer(alpha = 10, fit_intercept=True)
        self.lin_output = nn.Linear((self.n + 1) ** 2, 2, bias=False)

    def forward_to_hidden(self, X):
        return self.L2(self.L1(X))
    
    def forward(self, X): 
        return self.lin_output(self.forward_to_hidden(X))
    
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(3, 32, 3, 2)
        self.bn1 = nn.BatchNorm1d(32, momentum=0.997)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, 3, 2)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.997)

        self.conv3 = nn.Conv1d(64, 128, 3, 2)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.997)

        self.conv4 = nn.Conv1d(128, 256, 3, 2)
        self.bn4 = nn.BatchNorm1d(256, momentum=0.997)

        self.linear1 = nn.Linear(7936, 64)
        self.linear2 = nn.Linear(64, 2)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(x.shape[0], -1)
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        return x