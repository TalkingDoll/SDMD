import time
import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as la
from numpy import arange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from functorch import vmap, jacrev

from torch.func import jacrev
import joblib
from sde_coefficients_estimator import SDECoefficientEstimator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

# device = 'cuda'
# device = 'cpu'

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class KoopmanNNTorch(nn.Module):
    def __init__(self, input_size, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super(KoopmanNNTorch, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_psi_train = n_psi_train

        self.layers = nn.ModuleList()
        bias = False
        n_layers = len(layer_sizes)

        # First layer
        self.layers.append(nn.Linear(input_size, layer_sizes[0], bias=bias))
        # Hidden layers
        for ii in arange(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[ii], layer_sizes[ii+1], bias=True))
        # Activation and output layer
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(layer_sizes[-1], n_psi_train, bias=True))

    def forward(self, x):
        # 1) If input is a 1D vector, add batch dimension
        squeeze_back = False
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Convert to (1, D)
            squeeze_back = True

        # 2) Save original input
        in_x = x

        # 3) Normal forward pass
        for layer in self.layers:
            x = layer(x)

        # 4) Concatenate constant term, original input and network output
        const_out = torch.ones_like(in_x[:, :1])
        out = torch.cat([const_out, in_x, x], dim=1)

        # 5) If batch dimension was added at the beginning, remove it
        if squeeze_back:
            out = out.squeeze(0)  # Restore to original 1D

        return out
    


class KoopmanModelTorch(nn.Module):
    def __init__(self, dict_net, target_dim, k_dim):
        super(KoopmanModelTorch, self).__init__()
        self.dict_net = dict_net
        self.target_dim = target_dim
        self.k_dim = k_dim
        self.layer_K = nn.Linear(k_dim, k_dim, bias=False)
        self.layer_K.weight.requires_grad = False
    
    def forward(self, input_x, input_y):
        psi_x = self.dict_net.forward(input_x)
        psi_y = self.dict_net.forward(input_y)
        psi_next = self.layer_K(psi_x)
        outputs = psi_next - psi_y
        return outputs


class MLPModel(nn.Module):
    def __init__(self, num_features, num_outs, n_hid=128, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(dropout),
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid // 2),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 2, num_outs)
        )

        # Use Kaiming initialization
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
        
class KoopmanSolverTorch(object):
    '''
    Galerkin generator EDMD (gEDMD) solver.
    '''
    def __init__(self, dic, target_dim, reg=0.0, checkpoint_file='example_koopman_net001.torch',
                 fnn_checkpoint_file='example_fnn001.torch', a_b_file=None,
                 generator_batch_size=4, fnn_batch_size=32, delta_t=0.1,
                 patience=7, min_delta=1e-4):
        self.dic = dic  # dictionary class (PsiNN)
        self.dic_func = dic.forward
        self.target_dim = target_dim
        self.reg = reg
        self.psi_x = None
        self.dPsi_X = None
        self.checkpoint_file = checkpoint_file
        self.fnn_checkpoint_file = fnn_checkpoint_file
        self.generator_batch_size = generator_batch_size
        self.fnn_batch_size = fnn_batch_size
        self.delta_t = delta_t
        self.a_b_file = a_b_file
        self.patience = patience
        self.min_delta = min_delta

    def separate_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return data_x, data_y

    def build(self, data_train):
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)
        

    def eig_decomp(self):
        """
        Perform spectral decomposition of the generator L, save eigenvalues and eigenvectors.
        """
        L_np = self.L_Psi.detach().cpu().numpy()
        evalues, evectors = np.linalg.eig(L_np)
        self.eigenvalues = evalues
        self.eigenvectors = evectors
        return evalues, evectors

    def eigenfunctions(self, X):
        """
        Compute eigenfunctions: Psi(X) @ eigenvectors.
        Args:
            X: numpy array, shape (N, D)
        Returns:
            numpy array, shape (N, n_eigen)
        """
        X_tensor = torch.DoubleTensor(X).to(self.L_Psi.device)
        Psi_X = self.dic(X_tensor).detach().cpu().numpy()
        # Ensure spectral decomposition is performed first
        if not hasattr(self, 'eigenvectors'):
            self.eig_decomp()
        return Psi_X @ self.eigenvectors

    def get_Psi_X(self):
        """
        Return Psi_X on the training set.
        """
        data_x = torch.DoubleTensor(self.data_x_train).to(self.L_Psi.device)
        return self.dic(data_x).detach().cpu().numpy()

    def get_Psi_Y(self):
        """
        Return Psi_Y on the training set.
        """
        data_y = torch.DoubleTensor(self.data_train[1]).to(self.L_Psi.device)
        return self.dic(data_y).detach().cpu().numpy()

    def compute_neural_a_b(self, data_x, delta_t):
        """
        Compute the drift and diffusion coefficients using the SDECoefficientEstimator.
        """
        num_samples, state_dim = data_x.shape
        X_t_1 = data_x[:-1, :].to(device)
        X_t = data_x[1:, :].to(device)
        
        # Initialize the SDE coefficient estimator
        sde_estimator = SDECoefficientEstimator(device=device)
        
        # Build the model with customizable parameters
        hidden_size = 128
        n_hidden_layers = 1
        dropout = 0.01
        
        sde_estimator.build_model(
            state_dim=state_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            n_hidden_layers=n_hidden_layers
        )
        
        # Train the model
        learning_rate = 5e-4
        epochs = 50
        batch_size = self.fnn_batch_size
        
        sde_estimator.fit_model(
            X_t_1=X_t_1,
            X_t=X_t,
            checkpoint_file=self.fnn_checkpoint_file,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        # Estimate the coefficients
        b_Xt, a_Xt = sde_estimator.estimate_coefficients2(X_t_1, X_t, delta_t)
        
        # Ensure a_Xt is a 3D tensor even in 1D state space
        if state_dim == 1 and len(a_Xt.shape) == 2:
            # If 1D state space and a_Xt is a 2D tensor (M-1, 1)
            # Expand to a 3D tensor (M-1, 1, 1)
            a_Xt_final = a_Xt.unsqueeze(-1)
            print(f"Expanded a_Xt shape from {a_Xt.shape} to {a_Xt_final.shape}")
        else:
            a_Xt_final = a_Xt
            print(f"Using original a_Xt shape: {a_Xt_final.shape}")
        
        return b_Xt, a_Xt_final

    def process_batch(self, batch_inputs):
        """
        batch_inputs: Tensor of shape (B, D)
        Returns:
            J: Tensor of shape (B, F, D)
            H: Tensor of shape (B, F, D, D)
        """
        # 1) Compute the gradient function for a single sample x: x -> (F, D)
        jac_fn = jacrev(self.dic)
        # 2) Compute the Hessian function for a single sample x: x -> (F, D, D)
        hess_fn = jacrev(jac_fn)

        # 3) Vectorize over the batch dimension using vmap
        J = vmap(jac_fn)(batch_inputs)   # (B, F, D)
        H = vmap(hess_fn)(batch_inputs)  # (B, F, D, D)

        return J, H



    def compute_dPsi_X(self, data_x, b_Xt, a_Xt, delta_t):
        """
        Compute dPsi_X using batched derivatives, avoiding full Jacobian/Hessian storage.
        
        Args:
            data_x (Tensor): shape (M, D)
            b_Xt (Tensor): shape (M-1, D)
            a_Xt (Tensor): shape (M-1, D, D)
            delta_t (float): time step
            
        Returns:
            dPsi_X (Tensor): shape (M-1, F)
        """
        device = data_x.device
        num_samples = data_x.shape[0]
        num_features = self.dic(data_x[:1]).shape[1]  # F
        batch_size = 64
        num_batches = (num_samples + batch_size - 1) // batch_size

        dPsi_X = torch.zeros(num_samples - 1, num_features, device=device, dtype=data_x.dtype)
        batch_offset = 0

        for i, (batch_J, batch_H) in enumerate(self.get_derivatives(data_x, batch_size)):
            batch_size_actual = batch_J.shape[0]
            end_idx = min(batch_offset + batch_size_actual, num_samples - 1)

            batch_b = b_Xt[batch_offset:end_idx]
            batch_a = a_Xt[batch_offset:end_idx]

            term1 = torch.einsum('mfd,md->mf', batch_J[:end_idx - batch_offset], batch_b)
            term2 = 0.5 * torch.einsum('mfkl,mkl->mf', batch_H[:end_idx - batch_offset], batch_a)
            dPsi_X[batch_offset:end_idx] = term1 + term2

            batch_offset += batch_size_actual

        self.dPsi_X = dPsi_X
        return dPsi_X

    def get_derivatives(self, inputs, batch_size=64):
        """
        Yield batch-wise Jacobian and Hessian to avoid storing full tensors.

        Args:
            inputs (Tensor): shape (M, D)
            batch_size (int): size of each batch

        Yields:
            batch_J (Tensor): shape (batch_size, F, D)
            batch_H (Tensor): shape (batch_size, F, D, D)
        """
        with torch.no_grad():
            num_samples = inputs.shape[0]
            num_batches = (num_samples + batch_size - 1) // batch_size
            jac_fn = jacrev(self.dic)
            hess_fn = jacrev(jac_fn)

            for i in tqdm(range(num_batches), desc='Processing batches', unit='batch'):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_samples)
                batch_inputs = inputs[start:end]
                batch_J = vmap(jac_fn)(batch_inputs)  # (batch_size, F, D)
                batch_H = vmap(hess_fn)(batch_inputs)  # (batch_size, F, D, D)
                yield batch_J, batch_H
                del batch_J, batch_H
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()




    def compute_generator_L(self, data_x, b_Xt, a_Xt, delta_t, lambda_reg=0.1):
        """
        Compute the generator matrix L via
          L = (PsiX^T PsiX + λI)^{-1} (PsiX^T dPsi_X)
        using a Cholesky solve and caching PsiX, instead of full pinv.
        """
        # 1) Move to GPU once
        data_x = data_x.to(device)

        # 2) Compute dPsi_X with your vectorized routine
        dPsi_X = self.compute_dPsi_X(data_x, b_Xt, a_Xt, delta_t)
        self.dPsi_X = dPsi_X

        # 3) Evaluate dictionary on all but last sample, store PsiX
        psi_x = self.dic(data_x[:-1])           # shape (M-1, F)
        
        # 4) Form Gram matrix G = PsiX^T PsiX  (F×F)
        G = psi_x.T @ psi_x
        
        # 5) Regularize
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        G_reg = G + lambda_reg * I
        
        # 6) Compute RHS A = PsiX^T @ dPsi_X   (F×F)
        A = psi_x.T @ dPsi_X

        # 7) Solve G_reg · L_Psi = A via Cholesky (SPD solve)
        #    This is much faster and more stable than pinv:
        #    G_reg = L L^T
        L = torch.linalg.cholesky(G_reg)        # lower-triangular L
        L_Psi = torch.cholesky_solve(A, L)      # solves L L^T X = A

        # 8) Cache and return
        self.L_Psi = L_Psi
        return L_Psi

    def train_gEDMD(
        self,
        data_train,
        data_valid,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        lambda_reg=0.1,
        log_interval=10,
        lr_decay_factor=0.5
    ):
        """
        Main gEDMD training process, returns L and its spectrum.
        """
        self.data_train = data_train
        self.data_x_train, _ = self.separate_data(self.data_train)
        data_x_train_tensor = torch.DoubleTensor(self.data_x_train).to(device)

        # SDE coefficient estimation
        self.b_Xt, self.a_Xt = self.compute_neural_a_b(data_x_train_tensor, delta_t=self.delta_t)

        optimizer = torch.optim.AdamW(self.dic.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_decay_factor, patience=5, verbose=True
        )
        early_stopping = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        criterion = nn.MSELoss()
        best_loss = 1e15
        losses = []

        for epoch in range(epochs):
            self.dic.train()
            optimizer.zero_grad()

            # 1. Compute Psi_X and dPsi_X
            data_x = data_x_train_tensor
            dPsi_X = self.compute_dPsi_X(data_x, self.b_Xt, self.a_Xt, self.delta_t)
            Psi_X = self.dic(data_x[:-1])  # (M-1, F)

            # 2. Least squares solution for L
            G = Psi_X.T @ Psi_X
            I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
            G_reg = G + lambda_reg * I
            A = Psi_X.T @ dPsi_X
            L = torch.linalg.solve(G_reg, A)  # (F, F)

            # 3. loss = ||dPsi_X - L Psi_X||^2
            dPsi_X_pred = (Psi_X @ L)  # (M-1, F)
            loss = criterion(dPsi_X_pred, dPsi_X)
            loss.backward()
            optimizer.step()

            # 4. Logging and early stopping
            losses.append(loss.item())
            scheduler.step(loss.item())
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, gEDMD loss: {loss.item():.6e}")

            if loss.item() < best_loss:
                print(f"Saving best model at epoch {epoch+1}")
                torch.save(self.dic.state_dict(), self.checkpoint_file)
                best_loss = loss.item()

            early_stopping(loss.item())
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # After training, load the best PsiNN parameters
        self.dic.load_state_dict(torch.load(self.checkpoint_file))
        self.L_Psi = L.detach()
        return losses, best_loss

    def build_with_generator(
        self,
        data_train,
        data_valid,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        lambda_reg=0.1,
        log_interval=10,
        lr_decay_factor=0.5
    ):
        """
        Compatible with old interface, directly calls the main gEDMD process.
        """
        return self.train_gEDMD(
            data_train=data_train,
            data_valid=data_valid,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_reg=lambda_reg,
            log_interval=log_interval,
            lr_decay_factor=lr_decay_factor
        )