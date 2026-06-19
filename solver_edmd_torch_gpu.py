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
# from sde_coefficients_estimator import SDECoefficientEstimator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

# device = 'cuda'
# device = 'cpu'


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
    def __init__(self, num_features,num_outs, n_hid=128, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            
            nn.Linear(num_features, n_hid),
            nn.ReLU(),
            
            nn.Dropout(dropout),            
            #nn.Linear(n_hid, n_hid // 4),
            #nn.ReLU(),
            #nn.BatchNorm1d(n_hid // 4),
            #nn.Dropout(dropout),
            nn.Linear(n_hid , num_outs),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_tensor):
        return self.model(input_tensor)
        
class EDMDSolverTorch(object):
    '''
    Plain EDMD solver using a neural network dictionary.
    '''
    def __init__(self, dic, target_dim, reg=0.0, checkpoint_file='edmd_koopman_net.torch', batch_size=32):
        self.dic = dic
        self.dic_func = dic.forward
        self.target_dim = target_dim
        self.reg = reg
        self.psi_x = None
        self.psi_y = None
        self.checkpoint_file = checkpoint_file
        self.batch_size = batch_size
        self.koopman_model = None
        self.koopman_optimizer = None

    def separate_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return data_x, data_y

    def build(self, data_train):
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)
        self.compute_final_info(reg_final=self.reg)

    def compute_final_info(self, reg_final):
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, reg=reg_final)
        self.K_np = self.K.detach().cpu().numpy()
        self.eig_decomp(self.K_np)

    def compute_K(self, dic, data_x, data_y, reg):
        data_x = torch.DoubleTensor(data_x).to(device)
        data_y = torch.DoubleTensor(data_y).to(device)
        psi_x = dic(data_x)
        psi_y = dic(data_y)
        self.Psi_X = psi_x
        self.Psi_Y = psi_y
        psi_xt = psi_x.T
        idmat = torch.eye(psi_x.shape[-1]).to(device)
        xtx_inv = torch.linalg.pinv(reg * idmat + torch.matmul(psi_xt, psi_x))
        xty = torch.matmul(psi_xt, psi_y)
        self.K_reg = torch.matmul(xtx_inv, xty)
        return self.K_reg

    def eig_decomp(self, K):
        self.eigenvalues, self.eigenvectors = la.eig(K)
        idx = self.eigenvalues.real.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        self.eigenvectors_inv = la.inv(self.eigenvectors)

    def eigenfunctions(self, data_x):
        data_x = torch.DoubleTensor(data_x).to(device)
        psi_x = self.dic_func(data_x)
        psi_x = psi_x.detach().cpu().numpy()
        val = np.matmul(psi_x, self.eigenvectors)
        return val

    def get_Psi_X(self):
        return self.Psi_X

    def get_Psi_Y(self):
        return self.Psi_Y

    def build_model(self):
        self.koopman_model = KoopmanModelTorch(dict_net=self.dic, target_dim=self.target_dim, k_dim=self.K.shape[0]).to(device)
        dict_params = [p for n,p in self.koopman_model.named_parameters() if "layer_K.weight" not in n]
        self.koopman_optimizer = torch.optim.Adam(dict_params, lr=1e-3, weight_decay=1e-5)

    def fit_koopman_model(self, koopman_model, koopman_optimizer, checkpoint_file, xx_train, yy_train, xx_test, yy_test,
                      batch_size=32, lrate=1e-4, epochs=1000, initial_loss=1e15):
        train_dataset = torch.utils.data.TensorDataset(
            torch.DoubleTensor(xx_train),
            torch.DoubleTensor(yy_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.DoubleTensor(xx_test),
            torch.DoubleTensor(yy_test)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True
        )
        n_epochs = epochs
        best_loss = initial_loss
        mlp_mdl = koopman_model
        optimizer = koopman_optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrate
        criterion = nn.MSELoss()
        mlp_mdl.train()
        val_loss_list = []
        for epoch in range(n_epochs):
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = mlp_mdl(data, target)
                zeros_tensor = torch.zeros_like(output)
                loss = criterion(output, zeros_tensor)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                del data, target, output, zeros_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output_val = mlp_mdl(data, target)
                    zeros_tensor = torch.zeros_like(output_val)
                    loss = criterion(output_val, zeros_tensor)
                    val_loss += loss.item() * data.size(0)
                    del data, target, output_val, zeros_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            val_loss = val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} val loss: {:.6f}'.format(
                epoch + 1, train_loss, val_loss))
            if val_loss < best_loss:
                print('saving, val loss enhanced:', val_loss, best_loss)
                torch.save({
                    'model_state_dict': mlp_mdl.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file)
                best_loss = val_loss
        mlp_mdl.layer_K.requires_grad = False
        koopman_model = mlp_mdl
        koopman_optimizer = optimizer
        return val_loss_list, best_loss

    def train_psi(self, koopman_model, koopman_optimizer, epochs, lr, initial_loss=1e15):
        data_x_val, data_y_val = self.separate_data(self.data_valid)
        psi_losses, best_psi_loss = self.fit_koopman_model(self.koopman_model, koopman_optimizer, self.checkpoint_file, self.data_x_train,
                                                      self.data_y_train, data_x_val, data_y_val, self.batch_size,
                                                      lrate=lr, epochs=epochs, initial_loss=initial_loss)
        return psi_losses, best_psi_loss

    def build_with_edmd(self, data_train, data_valid, epochs, batch_size, lr, log_interval, lr_decay_factor):
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)
        self.data_valid = data_valid
        self.batch_size = batch_size
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, reg=self.reg)
        if not hasattr(self, 'koopman_model') or self.koopman_model is None:
            self.build_model()
        with torch.no_grad():
            self.koopman_model.layer_K.weight.data.copy_(self.K.T)
        self.koopman_model.layer_K.weight.requires_grad = False
        losses = []
        curr_lr = lr
        curr_last_loss = 1e15
        for ii in arange(epochs):
            print(f"Outer Epoch {ii+1}/{epochs}")
            self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, reg=self.reg)
            with torch.no_grad():
                self.koopman_model.layer_K.weight.data.copy_(self.K.T)
            self.koopman_model.layer_K.weight.requires_grad = False
            curr_losses, curr_best_loss = self.train_psi(self.koopman_model, self.koopman_optimizer, epochs=3, lr=curr_lr, initial_loss=curr_last_loss)
            if curr_last_loss > curr_best_loss:
                curr_last_loss = curr_best_loss
            if ii % log_interval == 0:
                losses.append(curr_losses[-1])
                if len(losses) > 2:
                    if losses[-1] > losses[-2]:
                        print("Error increased. Decay learning rate")
                        curr_lr = lr_decay_factor * curr_lr
        checkpoint = torch.load(self.checkpoint_file)
        self.koopman_model.load_state_dict(checkpoint['model_state_dict'])
        self.koopman_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.compute_final_info(reg_final=self.reg)