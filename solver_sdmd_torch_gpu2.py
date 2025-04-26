import time
import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as la
from numpy import arange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.func import jacrev
import joblib
from sde_coefficients_estimator import SDECoefficientEstimator

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

# device = 'cuda'
# device = 'cpu'


class KoopmanNNTorch(nn.Module):
    def __init__(self, input_size, layer_sizes=[64, 64], n_psi_train=22, **kwargs):
        super(KoopmanNNTorch, self).__init__()
        self.layer_sizes = layer_sizes
        self.n_psi_train = n_psi_train  # Using n_psi_train directly, consistent with DicNN
        
        self.layers = nn.ModuleList()
        bias = False
        n_layers = len(layer_sizes)
        
        self.layers.append(nn.Linear(input_size, layer_sizes[0], bias=bias))
        for ii in arange(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[ii - 1], layer_sizes[ii], bias=True))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(layer_sizes[n_layers - 1], n_psi_train, bias=True))
    
    def forward(self, x):
        in_x = x
        for layer in self.layers:
            x = layer(x)
        const_out = torch.ones_like(in_x[:, :1])  # print (const_out)
        x = torch.cat([const_out, in_x, x], dim=1)
        return x
    


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
        
class KoopmanSolverTorch(object):
    '''
    Build the Koopman solver

    This part represents a Koopman solver that can be used to build and solve Koopman operator models.

    Attributes:
        dic (class): The dictionary class used for Koopman operator approximation.
        dic_func (function): The dictionary functions used for Koopman operator approximation.
        target_dim (int): The dimension of the variable of the equation.
        reg (float, optional): The regularization parameter when computing K. Defaults to 0.0.
        psi_x (None): Placeholder for the feature matrix of the input data.
        psi_y (None): Placeholder for the feature matrix of the output data.
    '''

    def __init__(self, dic, target_dim, reg=0.0, checkpoint_file='example_koopman_net001.torch', fnn_checkpoint_file= 'example_fnn001.torch', 
                a_b_file= None, generator_batch_size= 4, fnn_batch_size= 32, delta_t= 0.1):
        """Initializer

        :param dic: dictionary
        :type dic: class
        :param target_dim: dimension of the variable of the equation
        :type target_dim: int
        :param reg: the regularization parameter when computing K, defaults to 0.0
        :type reg: float, optional
        """
        self.dic = dic  # dictionary class
        self.dic_func = dic.forward  # dictionary functions
        self.target_dim = target_dim
        self.reg = reg
        self.psi_x = None
        self.psi_y = None
        self.dPsi_X= None
        self.dPsi_Y= None
        self.checkpoint_file = checkpoint_file
        self.fnn_checkpoint_file=fnn_checkpoint_file
        self.generator_batch_size= generator_batch_size
        self.fnn_batch_size= fnn_batch_size
        self.delta_t= delta_t
        self.a_b_file= a_b_file

    def separate_data(self, data):
        data_x = data[0]
        data_y = data[1]
        return data_x, data_y

    def build(self, data_train):
        # Separate data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        # Compute final information
        self.compute_final_info(reg_final=0.0)

    def compute_final_info(self, reg_final):
        # Compute K
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, reg=reg_final)
        self.K_np = self.K.detach().cpu().numpy()
        self.eig_decomp(self.K_np)
        #self.compute_mode()

    def eig_decomp(self, K):
        """ eigen-decomp of K """
        self.eigenvalues, self.eigenvectors = la.eig(K)
        idx = self.eigenvalues.real.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        self.eigenvectors_inv = la.inv(self.eigenvectors)

    def eigenfunctions(self, data_x):
        """ estimated eigenfunctions """
        data_x = torch.DoubleTensor(data_x).to(device)
        psi_x = self.dic_func(data_x)
        psi_x = psi_x.detach().cpu().numpy()
        val = np.matmul(psi_x, self.eigenvectors)
        return val

    # def compute_mode(self):
    #     self.basis_func_number = self.K.shape[0]

    #     # Form B matrix
    #     self.B = self.dic.generate_B(self.data_x_train)

    #     # Compute modes
    #     self.modes = np.matmul(self.eigenvectors_inv, self.B).T
    #     return self.modes

    # def calc_psi_next(self, data_x, K):
    #     psi_x = self.dic_func(data_x)
    #     psi_next = torch.matmul(psi_x, K)
    #     return psi_next

    def compute_K(self, dic, data_x, data_y, reg):
        data_x = torch.DoubleTensor(data_x).to(device)
        data_y = torch.DoubleTensor(data_y).to(device)
        psi_x = dic(data_x)
        psi_y = dic(data_y)
        
        # Compute Psi_X and Psi_Y
        self.Psi_X = dic(data_x)
        self.Psi_Y = dic(data_y)
        
        psi_xt = psi_x.T
        idmat = torch.eye(psi_x.shape[-1]).to(device)
        xtx_inv = torch.linalg.pinv(reg * idmat + torch.matmul(psi_xt, psi_x))
        xty = torch.matmul(psi_xt, psi_y)
        self.K_reg = torch.matmul(xtx_inv, xty)
        return self.K_reg

    def compute_K_with_generator (self, dic, data_x, data_y, reg):
        data_x = torch.DoubleTensor(data_x).to(device)
        data_y = torch.DoubleTensor(data_y).to(device)
        #print ('data_x:', data_x.shape)
        psi_x = dic(data_x[:-1, :])
        psi_y = dic(data_y)
        
        # Compute Psi_X and Psi_Y
        self.Psi_X = dic(data_x)
        self.Psi_Y = dic(data_y)
        
        psi_xt = psi_x.T
        # idmat = torch.eye(psi_x.shape[-1]).to(device)
        # xtx_inv = torch.linalg.pinv(reg * idmat + torch.matmul(psi_xt, psi_x))
        # xt_gen = torch.matmul(psi_xt, psi_y)
        PsiX_np= self.Psi_X.detach().cpu().numpy()
        L_Psi_X_np= self.L_Psi.detach().cpu().numpy()
        dt= self.delta_t
        dPsiX_np = self.dPsi_X.detach().cpu().numpy()
        K_np = np.linalg.pinv(PsiX_np[:-1,:].T @ PsiX_np[:-1,:]) @ (PsiX_np[:-1,:].T @ (PsiX_np[:-1,:] + dt * dPsiX_np))
        #K_np  = np.linalg.inv(PsiX_np.T.conj() @ PsiX_np + reg * np.eye(PsiX_np.shape[1])) @ (PsiX_np.T.conj() @ (PsiX_np + dt * L_Psi_X_np))
        
        #self.K_gen = torch.complex (torch.DoubleTensor(K_np.real), torch.DoubleTensor(K_np.imag))
        self.K_gen = torch.DoubleTensor(K_np).to(device)
        return self.K_gen

    def get_Psi_X(self):
        return self.Psi_X

    def get_Psi_Y(self):
        return self.Psi_Y

    def build_model(self):
        self.koopman_model = KoopmanModelTorch(dict_net=self.dic, target_dim=self.target_dim, k_dim=self.K.shape[0]).to(device)
        
    def fit_koopman_model(self, koopman_model, koopman_optimizer, checkpoint_file, xx_train, yy_train, xx_test, yy_test,
                      batch_size=32, lrate=1e-4, epochs=1000, initial_loss=1e15):
        load_best = False
        xx_train_tensor = torch.DoubleTensor((xx_train)).to(device)
        yy_train_tensor = torch.DoubleTensor((yy_train)).to(device)
        xx_test_tensor = torch.DoubleTensor((xx_test)).to(device)
        yy_test_tensor = torch.DoubleTensor((yy_test)).to(device)
    
        train_dataset = torch.utils.data.TensorDataset(xx_train_tensor, yy_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
        val_dataset = torch.utils.data.TensorDataset(xx_test_tensor, yy_test_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
        n_epochs = epochs
        best_loss = initial_loss
        mlp_mdl = koopman_model
        #optimizer = torch.optim.Adam(mlp_mdl.parameters(), lr=lrate, weight_decay=1e-5)
        optimizer = koopman_optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrate
        criterion = nn.MSELoss()
    
        mlp_mdl.train()
        val_loss_list = []
    
        for epoch in range(n_epochs):
            train_loss = 0.0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = mlp_mdl(data, target)
                zeros_tensor = torch.zeros_like(output)
                loss = criterion(output, zeros_tensor)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
            train_loss = train_loss / len(train_loader.dataset)
    
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    output_val = mlp_mdl(data, target)
                    zeros_tensor = torch.zeros_like(output_val)
                    loss = criterion(output_val, zeros_tensor)
                    val_loss += loss.item() * data.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            val_loss_list.append(val_loss)
    
            print('Epoch: {} \tTraining Loss: {:.6f} val loss: {:.6f}'.format(
                epoch + 1, train_loss, val_loss))
    
            if val_loss < best_loss:
                print('saving, val loss enhanced:', val_loss, best_loss)
                #torch.save(mlp_mdl.state_dict(), checkpoint_file)
                torch.save({
                'model_state_dict': mlp_mdl.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file)
                best_loss = val_loss
                load_best = True
    
        if load_best:
            #mlp_mdl.load_state_dict(torch.load(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            mlp_mdl.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mlp_mdl.layer_K.requires_grad = False
            koopman_model = mlp_mdl
            koopman_optimizer= optimizer
    
        return val_loss_list, best_loss

    def train_psi(self, koopman_model, koopman_optimizer, epochs, lr, initial_loss=1e15):
        data_x_val, data_y_val = self.separate_data(self.data_valid)
        psi_losses, best_psi_loss = self.fit_koopman_model(self.koopman_model, koopman_optimizer, self.checkpoint_file, self.data_x_train,
                                                      self.data_y_train, data_x_val, data_y_val, self.batch_size,
                                                      lrate=lr, epochs=epochs, initial_loss=initial_loss)
        return psi_losses, best_psi_loss

    # # This is the separate NN for training the SDE coefficients
    # def fit_fnn_model(self, fnn_model, fnn_optimizer, checkpoint_file, xx_train, yy_train, xx_test, yy_test,
    #                   fnn_batch_size=32, lrate=1e-4, epochs=1000, initial_loss=10000):
    #     load_best = False
    #     if not torch.is_tensor (xx_train):
    #         xx_train_tensor = torch.DoubleTensor((xx_train)).to(device)
    #         yy_train_tensor = torch.DoubleTensor((yy_train)).to(device)
    #         xx_test_tensor = torch.DoubleTensor((xx_test)).to(device)
    #         yy_test_tensor = torch.DoubleTensor((yy_test)).to(device)
    #     else:
    #         xx_train_tensor = xx_train.to(device)
    #         yy_train_tensor = yy_train.to(device)
    #         xx_test_tensor = xx_test.to(device)
    #         yy_test_tensor = yy_test.to(device)
            
    #     #print (xx_train_tensor.shape, yy_train_tensor.shape)
    #     train_dataset = torch.utils.data.TensorDataset(xx_train_tensor, yy_train_tensor)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=fnn_batch_size, shuffle=False)
    
    #     val_dataset = torch.utils.data.TensorDataset(xx_test_tensor, yy_test_tensor)
    #     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=fnn_batch_size, shuffle=False)
    
    #     n_epochs = epochs
    #     best_loss = initial_loss
    #     mlp_mdl = fnn_model
    #     #optimizer = torch.optim.Adam(mlp_mdl.parameters(), lr=lrate, weight_decay=1e-5)
    #     optimizer = fnn_optimizer
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lrate
    #     criterion = nn.MSELoss()
    
    #     mlp_mdl.train()
    #     val_loss_list = []
    
    #     for epoch in range(n_epochs):
    #         train_loss = 0.0
    #         for data, target in train_loader:
    #             optimizer.zero_grad()
    #             output = mlp_mdl(data)
               
    #             loss = criterion(output,target)
    #             loss.backward()
    #             optimizer.step()
    #             train_loss += loss.item() * data.size(0)
    #         train_loss = train_loss / len(train_loader.dataset)
    
    #         val_loss = 0.0
    #         with torch.no_grad():
    #             for data, target in val_loader:
    #                 output_val = mlp_mdl(data)
    
    #                 loss = criterion(output_val, target)
    #                 val_loss += loss.item() * data.size(0)
    #         val_loss = val_loss / len(val_loader.dataset)
    #         val_loss_list.append(val_loss)
    
    #         print('Epoch: {} \tTraining Loss: {:.6f} val loss: {:.6f}'.format(
    #             epoch + 1, train_loss, val_loss))
    
    #         if val_loss < best_loss:
    #             print('saving, val loss enhanced:', val_loss, best_loss)
    #             #torch.save(mlp_mdl.state_dict(), checkpoint_file)
    #             torch.save({
    #             'model_state_dict': mlp_mdl.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             }, checkpoint_file)
    #             best_loss = val_loss
    #             load_best = True
                
    def process_batch(self, batch_inputs):
        #batch_inputs= torch.DoubleTensor(batch_inputs, requires_grad= True)
        batch_inputs.requires_grad_()
        #print (batch_inputs.shape)
        batch_outputs = self.dic(batch_inputs)
        #print (batch_outputs.shape)
        
        #batch_first_derivatives =jacobian(self.dic, batch_inputs, create_graph= True)
        batch_first_derivatives00= jacrev(self.dic)(batch_inputs)
        batch_first_derivatives= batch_first_derivatives00.sum(2)       
        batch_second_derivatives00 =jacrev(lambda b_inputs:jacrev(self.dic)(b_inputs)) ( batch_inputs)
        batch_second_derivatives= batch_second_derivatives00.sum ((2, 4))      
        return batch_first_derivatives, batch_second_derivatives

    def compute_dPsi_X(self, data_x, b_Xt, a_Xt, delta_t):
        """
        Vectorized computation of dPsi_X:
        Removes nested Python loops and uses four tensor operations instead.
        
        Args:
            data_x   (Tensor): shape (M, state_dim)
            b_Xt     (Tensor): shape (M-1, state_dim)  — drift coefficients
            a_Xt     (Tensor): shape (M-1, state_dim, state_dim) — diffusion matrices
            delta_t  (float): time step
        
        Returns:
            dPsi_X   (Tensor): shape (M-1, num_features)
        """
        # 1) Compute batched Jacobian and Hessian as before
        jacobian, hessian = self.get_derivatives(data_x, batch_size=self.generator_batch_size)
        # 2) Drop the last time point
        J = jacobian[:-1]   # (M-1, F, D)
        H = hessian[:-1]    # (M-1, F, D, D)
        B = b_Xt            # (M-1, D)
        A = a_Xt            # (M-1, D, D)

        # 3) term1[i,j] = sum_k J[i,j,k] * B[i,k]
        term1 = (J * B.unsqueeze(1)).sum(dim=-1)           # (M-1, F)
        # 4) term2[i,j] = 0.5 * sum_{k,l} H[i,j,k,l] * A[i,k,l]
        term2 = 0.5 * (H * A.unsqueeze(1)).sum(dim=(-1, -2))  # (M-1, F)

        # 5) Multiply by time step
        dPsi_X = (term1 + term2) * delta_t
        return dPsi_X

    def get_derivatives(self, inputs, batch_size=4):
        """
        Efficient batched derivative extractor.
        Processes inputs in mini-batches and returns:
          - first_derivatives: Jacobian sums, shape (N, F, D)
          - second_derivatives: Hessian sums, shape (N, F, D, D)
        
        Args:
            inputs     (Tensor): shape (N, state_dim)
            batch_size (int): size of each mini-batch
        
        Returns:
            first_derivatives  (Tensor): shape (N, F, D)
            second_derivatives (Tensor): shape (N, F, D, D)
        """
        num_samples = inputs.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        first_list = []
        second_list = []
        
        # Process in batches to limit memory use
        for i in tqdm(range(num_batches), desc='Processing batches', unit='batch'):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            batch = inputs[start:end].requires_grad_()
            
            # Compute Jacobian of each output feature w.r.t. inputs
            jac0 = jacrev(self.dic)(batch)    # shape (B, F, D)
            jac = jac0.sum(dim=2)             # sum over D for consistency
            
            # Compute Hessian similarly
            hess0 = jacrev(lambda x: jacrev(self.dic)(x))(batch)  # (B, F, D, D)
            hess = hess0.sum(dim=(2, 4))       # sum over D,D for consistency
            
            first_list.append(jac)
            second_list.append(hess)
        
        first_derivatives = torch.cat(first_list, dim=0)
        second_derivatives = torch.cat(second_list, dim=0)
        return first_derivatives, second_derivatives



    # Modify the compute_neural_a_b method in KoopmanSolverTorch class
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
        hidden_size = 128  # You can change this
        n_hidden_layers = 1  # You can change this
        dropout = 0.01  # You can change this
        
        sde_estimator.build_model(
            state_dim=state_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            n_hidden_layers=n_hidden_layers
        )
        
        # Train the model
        learning_rate = 5e-4  # You can change this
        epochs = 50  # You can change this
        batch_size = self.fnn_batch_size  # Use the batch size from the class
        
        sde_estimator.fit_model(
            X_t_1=X_t_1,
            X_t=X_t,
            checkpoint_file=self.fnn_checkpoint_file,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs
        )
        
        # Estimate the coefficients
        b_Xt, a_Xt_final = sde_estimator.estimate_coefficients(X_t_1, X_t, delta_t)
        
        return b_Xt, a_Xt_final

    
    def compute_generator_L(self, data_x, b_Xt, a_Xt, delta_t, lambda_reg=0.01):
        # Compute dPsi_X
        data_x= data_x.to(device)
        dPsi_X = self.compute_dPsi_X(data_x.to(device), b_Xt, a_Xt, delta_t)
        self.dPsi_X= dPsi_X
        print("dPsi_X shape: ", dPsi_X.shape)
        
        # Compute Psi_X^{-1}
        psi_x = self.dic(data_x[:-1])
        psi_x_inv = torch.linalg.pinv(psi_x)
        print("psi_x shape: ", psi_x.shape)
        print("psi_x_inv shape: ", psi_x_inv.shape)


        # Compute the transpose of psi_x
        psi_x_transpose = psi_x.T.to (device) #torch.transpose(psi_x)
        
       
        # Compute the matrix product of psi_x^T and psi_x defined as 'G'
        G = torch.matmul(psi_x_transpose, psi_x)        
        # Add regularization term to avoid singularity issue
        G_reg = G + lambda_reg * torch.eye(G.shape[0], dtype=G.dtype).to (device)        
        # Compute the inverse of the regularized matrix product
        G_inv = torch.linalg.pinv(G_reg)
        # Compute the matrix product of psi_x^T and dPsi_X defined as 'A'
        print (psi_x_transpose.device)
        print (dPsi_X.device)
        A = torch.matmul(psi_x_transpose, dPsi_X)        
        # Cast A to match the data type of G_inv, if necessary
        #A = torch.cast(A, dtype=G_inv.dtype)        
        # Compute L = G^{-1} * A = (psi_x^T * psi_x)^{-1} * (psi_x^T * dPsi_X)
        L_Psi = torch.matmul(G_inv, A)
        self.L_Psi= L_Psi
        return L_Psi


    
    def build(self, data_train, data_valid, epochs, batch_size, lr, log_interval, lr_decay_factor):
        """Train Koopman model and calculate the final information,
        such as eigenfunctions, eigenvalues and K.
        For each outer training epoch, the koopman dictionary is trained
        by several times (inner training epochs), and then compute matrix K.
        Iterate the outer training.

        :param data_train: training data
        :type data_train: [data at the current time, data at the next time]
        :param data_valid: validation data
        :type data_valid: [data at the current time, data at the next time]
        :param epochs: the number of the outer epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param log_interval: the patience of learning decay
        :type log_interval: int
        :param lr_decay_factor: the ratio of learning decay
        :type lr_decay_factor: float
        """
        # Separate training data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        self.data_valid = data_valid

        self.batch_size = batch_size
        self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, self.reg)
        
        # Build the Koopman DL model
        self.build_model()

        losses = []
        curr_lr = lr
        curr_last_loss = 1e12
        self.koopman_optimizer= torch.optim.Adam(self.koopman_model.parameters(), lr=lr, weight_decay=1e-5)
        for ii in arange(epochs):
            start_time = time.time()
            print(f"Outer Epoch {ii+1}/{epochs}")
            
            # One step for computing K
            self.K = self.compute_K(self.dic_func, self.data_x_train, self.data_y_train, self.reg)
            
            with torch.no_grad():
                self.koopman_model.layer_K.weight.data = self.K

            # Two steps for training PsiNN
            curr_losses, curr_best_loss = self.train_psi(self.koopman_model, self.koopman_optimizer, epochs=4, lr=curr_lr, initial_loss=curr_last_loss)
            
            if curr_last_loss > curr_best_loss:
                curr_last_loss = curr_best_loss

            if ii % log_interval == 0:
                losses.append(curr_losses[-1])

                # Adjust learning rate:
                if len(losses) > 2:
                    if losses[-1] > losses[-2]:
                        print("Error increased. Decay learning rate")
                        curr_lr = lr_decay_factor * curr_lr

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {ii+1} time: {epoch_time:.2f} seconds")

        # Compute final information
        #self.koopman_model.load_state_dict(torch.load(self.checkpoint_file))
        checkpoint = torch.load(self.checkpoint_file)
        self.koopman_model.load_state_dict(checkpoint['model_state_dict'])
        self.koopman_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.compute_final_info(reg_final=0.01)

    def build_with_generator(self, data_train, data_valid, epochs, batch_size, lr, log_interval, lr_decay_factor):
        """Train Koopman model and calculate the final information,
        such as eigenfunctions, eigenvalues and K.
        For each outer training epoch, the koopman dictionary is trained
        by several times (inner training epochs), and then compute matrix K.
        Iterate the outer training.

        :param data_train: training data
        :type data_train: [data at the current time, data at the next time]
        :param data_valid: validation data
        :type data_valid: [data at the current time, data at the next time]
        :param epochs: the number of the outer epochs
        :type epochs: int
        :param batch_size: batch size
        :type batch_size: int
        :param lr: learning rate
        :type lr: float
        :param log_interval: the patience of learning decay
        :type log_interval: int
        :param lr_decay_factor: the ratio of learning decay
        :type lr_decay_factor: float
        """
        # Separate training data
        self.data_train = data_train
        self.data_x_train, self.data_y_train = self.separate_data(self.data_train)

        self.data_valid = data_valid

        self.batch_size = batch_size
        data_x_train_tensor= torch.DoubleTensor(self.data_x_train)
        #here we load compute drift and diffusion coefficents using feed-forward neural network 
        self.b_Xt, self. a_Xt = self.compute_neural_a_b(data_x_train_tensor, delta_t= self.delta_t)
        self. L_Psi = self.compute_generator_L(data_x_train_tensor, self.b_Xt, self.a_Xt, self.delta_t)
        self.K = self.compute_K_with_generator (self.dic_func, self.data_x_train, self.data_y_train, self.reg)
        # here we save drift and diffusion coefficents to  the joblib file, if filename  is specified.
        if (self.a_b_file is not None):
            a_Xt_np= self.a_Xt.detach().cpu().numpy()
            b_Xt_np= self.b_Xt.detach().cpu().numpy()
            print ('saving FNN a and b to: ', self.a_b_file )
            joblib.dump ((a_Xt_np,b_Xt_np), self.a_b_file)
            
        # Build the Koopman DL model
        self.build_model()

        losses = []
        curr_lr = lr
        curr_last_loss = 1e15
        self.koopman_optimizer= torch.optim.Adam(self.koopman_model.parameters(), lr=lr, weight_decay=1e-5)
        for ii in arange(epochs):
            #starting outer epoch. In each outer epoch we compute generator L
            #Koopman operator K is computed from L each outer epoch,
            # and the matrix K is set as weighths of layer K of our Koopman NN. 
            #then we do several steps of training our NN that is the dictionary
            start_time = time.time()
            print(f"Outer Epoch {ii+1}/{epochs}")
            
            # One step for computing L and  K
            self. L_Psi = self.compute_generator_L(data_x_train_tensor, self.b_Xt, self. a_Xt, self.delta_t)
            self.K = self.compute_K_with_generator(self.dic_func, self.data_x_train, self.data_y_train, self.reg)
            
            with torch.no_grad():
                self.koopman_model.layer_K.weight.data = self.K

            #  steps (inner epochs) for training PsiNN, the number of inner epochs is given by epochs parameter below, here epochs= 4
            curr_losses, curr_best_loss = self.train_psi(self.koopman_model, self.koopman_optimizer, epochs=4, lr=curr_lr, initial_loss=curr_last_loss)
            
            if curr_last_loss > curr_best_loss:
                curr_last_loss = curr_best_loss

            if ii % log_interval == 0:
                losses.append(curr_losses[-1])

                # Adjust learning rate:
                if len(losses) > 2:
                    if losses[-1] > losses[-2]:
                        print("Error increased. Decay learning rate")
                        curr_lr = lr_decay_factor * curr_lr

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Epoch {ii+1} time: {epoch_time:.2f} seconds")

        # Compute final information
        #self.koopman_model.load_state_dict(torch.load(self.checkpoint_file))
        checkpoint = torch.load(self.checkpoint_file)
        self.koopman_model.load_state_dict(checkpoint['model_state_dict'])
        self.koopman_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.compute_final_info(reg_final=0.01)