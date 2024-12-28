import torch
import numpy as np
import torch.nn as nn
import os
from torch.linalg import vector_norm
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm


### Generate data

# Default parameter vector 'a' if not provided
    # a: 64-dimensional: 
    # a[0] = 1.0
    # a[30], a[58] are non-zero
    # all others 0.0
    
a_true = torch.zeros(64, dtype=torch.float32)
nonzero_indices = [0, 30, 58]
for idx, val in zip(nonzero_indices, [1.0, 1.2, 1.5]):
    a_true[idx] = val


def preanm_simulator_64d(true_function="softplus", n=10000, x_lower=0, x_upper=2, noise_std=1, 
                         noise_dist="gaussian", train=True, device=torch.device("cpu"), 
                         a=a_true, noise_corr=0):
    """Data simulator for a pre-additive noise model (pre-ANM) with 64-dimensional covariates.
    
    Args:
        true_function (str or callable, optional): the true function g*. Defaults to "softplus". 
            Choices: ["softplus", "square", "log", "cubic"] or a callable.
        n (int, optional): sample size. Defaults to 10000.
        x_lower (float, optional): lower bound of the training support. Defaults to 0.
        x_upper (float, optional): upper bound of the training support. Defaults to 2.
        noise_std (float, optional): standard deviation of the noise. Defaults to 1.
        noise_dist (str, optional): noise distribution. "gaussian" or "uniform". Defaults to "gaussian".
        train (bool, optional): if True, generates data for training. If False, generates evaluation data.
        device (str or torch.device, optional): device to place tensors. Defaults to CPU.
        a (torch.Tensor, optional): a 64-dim vector for the linear transformation. 
            If None, the vector is constructed as follows:
            a[0] = 1.0
            a[2], a[30], a[58] = some non-zero values
            all other elements = 0.0
        noise_corr (float, optional): length-scale parameter for the squared exponential kernel.
            If noise_corr=0, no correlation between pixels.

    Returns:
        For train=True:
            x (torch.Tensor): shape (n,64), input data
            y (torch.Tensor): shape (n,1), output data
        For train=False:
            x_eval (torch.Tensor): shape (n,64), evaluation inputs
            y_eval_med (torch.Tensor): shape (n,1), median (or deterministic) function values
            y_eval_mean (torch.Tensor): shape (n,1), mean output after noise
    """

    if isinstance(device, str):
        device = torch.device(device)
    
    if isinstance(true_function, str):
        if true_function == "softplus":
            true_function = lambda x: nn.Softplus()(x)
        elif true_function == "square":
            true_function = lambda x: (nn.functional.relu(x)).pow(2)/20
        elif true_function == "log":
            true_function = lambda x: (x/3 + np.log(3) - 2/3)*(x <= 2) + (torch.log(1 + x*(x > 2)))*(x > 2) 
        elif true_function == "cubic":
            true_function = lambda x: x.pow(3)/30

    a = a_true.to(device)
    nonzero_indices = [0, 30, 58]  # exactly 10 indices
    nonzero_indices = torch.tensor(nonzero_indices, dtype=torch.long, device=device)
    
    all_indices = torch.arange(64, device=device)
    mask = torch.ones(64, dtype=torch.bool, device=device)
    mask[nonzero_indices] = False
    zero_indices = all_indices[mask]  # these are the remaining 54 indices

    def pixel_distance(i, j, size=8):
        i_row, i_col = divmod(i, size)
        j_row, j_col = divmod(j, size)
        dist = np.sqrt((i_row - j_row)**2 + (i_col - j_col)**2)
        return dist

    def build_cov_matrix(dim=64, noise_std=1, noise_corr=0):
        cov = torch.zeros(dim, dim, device=device)
        if noise_corr == 0:
            # No correlation, just diagonal
            cov = (noise_std**2)*torch.eye(dim, device=device)
            return cov
        
        # If noise_corr > 0, use squared exponential kernel
        # cov[i,j] = noise_std^2 * exp(-(distance(i,j)^2)/(2*noise_corr^2))
        for i in range(dim):
            for j in range(dim):
                dist = pixel_distance(i, j, size=8)
                cov[i,j] = (noise_std**2)*np.exp(-(dist**2)/(2*(noise_corr**2)))
        return cov

    def generate_correlated_noise(n_samples, effect=True, noise_dist="gaussian", noise_std=1, noise_corr=0):
        if effect:
            dim=3
            a_effect = a[nonzero_indices]
            cov_matrix_init = build_cov_matrix(dim=3, noise_std=noise_std, noise_corr=noise_corr)
            var_factor = a_effect @ cov_matrix_init @ a_effect  
        else:
            dim=61
            cov_matrix_init = build_cov_matrix(dim=61, noise_std=noise_std, noise_corr=noise_corr) 
            var_factor = torch.tensor(1) 
         
        cov_matrix = (1/var_factor.item()) * cov_matrix_init ## modify the variance to account for the effect of inner product 
        L = torch.linalg.cholesky(cov_matrix)
        z = torch.randn(n_samples, dim, device=device)
        ERR = z @ L.T
        if noise_dist == "gaussian":
            eps = ERR
        else:
            # Transform Gaussian to uniform(-0.5,0.5), then scale
            eps = torch.distributions.Normal(0, 1).cdf(ERR) - 0.5
            eps = eps * np.sqrt(12)
        return eps


        
    if train:
        
        x_eff = (x_upper - x_lower) * torch.rand(n, 3, device=device) + x_lower
        x_noneff = (x_upper - x_lower) * torch.rand(n, 61, device=device) + x_lower
        
        x = torch.empty(n, 64, device=device)
        x[:, nonzero_indices] = x_eff
        x[:, zero_indices] = x_noneff
        
        #x = (x_upper - x_lower) * torch.rand(n, 64, device=device) + x_lower

        eps_eff = generate_correlated_noise(n, effect=True, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)
        eps_noneff = generate_correlated_noise(n, effect=False, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)
        
        eps = torch.empty(n, 64, device=device)
        eps[:, nonzero_indices] = eps_eff
        eps[:, zero_indices] = eps_noneff

        xn = x + eps

        s = xn @ a.unsqueeze(1)
        
        y = true_function(s)

        return x.to(device), y.to(device)

    else:
        
        x_eval_eff = torch.linspace(x_lower, x_upper, n, device=device).unsqueeze(1).repeat(1, 3)
        x_eval_noneff = torch.zeros(61,device=device)
        
        x_eval = torch.empty(n, 64, device=device)
        x_eval[:, nonzero_indices] = x_eval_eff
        x_eval[:, zero_indices] = x_eval_noneff
       
        #x_eval = torch.linspace(x_lower, x_upper, n, device=device).unsqueeze(1).repeat(1, 64)

        s_eval = x_eval @ a.unsqueeze(1)
        y_eval_med = true_function(s_eval)

        gen_sample_size = 10000
        x_rep = torch.repeat_interleave(x_eval, gen_sample_size, dim=0)

        #eps = generate_correlated_noise(x_rep.size(0), 64, noise_dist, noise_std, noise_corr)
        eps_eff = generate_correlated_noise(x_rep.size(0), effect=True, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)
        eps_noneff = generate_correlated_noise(x_rep.size(0), effect=False, noise_dist=noise_dist, noise_std=noise_std, noise_corr=noise_corr)
        
        eps = torch.empty(x_rep.size(0), 64, device=device)
        eps[:, nonzero_indices] = eps_eff
        eps[:, zero_indices] = eps_noneff

        xn_rep = x_rep + eps
        s_rep = xn_rep @ a.unsqueeze(1)

        y_rep = true_function(s_rep)
        y_eval_mean = y_rep.view(n, gen_sample_size).mean(dim=1).unsqueeze(1)

        return x_eval.to(device), y_eval_med.to(device), y_eval_mean.to(device)


## utility functions


def vectorize(x, multichannel=False):
    """Vectorize data in any shape. Each row corresponds to a sample, and columns represent features.

    Args:
        x (torch.Tensor): input data
        multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

    Returns:
        torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
    """
    if len(x.shape) == 1:
        return x.unsqueeze(1)
    if len(x.shape) == 2:
        return x
    else:
        if not multichannel: # one channel
            return x.reshape(x.shape[0], -1)
        else: # multi-channel
            return x.reshape(x.shape[0], x.shape[1], -1)
        
def cor(x, y):
    """Compute the correlation between two signals.

    Args:
        x (torch.Tensor): input data
        y (torch.Tensor): input data

    Returns:
        torch.Tensor: correlation between x and y
    """
    x = vectorize(x)
    y = vectorize(y)
    x = x - x.mean(0)
    y = y - y.mean(0)
    return ((x * y).mean()) / (x.std(unbiased=False) * y.std(unbiased=False))

def make_folder(name):
    """Make a folder.

    Args:
        name (str): folder name.
    """
    if not os.path.exists(name):
        print('Creating folder: {}'.format(name))
        os.makedirs(name)

def check_for_gpu(device):
    """Check if a CUDA device is available.

    Args:
        device (torch.device): current set device.
    """
    if device.type == "cuda":
        if torch.cuda.is_available():
            print("GPU is available, running on GPU.\n")
        else:
            print("GPU is NOT available, running instead on CPU.\n")
    else:
        if torch.cuda.is_available():
            print("Warning: You have a CUDA device, so you may consider using GPU for potential acceleration\n by setting device to 'cuda'.\n")
        else:
            print("Running on CPU.\n")


def make_dataloader(x, y=None, batch_size=128, shuffle=True, num_workers=0):
    """Make dataloader.

    Args:
        x (torch.Tensor): data of predictors.
        y (torch.Tensor): data of responses.
        batch_size (int, optional): batch size. Defaults to 128.
        shuffle (bool, optional): whether to shuffle data. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 0.

    Returns:
        DataLoader: data loader
    """
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def partition_data(x_full, y_full, cut_quantile=0.3, split_train="smaller"):
    """Partition data into training and test sets.

    Args:
        x_full (torch.Tensor): full data of x.
        y_full (torch.Tensor): full data of y.
        cut_quantile (float, optional): quantile of the cutting point of x. Defaults to 0.3.
        split_train (str, optional): which subset is used for for training. choices=["smaller", "larger"]. Defaults to "smaller".

    Returns:
        tuple of torch.Tensors: training and test data.
    """
    # Split data into training and test sets.
    x_cut = torch.quantile(x_full, cut_quantile)
    train_idx = x_full <= x_cut if split_train == "smaller" else x_full >= x_cut
    x_tr = x_full[train_idx]
    y_tr = y_full[train_idx]
    x_te = x_full[~train_idx]
    y_te = y_full[~train_idx]
    
    # Standardize data based on training statistics.
    x_tr_mean = x_tr.mean()
    x_tr_std = x_tr.std()
    y_tr_mean = y_tr.mean()
    y_tr_std = y_tr.std()
    x_tr = (x_tr - x_tr_mean)/x_tr_std
    y_tr = (y_tr - y_tr_mean)/y_tr_std
    x_te = (x_te - x_tr_mean)/x_tr_std
    y_te = (y_te - y_tr_mean)/y_tr_std
    x_full_normal = (x_full - x_tr_mean)/x_tr_std
    return x_tr.unsqueeze(1), y_tr.unsqueeze(1), x_te.unsqueeze(1), y_te.unsqueeze(1), x_full_normal         
            




## Energy Loss

def energy_loss(x_true, x_est, beta=1, verbose=True):
    """Loss function based on the energy score.

    Args:
        x_true (torch.Tensor): iid samples from the true distribution of shape (data_size, data_dim)
        x_est (list of torch.Tensor): 
            - a list of length sample_size, where each element is a tensor of shape (data_size, data_dim) that contains one sample for each data point from the estimated distribution, or 
            - a tensor of shape (data_size*sample_size, response_dim) such that x_est[data_size*(i-1):data_size*i,:] contains one sample for each data point, for i = 1, ..., sample_size.
        beta (float): power parameter in the energy score.
        verbose (bool): whether to return two terms of the loss.

    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x_true = vectorize(x_true).unsqueeze(1)
    if not isinstance(x_est, list):
        x_est = list(torch.split(x_est, x_true.shape[0], dim=0))
    m = len(x_est)
    x_est = [vectorize(x_est[i]).unsqueeze(1) for i in range(m)]
    x_est = torch.cat(x_est, dim=1)
        
    s1 = (vector_norm(x_est - x_true, 2, dim=2) + EPS).pow(beta).mean()
    s2 = (torch.cdist(x_est, x_est, 2) + EPS).pow(beta).mean() * m / (m - 1)
    if verbose:
        return torch.cat([(s1 - s2 / 2).reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return (s1 - s2 / 2)
    

def energy_loss_two_sample(x0, x, xp, x0p=None, beta=1, verbose=True, weights=None):
    """Loss function based on the energy score (estimated based on two samples).
    
    Args:
        x0 (torch.Tensor): an iid sample from the true distribution.
        x (torch.Tensor): an iid sample from the estimated distribution.
        xp (torch.Tensor): another iid sample from the estimated distribution.
        xp0 (torch.Tensor): another iid sample from the true distribution.
        beta (float): power parameter in the energy score.
        verbose (bool):  whether to return two terms of the loss.
    
    Returns:
        loss (torch.Tensor): energy loss.
    """
    EPS = 0 if float(beta).is_integer() else 1e-5
    x0 = vectorize(x0)
    x = vectorize(x)
    xp = vectorize(xp)
    if weights is None:
        weights = 1 / x0.size(0)
    if x0p is None:
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2 + ((vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) * weights).sum() / 2
        s2 = ((vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) * weights).sum() 
        loss = s1 - s2/2
    else:
        x0p = vectorize(x0p)
        s1 = ((vector_norm(x - x0, 2, dim=1) + EPS).pow(beta).sum() + (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta).sum() + 
              (vector_norm(x - x0p, 2, dim=1) + EPS).pow(beta).sum() + (vector_norm(xp - x0p, 2, dim=1) + EPS).pow(beta).sum()) / 4
        s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta).sum() 
        s3 = (vector_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta).sum() 
        loss = s1 - s2/2 - s3/2
    if verbose:
        return torch.cat([loss.reshape(1), s1.reshape(1), s2.reshape(1)], dim=0)
    else:
        return loss    


## Model: inner product layer + network layer      

def get_act_func(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid() 
    elif name == "tanh":
        return nn.Tanh() 
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "elu":
        return nn.ELU(inplace=True)
    else:
        return None

## Inner product layer
    
class InnerProductLayer(nn.Module):
    """A layer that computes an inner product with the input, where the first weight is fixed at 1, and the rest are learnable.

    Args:
        in_dim (int): input dimension.
        noise_std (float): standard deviation of the noise to be added to the input.
    """
    def __init__(self, in_dim, noise_std=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.noise_std = noise_std
        if in_dim < 1:
            raise ValueError("Input dimension must be at least 1")
        ## fix the first term at 1
        self.fixed_weight = torch.tensor(1.0)
        # The rest of the weights are learnable 
        if in_dim > 1:
            initial_values = a_true[1:]
            self.learnable_weights = nn.Parameter(initial_values)
        else:
            self.learnable_weights = None

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        
        noise = torch.randn_like(x) * self.noise_std
        #x_noisy = x + noise  # Shape: (batch_size, in_dim)
        
        fixed_weight = self.fixed_weight.to(device=device, dtype=dtype)
        x_fixed = x[:, 0] * fixed_weight 
        noise_fixed = noise[:, 0] * fixed_weight 
        
        if self.in_dim > 1:
            x_learnable = torch.matmul(x[:, 1:], self.learnable_weights)  
            x_reduced = x_fixed + x_learnable
            noise_learnable = torch.matmul(noise[:, 1:], self.learnable_weights)
            noise_reduced = noise_fixed + noise_learnable
        else:
            x_reduced = x_fixed 
            noise_reduced = noise_fixed 
            
            
        x_reduced = x_reduced.unsqueeze(-1)
        noise_reduced = noise_reduced.unsqueeze(-1)    
        
        x_reduced = torch.cat([x_reduced, noise_reduced], dim=1)
        
        return x_reduced
    
    def get_weight_vector(self):
        """Returns the full weight vector (including fixed and learnable weights)."""
        device = self.fixed_weight.device
        dtype = self.fixed_weight.dtype
        fixed_weight = self.fixed_weight.to(device=device, dtype=dtype).detach().cpu().numpy()

        if self.learnable_weights is not None:
            learnable_weights = self.learnable_weights.detach().cpu().numpy()
            weight_vector = np.concatenate(([fixed_weight], learnable_weights))
        else:
            weight_vector = np.array([fixed_weight])

        return weight_vector

## Network layer

class Net(nn.Module):
    """Deterministic neural network.

    Args:
        in_dim (int, optional): input dimension. Defaults to 1.
        out_dim (int, optional): output dimension. Defaults to 1.
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
    """
    def __init__(self, in_dim=1, out_dim=1, num_layer=3, hidden_dim=100, noise_dim=100,
                 add_bn=True, sigmoid=False, out_act=None, noise_std=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid
        self.noise_std = noise_std
        
        self.out_act = get_act_func(out_act)
        
        net = [nn.Linear(in_dim + noise_dim, hidden_dim)]
        if add_bn:
            net += [nn.BatchNorm1d(hidden_dim)]
        net += [nn.ReLU(inplace=True)]
        for _ in range(num_layer - 2):
            net += [nn.Linear(hidden_dim, hidden_dim)]
            if add_bn:
                net += [nn.BatchNorm1d(hidden_dim)]
            net += [nn.ReLU(inplace=True)]
        net.append(nn.Linear(hidden_dim, out_dim))
        if sigmoid:
            out_act = nn.Sigmoid() if out_dim == 1 else nn.Softmax(dim=1)
            net.append(out_act)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        device = x.device
        eps = torch.randn(x.size(0),self.noise_dim + self.in_dim - x.size(1),device=device) * self.noise_std
        out = torch.cat([x, eps], dim=1)
        out = self.net(out)
        if self.out_act is not None:
            out = self.out_act(out)
        return out
    

class MyNet(nn.Module):
    """Full neural network.

    Args:
        in_dim (int, optional): input dimension. 
        out_dim (int, optional): output dimension. 
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
    """
    def __init__(self, in_dim, out_dim, num_layer=3, hidden_dim=100, 
                 add_bn=True, sigmoid=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid   
        
        self.inner_product = InnerProductLayer(in_dim=self.in_dim)
        self.net_layer = Net(in_dim=2, out_dim=out_dim, num_layer=num_layer, hidden_dim=hidden_dim, add_bn=add_bn, sigmoid=sigmoid, out_act="relu")
    
    def forward(self,x):
        x = self.inner_product(x)
        x = self.net_layer(x)    
        return x
         
    
    def get_weight_vector(self):
        if hasattr(self, 'inner_product'):
            return self.inner_product.get_weight_vector()
        else:
            return None
        
    def sample(self, x, sample_size=512, expand_dim=True):
        """
        Sample new response data.

        Args:
            x (torch.Tensor): input data of predictors.
            sample_size (int, optional): number of samples to generate for each input. Defaults to 512.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor: A tensor of shape (data_size, out_dim, sample_size) if expand_dim is True,
                          otherwise (data_size * sample_size, out_dim).
        """
        data_size = x.size(0)  # Number of input samples
        #device = x.device

        # Generate noise for each sample
        x_samples = []
        for _ in range(sample_size):
            #noise = torch.randn_like(x) * self.inner_product.noise_std
            #x_noisy = x + noise
            x_noisy = x
            x_samples.append(x_noisy)
        
        # Concatenate all the noisy inputs
        x_rep = torch.cat(x_samples, dim=0)  # Shape: (data_size * sample_size, in_dim)
        
        # Pass through the network
        y_samples = self.forward(x_rep)  # Shape: (data_size * sample_size, out_dim)

        if not expand_dim:
            return y_samples
        else:
            # Reshape the output to include the sample dimension
            y_samples = y_samples.view(sample_size, data_size, -1)  # Shape: (sample_size, data_size, out_dim)
            y_samples = y_samples.permute(1, 2, 0)  # Shape: (data_size, out_dim, sample_size)
            return y_samples
    

    @torch.no_grad()
    def predict(self, x, target="mean", sample_size=512):
        """
        Point prediction.

        Args:
            x (torch.Tensor): input data of predictors.
            target (str or float or list, optional): a quantity of interest to predict. float refers to the quantiles. Defaults to "mean".
            sample_size (int, optional): number of samples to generate for each input. Defaults to 100.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions.
        """
        data_size = x.size(0)
        samples = self.sample(x, sample_size=sample_size, expand_dim=True)  # Shape: (data_size, out_dim, sample_size)

        if not isinstance(target, list):
            target = [target]

        results = []
        extremes = []
        for t in target:
            if t == "mean":
                results.append(samples.mean(dim=-1))  # Shape: (data_size, out_dim)
            else:
                if t == "median":
                    t = 0.5
                assert isinstance(t, float), "Target must be a float for quantile prediction."
                results.append(samples.quantile(t, dim=-1))  # Shape: (data_size, out_dim)
                if min(t, 1 - t) * sample_size < 10:
                    extremes.append(t)

        if len(extremes) > 0:
            print(f"Warning: the estimate for quantiles at {extremes} with a sample size of {sample_size} could be inaccurate. Please increase the sample_size.")

        if len(results) == 1:
            return results[0]
        else:
            return results 
    

## engression

def engression(x, y, 
               num_layer=3, hidden_dim=100, sigmoid=False,
               add_bn=True, beta=1,
               lr=0.0001, num_epochs=500, batch_size=None, 
               print_every_nepoch=100, print_times_per_epoch=1,
               device="cpu", standardize=True, verbose=True): 
    """This function fits an engression model to the data. It allows multivariate predictors and response variables. Variables are per default internally standardized (training with standardized data, while predictions and evaluations are on original scale).

    Args:
        x (torch.Tensor): training data of predictors.
        y (torch.Tensor): training data of responses.
        num_layer (int, optional): number of (linear) layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        beta (float, optional): power parameter in the energy loss.
        lr (float, optional): learning rate. Defaults to 0.0001.
        num_epochs (int, optional): number of epochs. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None.
        print_every_nepoch (int, optional): print losses every print_every_nepoch number of epochs. Defaults to 100.
        print_times_per_epoch (int, optional): print losses for print_times_per_epoch times per epoch. Defaults to 1.
        device (str, torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"].
        standardize (bool, optional):  whether to standardize data during training. Defaults to True.
        verbose (bool, optional): whether to print losses and info. Defaults to True.

    Returns:
        Engressor object: a fitted engression model.
    """
    if x.shape[0] != y.shape[0]:
        raise Exception("The sample sizes for the covariates and response do not match. Please check.")
    engressor = Engressor(in_dim=x.shape[1], out_dim=y.shape[1], 
                          num_layer=num_layer, hidden_dim=hidden_dim, 
                          sigmoid=sigmoid, add_bn=add_bn, beta=beta,
                          lr=lr, num_epochs=num_epochs, batch_size=batch_size, 
                          standardize=standardize, device=device, check_device=verbose, verbose=verbose)
    engressor.train(x, y, num_epochs=num_epochs, batch_size=batch_size, 
                    print_every_nepoch=print_every_nepoch, print_times_per_epoch=print_times_per_epoch, 
                    standardize=standardize, verbose=verbose)
    return engressor


class Engressor(object):
    """Engressor class.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        num_layer (int, optional): number of layers. Defaults to 2.
        hidden_dim (int, optional): number of neurons per layer. Defaults to 100.
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.
        add_bn (bool, optional): whether to add BN layer. Defaults to True.
        beta (float, optional): power parameter in the energy loss.
        lr (float, optional): learning rate. Defaults to 0.0001.
        num_epochs (int, optional): number of epochs. Defaults to 500.
        batch_size (int, optional): batch size. Defaults to None, referring to the full batch.
        standardize (bool, optional): whether to standardize data during training. Defaults to True.
        device (str or torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"].
        check_device (bool, optional): whether to check the device. Defaults to True.
    """
    def __init__(self, 
                 in_dim, out_dim, 
                 num_layer=3, hidden_dim=100, 
                 sigmoid=False, add_bn=True, beta=1,
                 lr=0.0001, num_epochs=500, batch_size=None, standardize=True, 
                 device="cpu", check_device=True, verbose=True): 
        super().__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.sigmoid = sigmoid
        self.add_bn = add_bn
        self.beta = beta
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        if isinstance(device, str):
            if device == "gpu" or device == "cuda":
                device = torch.device("cuda")
            else:
                device = torch.device(device)
        self.device = device
        if check_device:
            check_for_gpu(self.device)
        self.standardize = standardize
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        
        self.model = MyNet(in_dim, out_dim, num_layer, hidden_dim, add_bn, sigmoid).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.verbose = verbose
        
        self.tr_loss = None
            
    def train_mode(self):
        self.model.train()
        
    def eval_mode(self):
        self.model.eval()
        
    def summary(self):
        """Print the model architecture and hyperparameters."""
        print("Engression model with\n" +
              "\t number of layers: {}\n".format(self.num_layer) +
              "\t hidden dimensions: {}\n".format(self.hidden_dim) +
              "\t number of epochs: {}\n".format(self.num_epochs) +
              "\t batch size: {}\n".format(self.batch_size) +
              "\t learning rate: {}\n".format(self.lr) +
              "\t standardization: {}\n".format(self.standardize) +
              "\t training mode: {}\n".format(self.model.training) +
              "\t device: {}\n".format(self.device))
        print("Training loss (original scale):\n" +
              "\t energy-loss: {:.2f}, \n\tE(|Y-Yhat|): {:.2f}, \n\tE(|Yhat-Yhat'|): {:.2f}".format(
                  self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))
        
    def _standardize_data_and_record_stats(self, x, y):
        """Standardize the data and record the mean and standard deviation of the training data.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor): training data of responses.

        Returns:
            torch.Tensor: standardized data.
        """
        self.x_mean = torch.mean(x, dim=0)
        self.x_std = torch.std(x, dim=0)
        self.x_std[self.x_std == 0] += 1e-5
        
        self.y_mean = torch.zeros(y.shape[1:], device=y.device).unsqueeze(0)
        self.y_std = torch.ones(y.shape[1:], device=y.device).unsqueeze(0)
        
        x_standardized = (x - self.x_mean) / self.x_std
        y_standardized = (y - self.y_mean) / self.y_std
        self.x_mean = self.x_mean.to(self.device)
        self.x_std = self.x_std.to(self.device)
        self.y_mean = self.y_mean.to(self.device)
        self.y_std = self.y_std.to(self.device)
        return x_standardized, y_standardized

    def standardize_data(self, x, y=None):
        """Standardize the data, if self.standardize is True.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor, optional): training data of responses. Defaults to None.

        Returns:
            torch.Tensor: standardized or original data.
        """
        if y is None:
            if self.standardize:
                return (x - self.x_mean) / self.x_std
            else:
                return x
        else:
            if self.standardize:
                return (x - self.x_mean) / self.x_std, (y - self.y_mean) / self.y_std
            else:
                return x, y
    
    def unstandardize_data(self, y, x=None, expand_dim=False):
        """Transform the predictions back to the original scale, if self.standardize is True.

        Args:
            y (torch.Tensor): data in the standardized scale

        Returns:
            torch.Tensor: data in the original scale
        """
        if x is None:
            if self.standardize:
                if expand_dim:
                    return y * self.y_std.unsqueeze(0).unsqueeze(2) + self.y_mean.unsqueeze(0).unsqueeze(2)
                else:
                    return y * self.y_std + self.y_mean
            else:
                return y
        else:
            if self.standardize:
                return x * self.x_std + self.x_mean, y * self.y_std + self.y_mean
            else:
                return x, y
        
    def train(self, x, y, num_epochs=None, batch_size=None, lr=None, print_every_nepoch=100, print_times_per_epoch=1, standardize=None, verbose=True):
        """Fit the model.

        Args:
            x (torch.Tensor): training data of predictors.
            y (torch.Tensor): trainging data of responses.
            num_epochs (int, optional): number of training epochs. Defaults to None.
            batch_size (int, optional): batch size for mini-batch SGD. Defaults to None.
            lr (float, optional): learning rate.
            print_every_nepoch (int, optional): print losses every print_every_nepoch number of epochs. Defaults to 100.
            print_times_per_epoch (int, optional): print losses for print_times_per_epoch times per epoch. Defaults to 1.
            standardize (bool, optional): whether to standardize the data. Defaults to True.
            verbose (bool, optional): whether to print losses and info. Defaults to True.
        """
        self.train_mode()
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is None:
            batch_size = self.batch_size if self.batch_size is not None else x.size(0)
        if lr is not None:
            if lr != self.lr:
                self.lr = lr
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if standardize is not None:
            self.standardize = standardize
            
        x = vectorize(x)
        y = vectorize(y)
        if self.standardize:
            if verbose:
                print("Data is standardized for training only; the printed training losses are on the standardized scale. \n" +
                    "However during evaluation, the predictions, evaluation metrics, and plots will be on the original scale.\n")
            x, y = self._standardize_data_and_record_stats(x, y)
        x = x.to(self.device)
        y = y.to(self.device)
        
        if batch_size >= x.size(0)//2:
            if verbose:
                print("Batch is larger than half of the sample size. Training based on full-batch gradient descent.")
            self.batch_size = x.size(0)
            for epoch_idx in range(self.num_epochs):
                self.model.zero_grad()
                y_sample1 = self.model(x)
                y_sample2 = self.model(x)
                loss, loss1, loss2 = energy_loss_two_sample(y, y_sample1, y_sample2, beta=self.beta, verbose=True)
                loss.backward()
                self.optimizer.step()
                if (epoch_idx == 0 or  (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                    print("[Epoch {} ({:.0f}%)] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                        epoch_idx + 1, 100 * epoch_idx / num_epochs, loss.item(), loss1.item(), loss2.item()))
        else:
            train_loader = make_dataloader(x, y, batch_size=batch_size, shuffle=True)
            if verbose:
                print("Training based on mini-batch gradient descent with a batch size of {}.".format(batch_size))
            for epoch_idx in range(self.num_epochs):
                self.zero_loss()
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    self.train_one_iter(x_batch, y_batch)
                    if (epoch_idx == 0 or (epoch_idx + 1) % print_every_nepoch == 0) and verbose:
                        if (batch_idx + 1) % ((len(train_loader) - 1) // print_times_per_epoch) == 0:
                            self.print_loss(epoch_idx, batch_idx)

        # Evaluate performance on the training data (on the original scale)
        self.model.eval()
        x, y = self.unstandardize_data(y, x)
        self.tr_loss = self.eval_loss(x, y, loss_type="energy", verbose=True)
        
        if verbose:
            print("\nTraining loss on the original (non-standardized) scale:\n" +
                "\tEnergy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
                    self.tr_loss[0], self.tr_loss[1], self.tr_loss[2]))
            
        if verbose:
            print("\nPrediction-loss E(|Y-Yhat|) and variance-loss E(|Yhat-Yhat'|) should ideally be equally large" +
                "\n-- consider training for more epochs or adjusting hyperparameters if there is a mismatch ")
    
    def zero_loss(self):
        self.tr_loss = 0
        self.tr_loss1 = 0
        self.tr_loss2 = 0
    
    def train_one_iter(self, x_batch, y_batch):
        self.model.zero_grad()
        y_sample1 = self.model(x_batch)
        y_sample2 = self.model(x_batch)
        loss, loss1, loss2 = energy_loss_two_sample(y_batch, y_sample1, y_sample2, beta=self.beta, verbose=True)
        loss.backward()
        self.optimizer.step()
        self.tr_loss += loss.item()
        self.tr_loss1 += loss1.item()
        self.tr_loss2 += loss2.item()
        
    def print_loss(self, epoch_idx, batch_idx, return_loss=False):
        loss_str = "[Epoch {} ({:.0f}%), batch {}] energy-loss: {:.4f},  E(|Y-Yhat|): {:.4f},  E(|Yhat-Yhat'|): {:.4f}".format(
            epoch_idx + 1, 100 * epoch_idx / self.num_epochs, batch_idx + 1, 
            self.tr_loss / (batch_idx + 1), self.tr_loss1 / (batch_idx + 1), self.tr_loss2 / (batch_idx + 1))
        if return_loss:
            return loss_str
        else:
            print(loss_str)
    
    @torch.no_grad()
    def predict(self, x, target="mean", sample_size=512):
        """Point prediction. 

        Args:
            x (torch.Tensor): data of predictors.
            target (str or float or list, optional): a quantity of interest to predict. float refers to the quantiles. Defaults to "mean".
            sample_size (int, optional): generated sample sizes for each x. Defaults to 512.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions.
        """
        self.eval_mode()  
        x = vectorize(x)
        x = x.to(self.device)
        x = self.standardize_data(x)
        y_pred = self.model.predict(x, target, sample_size)
        if isinstance(y_pred, list):
            for i in range(len(y_pred)):
                y_pred[i] = self.unstandardize_data(y_pred[i])
        else:
            y_pred = self.unstandardize_data(y_pred)
        return y_pred
    
    @torch.no_grad()
    def sample(self, x, sample_size=512, expand_dim=True):
        """Sample new response data.

        Args:
            x (torch.Tensor): test data of predictors.
            sample_size (int, optional): generated sample sizes for each x. Defaults to 512.
            expand_dim (bool, optional): whether to expand the sample dimension. Defaults to True.

        Returns:
            torch.Tensor of shape (data_size, response_dim, sample_size).
                - [:,:,i] consists of the i-th sample of all x.
                - [i,:,:] consists of all samples of x_i.
        """
        self.eval_mode()
        x = vectorize(x)
        x = x.to(self.device)
        x = self.standardize_data(x)
        y_samples = self.model.sample(x, sample_size, expand_dim=expand_dim)            
        y_samples = self.unstandardize_data(y_samples, expand_dim=expand_dim)
        if sample_size == 1:
            y_samples = y_samples.squeeze(len(y_samples.shape) - 1)
        return y_samples
    
    
    @torch.no_grad()
    def eval_loss(self, x, y, loss_type="l2", sample_size=None, beta=1, verbose=False):
        """Compute the loss for evaluation.

        Args:
            x (torch.Tensor): data of predictors.
            y (torch.Tensor): data of responses.
            loss_type (str, optional): loss type. Defaults to "l2". Choices: ["l2", "l1", "energy", "cor"].
            sample_size (int, optional): generated sample sizes for each x. Defaults to 100.
            beta (float, optional): beta in energy score. Defaults to 1.
        
        Returns:
            float: evaluation loss.
        """
        if sample_size is None:
            sample_size = 2 if loss_type == "energy" else 100
        self.eval_mode()
        x = vectorize(x)
        y = vectorize(y)
        x = x.to(self.device)
        y = y.to(self.device)
        if loss_type == "l2":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = (y - y_pred).pow(2).mean()
        elif loss_type == "cor":
            y_pred = self.predict(x, target="mean", sample_size=sample_size)
            loss = cor(y, y_pred)
        elif loss_type == "l1":
            y_pred = self.predict(x, target=0.5, sample_size=sample_size)
            loss = (y - y_pred).abs().mean()
        else:
            assert loss_type == "energy"
            y_samples = self.sample(x, sample_size=sample_size, expand_dim=False)
            loss = energy_loss(y, y_samples, beta=beta, verbose=verbose)
        if not verbose:
            return loss.item()
        else:
            loss, loss1, loss2 = loss
            return loss.item(), loss1.item(), loss2.item()
        
    @torch.no_grad()      
    def get_weight_vector(self):
        """Returns the learned weight vector from the model."""
        return self.model.get_weight_vector() 
    
    @torch.no_grad()
    def predict_g(self, x, custom_weight_vector=a_true[1:], target="mean", sample_size=512):
        """Point prediction using a customized weight vector in the inner product layer.

        Args:
            x (torch.Tensor): data of predictors.
            custom_weight_vector (torch.Tensor): custom weight vector for the inner product layer.
            target (str or float or list, optional): a quantity of interest to predict. float refers to the quantiles. Defaults to "mean".
            sample_size (int, optional): generated sample sizes for each x. Defaults to 512.

        Returns:
            torch.Tensor or list of torch.Tensor: point predictions.
        """
        self.eval_mode()
        x = vectorize(x)
        x = x.to(self.device)
        x = self.standardize_data(x)
        
        # Temporarily replace the learnable weight vector with the custom one
        original_weight = self.model.inner_product.learnable_weights.data.clone()  # Save original weights
        self.model.inner_product.learnable_weights.data = custom_weight_vector.to(self.device)

        y_pred = self.model.predict(x, target, sample_size)

        # Restore original weights
        self.model.inner_product.learnable_weights.data = original_weight

        if isinstance(y_pred, list):
            for i in range(len(y_pred)):
                y_pred[i] = self.unstandardize_data(y_pred[i])
        else:
            y_pred = self.unstandardize_data(y_pred)

        return y_pred 
    
    
## conduct simulation study


def run_engression_simulations(
    N_sim, 
    x_min=0, x_max=4, num_points=1000, 
    n_train=10000, x_lower=0, x_upper=2, noise_std=1, 
    lr=0.01, num_epochs=500, batch_size=5000,
    true_function="softplus", noise_dist="gaussian", noise_corr=0,
    sample_size=512,device='cpu',sigmoid=False
):
    """
    Run multiple Engression simulations and store results.

    Args:
        N_sim (int): Number of simulations to run.
        x_min (float): Minimum value of x for the grid.
        x_max (float): Maximum value of x for the grid.
        num_points (int): Number of grid points for prediction.
        n_train (int): Number of training samples.
        x_lower (float): Lower bound for x in training data.
        x_upper (float): Upper bound for x in training data.
        noise_std (float): Standard deviation of the noise in data.
        lr (float): Learning rate for training.
        num_epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        true_function (str or callable): The true function used in the simulator.
        noise_dist (str): The noise distribution ("gaussian" or "uniform").
        noise_corr (float): Pairwise correlation between noise components.
        sample_size (int): generated sample sizes for each x. Defaults to 512.
        device (str or torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"]. 
        sigmoid (bool, optional): whether to add sigmoid or softmax at the end. Defaults to False.

    Returns:
        dict: A dictionary containing predicted means, medians, L2 errors, and energy losses.
    """
    # Arrays to store results
    predicted_means = []
    predicted_medians = []
    predicted_gs_mean = []
    predicted_gs_median = []
    L2_errors = []
    L1_errors = []
    weight_estimates = []
    
    # Set the seed
    torch_seed = 42
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    
    
    x_eval, y_eval_med, y_eval_mean = preanm_simulator_64d(true_function=true_function, n=num_points, 
                                                           x_lower=x_min, x_upper=x_max, noise_std=noise_std, noise_dist=noise_dist, 
                                                           train=False, noise_corr=noise_corr,device=device)
    
    y_true_med = y_eval_med.cpu().numpy()
    y_true_mean = y_eval_mean.cpu().numpy()
    
    
    a1 = a_true
    x_test_1d = x_eval @ a1.unsqueeze(1)
    
    
    
    # Run simulations
    for sim in range(N_sim):
        
        print(f"\nRunning simulation {sim + 1}/{N_sim}")
        
        torch_seed = sim
        torch.manual_seed(torch_seed)
        np.random.seed(torch_seed)
    
        
        # Generate training data
        x_train, y_train = preanm_simulator_64d(
            true_function=true_function,       
            n=n_train, x_lower=x_lower, x_upper=x_upper, 
            noise_std=noise_std, noise_dist=noise_dist, 
            train=True, noise_corr=noise_corr, device=device 
        )
    
        
        # Instantiate and train the Engression model
        engressor = engression(
            x_train, y_train, lr=lr, num_epochs=num_epochs, 
            batch_size=batch_size, device=device, sigmoid=sigmoid
        )
        
        learned_weight_vector = engressor.get_weight_vector()
        learned_weight_vector_tensor = torch.from_numpy(learned_weight_vector)
        print("nonzero weight values:",learned_weight_vector_tensor[[0,30,58]])
        diff = learned_weight_vector_tensor - a1
        disc_diff = torch.norm(diff, p=2)
        disc_diff = np.float32(disc_diff.item())
        print(f"Difference between true weight vector and learned weight vector: {disc_diff:.4f}")      

        # Move x_grid to the appropriate device
        # x_grid_device = x_grid.to(engressor.device)
        x_eval = x_eval.to(engressor.device)

        # Predict mean and median
        y_pred_mean = engressor.predict(x_eval, target="mean", sample_size=sample_size).cpu().numpy()
        y_pred_median = engressor.predict(x_eval, target="median", sample_size=sample_size).cpu().numpy()
        g_pred_mean = engressor.predict_g(x_eval, target="mean", sample_size=sample_size).cpu().numpy()
        g_pred_median = engressor.predict_g(x_eval, target="median", sample_size=sample_size).cpu().numpy()

        # Evaluate L2 error and energy loss
        L2_error = engressor.eval_loss(
            x_eval.to(engressor.device), 
            torch.from_numpy(y_true_mean).to(engressor.device), 
            loss_type="l2"
        )
        L1_error = engressor.eval_loss(
            x_eval.to(engressor.device), 
            torch.from_numpy(y_true_med).to(engressor.device), 
            loss_type="l1"
        )
        
        # Store results
        predicted_means.append(y_pred_mean)
        predicted_medians.append(y_pred_median)
        predicted_gs_mean.append(g_pred_mean)
        predicted_gs_median.append(g_pred_median)
        L2_errors.append(L2_error)
        L1_errors.append(L1_error)
        weight_estimates.append(learned_weight_vector)
        

    # Convert lists to NumPy arrays
    predicted_means = np.array(predicted_means) 
    predicted_medians = np.array(predicted_medians)  
    L2_errors = np.array(L2_errors) 
    L1_errors = np.array(L1_errors) 
    predicted_gs_mean = np.array(predicted_gs_mean)
    predicted_gs_median = np.array(predicted_gs_median)
    weight_estimates = np.array(weight_estimates)
    
    x_train_example = x_train @ a1.unsqueeze(1)
    y_train_example = y_train

    # Return results in a dictionary
    results = {
        'predicted_means': predicted_means,
        'predicted_medians': predicted_medians,
        'predicted_funs_mean': predicted_gs_mean,
        'predicted_funs_median': predicted_gs_median,
        'L2_errors': L2_errors,
        'L1_errors': L1_errors,
        'weight_estimates': weight_estimates,
        # Include for plotting
        'x_train': x_train_example, 
        'y_train': y_train_example,
        'x_test': x_test_1d,
        'y_true_mean': y_true_mean,
        'y_true_med': y_true_med
    }
       
    return results


    