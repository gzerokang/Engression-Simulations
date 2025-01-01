import torch
import numpy as np
import torch.nn as nn
import os
from torch.linalg import vector_norm
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import norm
import torch.optim as optim


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
            

class InnerProductLayer(nn.Module):
    """A layer that computes an inner product with the input, where the first weight is fixed at 1, and the rest are learnable.

    Args:
        in_dim (int): input dimension
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
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
        fixed_weight = self.fixed_weight.to(device=device, dtype=dtype)
        x_fixed = x[:, 0] * fixed_weight  
        if self.in_dim > 1:
            x_learnable = torch.matmul(x[:, 1:], self.learnable_weights)  
            x_reduced = x_fixed + x_learnable
        else:
            x_reduced = x_fixed  
        x_reduced = x_reduced.unsqueeze(1) 
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
    def __init__(self, in_dim=1, out_dim=1, num_layer=3, hidden_dim=100, 
                 add_bn=True, sigmoid=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.add_bn = add_bn
        self.sigmoid = sigmoid
        
        net = [nn.Linear(in_dim, hidden_dim)]
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
        return self.net(x)


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
        self.net_layer = Net(in_dim=1, out_dim=out_dim, hidden_dim=hidden_dim, add_bn=add_bn, sigmoid=sigmoid)
    
    def forward(self,x):
        x = self.inner_product(x)
        x = self.net_layer(x)    
        return x
         
    
    def get_weight_vector(self):
        if hasattr(self, 'inner_product'):
            return self.inner_product.get_weight_vector()
        else:
            return None
        


def train_model(model, X_train, y_train, num_epochs=100, batch_size=32, learning_rate=1e-3, device=torch.device('cpu')):
    # Create DataLoader
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to device
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(dataloader.dataset)
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')

def predict_model(model, X_test, device=torch.device('cpu')):
    model.eval()
    X_test = X_test.to(device)
    with torch.no_grad():
        outputs = model(X_test)
    return outputs.cpu()

def predict_g(model, X_test, device=torch.device('cpu')):
    model.eval()
    X_test = X_test.to(device)
    
    # Temporarily set the learnable weights to (1.2, 1.5)
    fixed_learnable_weights = a_true[1:].to(device)
    
    with torch.no_grad():
       
        original_weights = model.inner_product.learnable_weights.clone().detach()
        model.inner_product.learnable_weights.data = fixed_learnable_weights
        outputs = model(X_test)
        model.inner_product.learnable_weights.data = original_weights

    return outputs.cpu()



def run_compare_simulations(
    N_sim, 
    x_min=0, x_max=4, num_points=1000, 
    n_train=10000, x_lower=0, x_upper=2, noise_std=1, 
    lr=0.01, num_epochs=100, batch_size=5000,
    true_function="softplus", noise_dist="gaussian", noise_corr=0, device="cpu"
):
    """
    Run multiple L2 simulations and store results.

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
        device (str, torch.device, optional): device. Defaults to "cpu". Choices = ["cpu", "gpu", "cuda"].

    Returns:
        dict: A dictionary containing predicted means and L2 errors.
    """
    # Arrays to store results
    predicted_means = []
    L2_errors = []
    weight_estimates = []
    predicted_gs = []

    # Set the seed
    torch_seed = 42
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    
    
    ## truth for comparison
    x_eval, y_eval_med, y_eval_mean = preanm_simulator_64d(true_function=true_function, n=num_points, 
                                                           x_lower=x_min, x_upper=x_max, noise_std=noise_std, noise_dist=noise_dist, 
                                                           train=False, noise_corr=noise_corr, device=device)
    
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
    
        
        model = MyNet(
        in_dim=64,
        out_dim=1,
        num_layer=3,  
        hidden_dim=100,
        add_bn=True,
        sigmoid=False
        )
        
        train_model(
        model=model,
        X_train=x_train,
        y_train=y_train,
        num_epochs=num_epochs,   
        batch_size=batch_size,
        learning_rate=lr,
        device=device
        )
        
        y_pred_mean = predict_model(model, x_eval, device)  ## using beta_hat
        g_pred = predict_g(model, x_eval, device)  ## using true beta
        
        mse_loss = nn.MSELoss()
        L2_error = mse_loss(y_pred_mean, y_eval_mean)
        
        # Retrieve the weight vector
        weight_vector = model.get_weight_vector()
        learned_weight_vector_tensor = torch.from_numpy(weight_vector)
        print("nonzero weight values:",learned_weight_vector_tensor[[0,30,58]])
        diff = learned_weight_vector_tensor - a1
        disc_diff = torch.norm(diff, p=2)
        disc_diff = np.float32(disc_diff.item())
        print(f"Difference between true weight vector and learned weight vector: {disc_diff:.8f}")      

        
        
        # Store results
        predicted_means.append(y_pred_mean)
        L2_errors.append(L2_error)
        weight_estimates.append(weight_vector)
        predicted_gs.append(g_pred)
        
    # Convert lists to NumPy arrays
    predicted_means = np.array(predicted_means) 
    L2_errors = np.array(L2_errors) 
    weight_estimates = np.array(weight_estimates)
    predicted_gs = np.array(predicted_gs)
    
    x_train_example = x_train @ a1.unsqueeze(1)
    y_train_example = y_train

    # Return results in a dictionary
    results = {
        'predicted_means': predicted_means,
        'predicted_funs': predicted_gs,
        'L2_errors': L2_errors,
        'weight_estimates': weight_estimates,
        # Include for plotting
        'x_train': x_train_example, 
        'y_train': y_train_example,
        'x_test': x_test_1d,
        'y_true_mean': y_true_mean
    }

    return results

