import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

def create_grid(x_range, y_range, n=100):
    x = torch.linspace(x_range[0], x_range[1], n)
    y = torch.linspace(y_range[0], y_range[1], n)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    pos = torch.stack((X, Y), dim=-1)
    return X, Y, pos

def gaussian_mixture(pos, weights, means, covs):
    z = torch.zeros_like(pos[:,:,0])
    for w, m, c in zip(weights, means, covs):
        mvn = MultivariateNormal(m, c)
        z += w * torch.exp(mvn.log_prob(pos))
    return z

def plot_distribution(X, Y, Z, title):
    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=20, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.title(title)
    plt.xlabel('x₀₁')
    plt.ylabel('x₀₂')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def calculate_posterior(y, weights, means, covs, cov_y):
    new_weights = []
    new_means = []
    new_covs = []
    
    for w, m, c in zip(weights, means, covs):
        S = c + cov_y
        K = c @ torch.inverse(S)
        new_mean = m + K @ (y - m)
        new_cov = c - K @ c
        mvn = MultivariateNormal(m, S)
        new_weight = w * torch.exp(mvn.log_prob(y))
        
        new_weights.append(new_weight)
        new_means.append(new_mean)
        new_covs.append(new_cov)
    
    new_weights = torch.tensor(new_weights)
    new_weights /= new_weights.sum()
    return new_weights, new_means, new_covs

def make_positive_semidefinite(A, epsilon=1e-6):
    """
    Make a symmetric matrix positive semidefinite by adding a small value to the diagonal if needed.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    if torch.min(eigenvalues) < 0:
        A += (torch.abs(torch.min(eigenvalues)) + epsilon) * torch.eye(A.shape[0])
    return A

def create_grid(x_range, y_range, n=100):
    x = torch.linspace(x_range[0], x_range[1], n)
    y = torch.linspace(y_range[0], y_range[1], n)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    pos = torch.stack((X, Y), dim=-1)
    return X, Y, pos

def gaussian_mixture(pos, weights, means, covs):
    z = torch.zeros_like(pos[:,:,0])
    for w, m, c in zip(weights, means, covs):
        mvn = MultivariateNormal(m, c)
        z += w * torch.exp(mvn.log_prob(pos))
    return z

def plot_distribution(X, Y, Z, title, ax):
    im = ax.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=20, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('x₀₁')
    ax.set_ylabel('x₀₂')
    ax.axis('equal')
    return im

def calculate_posterior(y, weights, means, covs, cov_y):
    new_weights = []
    new_means = []
    new_covs = []
    
    for w, m, c in zip(weights, means, covs):
        S = make_positive_semidefinite(c + cov_y)
        K = c @ torch.inverse(S)
        new_mean = m + K @ (y - m)
        new_cov = make_positive_semidefinite(c - K @ c)
        mvn = MultivariateNormal(m, S)
        new_weight = w * torch.exp(mvn.log_prob(y))
        
        new_weights.append(new_weight)
        new_means.append(new_mean)
        new_covs.append(new_cov)
    
    new_weights = torch.tensor(new_weights) / torch.sum(torch.tensor(new_weights))
    return new_weights, new_means, new_covs

def calculate_posterior_xt(x_t, weights, means, covs, cov_xt):
    new_weights = []
    new_means = []
    new_covs = []
    
    for w, m, c in zip(weights, means, covs):
        S = make_positive_semidefinite(c + cov_xt)
        K = c @ torch.inverse(S)
        # print(m.shape, K.shape, x_t.shape)
        new_mean = m + K @ (x_t.squeeze() - m)
        new_cov = make_positive_semidefinite(c - K @ c)
        mvn = MultivariateNormal(m, S)
        new_weight = w * torch.exp(mvn.log_prob(x_t))
        
        new_weights.append(new_weight)
        new_means.append(new_mean)
        new_covs.append(new_cov)
    
    new_weights = torch.tensor(new_weights) / torch.sum(torch.tensor(new_weights))
    return new_weights, new_means, new_covs

def calculate_posterior_xt_y(x_t, y, weights, means, covs, cov_xt, cov_y):
    new_weights = []
    new_means = []
    new_covs = []
    
    for w, m, c in zip(weights, means, covs):
        # Combine information from x_t and y
        combined_obs = torch.cat([x_t, y])
        combined_mean = torch.cat([m, m])
        
        # Calculate individual likelihoods
        mvn_xt = MultivariateNormal(m, cov_xt + c)
        mvn_y = MultivariateNormal(m, cov_y + c)
        likelihood_xt = torch.exp(mvn_xt.log_prob(x_t))
        likelihood_y = torch.exp(mvn_y.log_prob(y))
        
        # Update mean using both observations
        K_xt = c @ torch.inverse(c + cov_xt)
        K_y = c @ torch.inverse(c + cov_y)
        new_mean = m + K_xt @ (x_t - m) + K_y @ (y - m)
        
        # Update covariance
        new_cov = torch.inverse(torch.inverse(c) + torch.inverse(cov_xt) + torch.inverse(cov_y))
        
        # Add a small amount of noise to prevent overconfidence
        new_cov += 1e-3 * torch.eye(new_cov.shape[0])
        
        # Ensure positive semidefiniteness
        new_cov = make_positive_semidefinite(new_cov)
        
        # Calculate new weight
        new_weight = w * likelihood_xt * likelihood_y
        
        new_weights.append(new_weight)
        new_means.append(new_mean)
        new_covs.append(new_cov)
    
    new_weights = torch.tensor(new_weights) / torch.sum(torch.tensor(new_weights))
    return new_weights, new_means, new_covs


def gaussian_mixture_log_gradient(x, weights, data_means, data_covs, diffusion_sigma):
    """
    Calculate the gradient of the log PDF for a Gaussian mixture model.
    
    Args:
    x: (batch_size, d) - Input points
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    
    Returns:
    grads: (batch_size, d) - Gradients of the log PDF
    """
    batch_size, d = x.shape
    n = len(data_means)
    
    # Combine covariances
    combined_covs = data_covs + diffusion_sigma**2 * torch.eye(d).unsqueeze(0).repeat(n, 1, 1)
    
    # Calculate precision matrices (inverse of covariance matrices)
    precisions = torch.inverse(combined_covs)
    
    # Expand dimensions for broadcasting
    x_expanded = x.unsqueeze(1)  # (batch_size, 1, d)
    means_expanded = data_means.unsqueeze(0)  # (1, n, d)
    
    # Calculate differences (mean - x instead of x - mean)
    diff = means_expanded - x_expanded  # (batch_size, n, d)
    
    # Calculate log probabilities (without constants as they'll cancel out in normalization)
    log_probs = -0.5 * torch.sum(diff.unsqueeze(-2) @ precisions.unsqueeze(0) @ diff.unsqueeze(-1), dim=(-2, -1))  # (batch_size, n)
    log_probs += torch.log(weights).unsqueeze(0)  # Add log weights
    
    # Calculate responsibilities (normalized probabilities)
    log_resp = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)  # (batch_size, n)
    resp = torch.exp(log_resp)  # (batch_size, n)
    
    # Calculate gradients
    grads = torch.sum(resp.unsqueeze(-1) * (precisions.unsqueeze(0) @ diff.unsqueeze(-1)).squeeze(-1), dim=1)
    
    return grads

def gaussian_mixture_posterior_mean(x, weights, data_means, data_covs, diffusion_sigma):
    """
    Calculate the posterior mean E[x_0|x_t] for a Gaussian mixture model.
    
    Args:
    x: (batch_size, d) - Input points (x_t)
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    
    Returns:
    posterior_mean: (batch_size, d) - Posterior mean E[x_0|x_t]
    """
    # Calculate the gradient of the log PDF
    grad_log_pdf = gaussian_mixture_log_gradient(x, weights, data_means, data_covs, diffusion_sigma)
    
    # Calculate the posterior mean
    posterior_mean = x + diffusion_sigma**2 * grad_log_pdf
    
    return posterior_mean

def gaussian_mixture_log_hessian(x, weights, data_means, data_covs, diffusion_sigma):
    # TODO: THIS IS NOT CORRECT
    """
    Calculate the Hessian of the log PDF for a Gaussian mixture model.
    This is equivalent to the Jacobian of the gradient of the log PDF.
    
    Args:
    x: (batch_size, d) - Input points
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    
    Returns:
    hessian: (batch_size, d, d) - Hessian matrices
    """
    batch_size, d = x.shape
    n = len(data_means)
    
    # Combine covariances
    combined_covs = data_covs + diffusion_sigma**2 * torch.eye(d).unsqueeze(0).repeat(n, 1, 1)
    
    # Calculate precision matrices (inverse of covariance matrices)
    precisions = torch.inverse(combined_covs)
    
    # Expand dimensions for broadcasting
    x_expanded = x.unsqueeze(1)  # (batch_size, 1, d)
    means_expanded = data_means.unsqueeze(0)  # (1, n, d)
    
    # Calculate differences
    diff = means_expanded - x_expanded  # (batch_size, n, d)
    
    # Calculate log probabilities (without constants as they'll cancel out in normalization)
    log_probs = -0.5 * torch.sum(diff.unsqueeze(-2) @ precisions.unsqueeze(0) @ diff.unsqueeze(-1), dim=(-2, -1))  # (batch_size, n)
    log_probs += torch.log(weights).unsqueeze(0)  # Add log weights
    
    # Calculate responsibilities (normalized probabilities)
    log_resp = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)  # (batch_size, n)
    resp = torch.exp(log_resp)  # (batch_size, n)
    
    # Calculate the first term of the Hessian
    first_term = torch.sum(resp.unsqueeze(-1).unsqueeze(-1) * precisions.unsqueeze(0), dim=1)  # (batch_size, d, d)
    
    # Calculate the second term of the Hessian
    grad = torch.sum(resp.unsqueeze(-1) * (precisions.unsqueeze(0) @ diff.unsqueeze(-1)).squeeze(-1), dim=1)  # (batch_size, d)
    second_term = grad.unsqueeze(-1) @ grad.unsqueeze(-2)  # (batch_size, d, d)
    
    # Combine the terms to get the Hessian
    hessian = first_term - second_term
    
    return hessian

def gaussian_mixture_log_hessian_autograd(x, weights, data_means, data_covs, diffusion_sigma):
    """
    Calculate the Hessian of the log PDF for a Gaussian mixture model using autograd.
    This is equivalent to the Jacobian of the gradient of the log PDF.
    
    Args:
    x: (batch_size, d) - Input points
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    
    Returns:
    hessian: (batch_size, d, d) - Hessian matrices
    """
    batch_size, d = x.shape
    
    def gradient_function(x_i):
        return gaussian_mixture_log_gradient(x_i, weights, data_means, data_covs, diffusion_sigma)#.squeeze(0)
    
    hessian = torch.zeros(batch_size, d, d, device=x.device, dtype=x.dtype)
    
    x.requires_grad_(True)
    
    for i in range(d):
        grad_i = torch.autograd.grad(gradient_function(x), x, grad_outputs=torch.eye(d)[i].expand(batch_size, d).to(x), create_graph=True)[0]
        hessian[:, i, :] = grad_i
    
    return hessian

def gaussian_mixture_log_hessian_autograd_test():
    weights = torch.tensor([0.5, 0.5])
    data_means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    data_covs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    diffusion_sigma = 1.0
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    hessian = gaussian_mixture_log_hessian_autograd(x, weights, data_means, data_covs, diffusion_sigma)
    print(hessian)

def gaussian_mixture_denoiser_covariance(x, weights, data_means, data_covs, diffusion_sigma, use_autograd=True):
    """
    Calculate the denoiser covariance for a Gaussian mixture model.
    """
    if use_autograd:
        hessian = gaussian_mixture_log_hessian_autograd(x, weights, data_means, data_covs, diffusion_sigma)
    else:
        hessian = gaussian_mixture_log_hessian(x, weights, data_means, data_covs, diffusion_sigma)
    d = hessian.shape[1]
    denoiser_covariance = (torch.eye(d) + diffusion_sigma**2 * hessian) * diffusion_sigma**2
    return denoiser_covariance

def gaussian_mixture_log_pdf(x, weights, data_means, data_covs, diffusion_sigma):
    """
    Calculate the log PDF for a Gaussian mixture model.
    
    Args:
    x: (batch_size, d) - Input points
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    
    Returns:
    log_pdf: (batch_size,) - Log PDF values
    """
    batch_size, d = x.shape
    n = data_means.shape[0]
    
    # Combine covariances
    combined_covs = data_covs + diffusion_sigma**2 * torch.eye(d).unsqueeze(0).repeat(n, 1, 1)
    
    # Calculate precision matrices (inverse of covariance matrices)
    precisions = torch.inverse(combined_covs)
    
    # Calculate log determinants
    log_dets = torch.logdet(combined_covs)
    
    # Expand dimensions for broadcasting
    x_expanded = x.unsqueeze(1)  # (batch_size, 1, d)
    means_expanded = data_means.unsqueeze(0)  # (1, n, d)
    
    # Calculate Mahalanobis distances
    diff = x_expanded - means_expanded  # (batch_size, n, d)
    mahalanobis = torch.sum(diff.unsqueeze(-2) @ precisions.unsqueeze(0) @ diff.unsqueeze(-1), dim=(-2, -1))  # (batch_size, n)
    
    # Calculate log probabilities
    log_probs = -0.5 * (d * np.log(2 * np.pi) + log_dets + mahalanobis)  # (batch_size, n)
    log_probs += torch.log(weights).unsqueeze(0)  # Add log weights
    
    # Compute log PDF using log-sum-exp trick for numerical stability
    log_pdf = torch.logsumexp(log_probs, dim=1)
    
    return log_pdf

def visualize_gaussian_mixture(weights, data_means, data_covs, diffusion_sigma, grid_size=50, plot_range=(-4, 4)):
    # Create a grid of points
    x = torch.linspace(plot_range[0], plot_range[1], grid_size)
    y = torch.linspace(plot_range[0], plot_range[1], grid_size)
    xx, yy = torch.meshgrid(x, y)
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Calculate log PDF and gradients
    log_pdf = gaussian_mixture_log_pdf(grid_points, weights, data_means, data_covs, diffusion_sigma)
    gradients = gaussian_mixture_log_gradient(grid_points, weights, data_means, data_covs, diffusion_sigma)

    # Reshape for plotting
    log_pdf = log_pdf.reshape(grid_size, grid_size).detach().numpy()
    gradients = gradients.reshape(grid_size, grid_size, 2).detach().numpy()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot log PDF contours
    contour = ax.contourf(xx, yy, log_pdf, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Log PDF')

    # Plot gradient quiver
    skip = (slice(None, None, 3), slice(None, None, 3))
    quiver = ax.quiver(xx[skip], yy[skip], gradients[skip[0], skip[1], 0], gradients[skip[0], skip[1], 1],
                       scale=50, color='red', alpha=0.7)
    

    # Plot the means of the Gaussian components
    # ax.scatter(data_means[:, 0], data_means[:, 1], c='red', s=100, label='Component Means')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Gaussian Mixture Log PDF and Gradient (diffusion_sigma={diffusion_sigma})')
    ax.legend()

    plt.show()
    
def visualize_gaussian_mixture(weights, data_means, data_covs, diffusion_sigma, grid_size=50, plot_range=(-4, 4)):
    # Create a grid of points
    x = torch.linspace(plot_range[0], plot_range[1], grid_size)
    y = torch.linspace(plot_range[0], plot_range[1], grid_size)
    xx, yy = torch.meshgrid(x, y)
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Calculate log PDF and gradients
    log_pdf = gaussian_mixture_log_pdf(grid_points, weights, data_means, data_covs, diffusion_sigma)
    gradients = gaussian_mixture_log_gradient(grid_points, weights, data_means, data_covs, diffusion_sigma)

    # Reshape for plotting
    log_pdf = log_pdf.reshape(grid_size, grid_size).detach().numpy()
    gradients = gradients.reshape(grid_size, grid_size, 2).detach().numpy()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot log PDF contours
    contour = ax.contourf(xx, yy, log_pdf, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Log PDF')

    # Plot gradient quiver
    skip = (slice(None, None, 3), slice(None, None, 3))
    quiver = ax.quiver(xx[skip], yy[skip], gradients[skip[0], skip[1], 0], gradients[skip[0], skip[1], 1],
                       scale=50, color='red', alpha=0.7)

    # Plot the means of the Gaussian components
    scatter = ax.scatter(data_means[:, 0], data_means[:, 1], c='white', edgecolor='black', s=100, label='Component Means')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Gaussian Mixture Log PDF and Gradient (σ={diffusion_sigma:.2f})')
    ax.legend()

    return fig, ax, contour, quiver, scatter, xx, yy

def animate_gaussian_mixture(weights, data_means, data_covs, max_sigma=2.0, num_frames=50, grid_size=50, plot_range=(-4, 4)):
    # Create a grid of points
    x = torch.linspace(plot_range[0], plot_range[1], grid_size)
    y = torch.linspace(plot_range[0], plot_range[1], grid_size)
    xx, yy = torch.meshgrid(x, y)
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))

    def update_wrapper(frame):
        return update(frame, grid_points, weights, data_means, data_covs, max_sigma, num_frames, xx, yy, grid_size, ax)

    anim = FuncAnimation(fig, update_wrapper, frames=num_frames, interval=100, blit=False)
    
    # Display the animation in the notebook
    return HTML(anim.to_jshtml())

def update_noised_gmm_plot(frame, grid_points, weights, data_means, data_covs, max_sigma, num_frames, xx, yy, grid_size, ax):
    diffusion_sigma = frame * max_sigma / (num_frames - 1)
    
    # Calculate log PDF and gradients
    log_pdf = gaussian_mixture_log_pdf(grid_points, weights, data_means, data_covs, diffusion_sigma)
    gradients = gaussian_mixture_log_gradient(grid_points, weights, data_means, data_covs, diffusion_sigma)

    # Reshape for plotting
    log_pdf = log_pdf.reshape(grid_size, grid_size).detach().numpy()
    gradients = gradients.reshape(grid_size, grid_size, 2).detach().numpy()

    # Clear all previous elements
    ax.clear()

    # Redraw the plot
    new_contour = ax.contourf(xx, yy, log_pdf, levels=20, cmap='viridis', alpha=0.8)
    
    skip = (slice(None, None, 3), slice(None, None, 3))
    new_quiver = ax.quiver(xx[skip], yy[skip], gradients[skip[0], skip[1], 0], gradients[skip[0], skip[1], 1],
                           scale=50, color='red', alpha=0.7)
    
    # new_scatter = ax.scatter(data_means[:, 0], data_means[:, 1], c='white', edgecolor='black', s=100, label='Component Means')

    # Reset labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Gaussian Mixture Log PDF and Gradient (σ={diffusion_sigma:.2f})')
    ax.legend()

    return new_contour.collections + [new_quiver, new_scatter]

def update_noised_gmm_surface_plot(frame, grid_points, weights, data_means, data_covs, max_sigma, num_frames, xx, yy, grid_size, ax, log=True):
    diffusion_sigma = frame * max_sigma / (num_frames - 1)
    # Calculate log PDF
    log_pdf = gaussian_mixture_log_pdf(grid_points, weights, data_means, data_covs, diffusion_sigma)
    # Reshape for plotting
    log_pdf = log_pdf.reshape(grid_size, grid_size).detach().numpy()
    if not log:
        log_pdf = np.exp(log_pdf)
    # Clear all previous elements
    ax.clear()
    
    if log:
        ax.set_zlim(-10, 0)
        log_pdf[log_pdf < -10] = -10
    
    # Redraw the plot
    new_surface = ax.plot_surface(xx, yy, log_pdf, cmap='viridis')
    ax.set_zlabel('Log PDF')
    ax.view_init(elev=40, azim=45)  # Set the viewing angle
    
    # Reset labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if log:
        ax.set_title(f'Log PDF during forward process (σ={diffusion_sigma:.2f})', fontsize=20)
    else:
        ax.set_title(f'PDF during forward process (σ={diffusion_sigma:.2f})', fontsize=20)
    ax.legend()

    return new_surface, xx, yy, log_pdf
    

# Diffusion schedule
def linear_sigma(t, min_sigma, max_sigma):
    return min_sigma + (max_sigma - min_sigma) * t

def linear_sigma_derivative(t, min_sigma, max_sigma):
    return (max_sigma - min_sigma)

# define the reverse ODE step
def reverse_ode_step_euler(x, t, dt, score, schedule, schedule_derivative):
    return x + score * schedule(t) * schedule_derivative(t) * dt

def forward_ode_step_euler(x, t, dt, score, schedule, schedule_derivative):
    return x - score * schedule(t) * schedule_derivative(t) * dt

def sample_from_gaussian_mixture(weights, means, covs, n_samples):
    return torch.distributions.MixtureSameFamily(torch.distributions.Categorical(weights), 
                                               torch.distributions.MultivariateNormal(means, covs)).sample(sample_shape=(n_samples,))

def sample_ode(n_samples, d, schedule, schedule_derivative, score_function, num_steps, device, rho=1.0,
               get_sample_paths=False):
    # Sample from the prior
    ts = torch.linspace(1, 0, num_steps + 1, device=device)**rho
    dts = ts[:-1] - ts[1:]
    # Reverse ODE step
    x = torch.randn(n_samples, d, dtype=torch.float, device=device) * schedule(1)
    xs = [x]
    for i,t in enumerate(ts[:-1]):
        x = reverse_ode_step_euler(x, t, dts[i], score_function(x, t), schedule, schedule_derivative)
        if get_sample_paths:
            xs.append(x)
    if get_sample_paths:
        xs = torch.cat([x[:,None,:] for x in xs], dim=1)
        return x, xs, ts
    else:
        return x

def sample_ode_with_predefined_prior_samples(prior_samples, schedule, schedule_derivative, score_function, num_steps, device, rho=1.0,
                                             get_sample_paths=False):
    # Sample from the prior
    ts = torch.linspace(1, 0, num_steps + 1, device=device)**rho
    dts = ts[:-1] - ts[1:]
    # Reverse ODE step
    x = prior_samples
    xs = [x]
    for i,t in enumerate(ts[:-1]):
        x = reverse_ode_step_euler(x, t, dts[i], score_function(x, t), schedule, schedule_derivative)
        if get_sample_paths:
            xs.append(x)
    if get_sample_paths:
        xs = torch.cat([x[:,None,:] for x in xs], dim=1)
        return x, xs, ts
    else:
        return x
    
def gaussian_mixture_conditional_expectation(x_t, y, weights, data_means, data_covs, diffusion_sigma, observation_cov):
    """
    Calculate E[x₀|x_t,y] for a Gaussian mixture model.
    
    Args:
    x_t: (batch_size, d) - Input points x_t
    y: (batch_size, d) - Observations y
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    observation_cov: (d, d) - Covariance matrix of p(y|x_0)
    
    Returns:
    expectation: (batch_size, d) - E[x₀|x_t,y]
    """
    batch_size, d = x_t.shape
    n = len(data_means)
    
    # Calculate posterior parameters
    sigma_sq_inv = 1 / (diffusion_sigma ** 2)
    observation_precision = torch.inverse(observation_cov)
    
    posterior_means = []
    log_weights = []
    
    for i in range(n):
        # Calculate Σ'ᵢ⁻¹ = (σ²I)⁻¹ + Σy⁻¹ + Σᵢ⁻¹
        posterior_precision = sigma_sq_inv * torch.eye(d) + observation_precision + torch.inverse(data_covs[i])
        posterior_cov = torch.inverse(posterior_precision)
        
        # Calculate μ'ᵢ = Σ'ᵢ ((σ²I)⁻¹x_t + Σy⁻¹y + Σᵢ⁻¹μᵢ)
        posterior_mean = posterior_cov @ (sigma_sq_inv * x_t.T + 
                                            observation_precision @ y.T.unsqueeze(-1) + 
                                            torch.inverse(data_covs[i]) @ data_means[i].unsqueeze(1))
        posterior_means.append(posterior_mean.T)

        # Calculate log weights
        log_weight = torch.log(weights[i]) + \
                        torch.distributions.MultivariateNormal(data_means[i], data_covs[i] + diffusion_sigma**2 * torch.eye(d)).log_prob(x_t) + \
                        torch.distributions.MultivariateNormal(data_means[i], data_covs[i] + observation_cov).log_prob(y)
        log_weights.append(log_weight)
    
    posterior_means = torch.stack(posterior_means)
    log_weights = torch.stack(log_weights)
    
    # Normalize weights
    normalized_weights = torch.softmax(log_weights, dim=0)
    
    # Calculate E[x₀|x_t,y]
    expectation = torch.sum(normalized_weights.unsqueeze(-1) * posterior_means, dim=0)
    
    return expectation

def gaussian_mixture_log_gradient_conditional(x_t, y, weights, data_means, data_covs, diffusion_sigma, observation_cov):
    """
    Calculate ∇ₓₜ log p(x_t|y) for a Gaussian mixture model.
    
    Args:
    x_t: (batch_size, d) - Input points x_t
    y: (batch_size, d) - Observations y
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    observation_cov: (d, d) - Covariance matrix of p(y|x_0)
    
    Returns:
    gradient: (batch_size, d) - ∇ₓₜ log p(x_t|y)
    """
    expectation = gaussian_mixture_conditional_expectation(x_t, y, weights, data_means, data_covs, diffusion_sigma, observation_cov)
    
    # Calculate ∇ₓₜ log p(x_t|y)
    gradient = -(1 / (diffusion_sigma ** 2)) * (x_t - expectation)
    
    return gradient

def compute_gradient(x, y, weights, data_means, data_covs, diffusion_sigma, observation_cov, Sigma):
    """
    Compute the gradient ∇ₓₜ[log ∫ p(y|x₀)N(x₀|μ(x_t),Σ)dx₀]
    
    Args:
    x: (batch_size, d) - Input points (x_t)
    y: (d,) - Single observation for all batch elements
    weights: (n,) - Mixture weights
    data_means: (n, d) - Means of the mixture components
    data_covs: (n, d, d) - Covariance matrices of the mixture components
    diffusion_sigma: float - Standard deviation of p(x_t|x_0)
    observation_cov: (d, d) - Covariance matrix of p(y|x_0)
    Sigma: (d, d) or (batch_size, d, d) - Covariance matrix approximation for p(x_0|x_t)
    
    Returns:
    gradient: (batch_size, d) - Computed gradient
    """
    # Ensure x requires gradient
    x = x.detach().requires_grad_(True)
    
    # Compute μ(x_t)
    mu = gaussian_mixture_posterior_mean(x, weights, data_means, data_covs, diffusion_sigma)
    
    # Compute A = (Σy + Σ)^-1
    if Sigma.dim() == 2:
        A = torch.inverse(observation_cov + Sigma)
        v = A @ (y.unsqueeze(1) - mu.T)
    else:  # Sigma.dim() == 3
        A = torch.inverse(observation_cov.unsqueeze(0) + Sigma)
        v = torch.bmm(A, (y.unsqueeze(1) - mu.unsqueeze(2)))
    
    # Compute the gradient using autograd
    # Transpose v to match the shape of mu (batch_size, d)
    if v.dim() == 2:
        v_transposed = v.transpose(0, 1)
    else:  # v.dim() == 3
        v_transposed = v.squeeze(-1)
    gradient = torch.autograd.grad(mu, x, grad_outputs=v_transposed, create_graph=True)[0]
    
    return gradient

def visualize_gaussian_mixture_with_obs(y_obs, observation_cov, Sigma, weights, data_means, data_covs, diffusion_sigma, grid_size=50, plot_range=(-4, 4)):
    # Create a grid of points
    x = torch.linspace(plot_range[0], plot_range[1], grid_size)
    y = torch.linspace(plot_range[0], plot_range[1], grid_size)
    xx, yy = torch.meshgrid(x, y)
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # Calculate log PDF and gradients
    log_pdf = gaussian_mixture_log_pdf(grid_points, weights, data_means, data_covs, diffusion_sigma)
    gradients = gaussian_mixture_log_gradient(grid_points, weights, data_means, data_covs, diffusion_sigma)
    obs_gradient = compute_gradient(grid_points, torch.tensor(y_obs, dtype=torch.float32), weights, data_means, data_covs, diffusion_sigma, observation_cov, Sigma)
    gradients = gradients + obs_gradient

    # Reshape for plotting
    log_pdf = log_pdf.reshape(grid_size, grid_size).detach().numpy()
    gradients = gradients.reshape(grid_size, grid_size, 2).detach().numpy()

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot log PDF contours
    contour = ax.contourf(xx, yy, log_pdf, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Log PDF')

    # Plot gradient quiver
    skip = (slice(None, None, 3), slice(None, None, 3))
    quiver = ax.quiver(xx[skip], yy[skip], gradients[skip[0], skip[1], 0], gradients[skip[0], skip[1], 1],
                       scale=50, color='red', alpha=0.7)
    
    # Plot the observation as a red dot
    ax.scatter(y_obs[0], y_obs[1], c='red', s=100, marker='o', label='Observation')

    # Plot the means of the Gaussian components
    # ax.scatter(data_means[:, 0], data_means[:, 1], c='red', s=100, label='Component Means')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Gaussian Mixture Log PDF and Gradient (diffusion_sigma={diffusion_sigma})')
    ax.legend()

    plt.show()
    
def update_posterior_during_sampling_plot(frame, fig, sample_trajectories, trajectory_index, step_subset, anim_ts, weights, data_means, data_covs, schedule):
    # Clear the entire figure
    fig.clear()
    
    lim = 5
    
    # Create new subplots
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    # Extract the sample at the current time step
    sample = sample_trajectories[trajectory_index][step_subset[frame]:step_subset[frame]+1]
    t = anim_ts[frame]
    
    # Calculate the denoiser covariance matrix and mean
    denoiser_cov = gaussian_mixture_denoiser_covariance(sample, weights, data_means, data_covs, schedule(t), use_autograd=True)
    denoiser_mean = gaussian_mixture_posterior_mean(sample, weights, data_means, data_covs, schedule(t))

    # Create a grid of points
    X, Y = torch.meshgrid(torch.linspace(-lim, lim, 100), torch.linspace(-lim, lim, 100))
    pos = np.dstack((X, Y))

    # Calculate the PDF for the denoiser
    try:
        L = torch.linalg.cholesky(denoiser_cov)
        denoiser_cov = L @ L.transpose(-2, -1)
    except RuntimeError:
        # If Cholesky fails, add a small value to the diagonal
        epsilon = 1e-5
        denoiser_cov = denoiser_cov + epsilon * torch.eye(denoiser_cov.size(-1))

    mv_normal = torch.distributions.MultivariateNormal(denoiser_mean.squeeze(), denoiser_cov)
    Z = mv_normal.log_prob(torch.tensor(pos)).exp().detach().numpy()

    # Plot the contour of the denoiser PDF
    im1 = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
    fig.colorbar(im1, ax=ax1, label='Probability Density')
    ax1.scatter(sample[0, 0].detach().item(), sample[0, 1].detach().item(), color='red', s=100, label='Sample')
    ax1.scatter(denoiser_mean[0, 0].detach().item(), denoiser_mean[0, 1].detach().item(), color='blue', s=100, label='Denoiser Mean')
    ax1.set_title(f'Gaussian PDF of Denoiser at t={t:.2f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)

    # Calculate the true posterior distribution
    X, Y = torch.meshgrid(torch.linspace(-lim, lim, 100), torch.linspace(-lim, lim, 100))
    pos = torch.stack((X, Y), dim=-1)
    post_weights, post_means, post_covs = calculate_posterior_xt(sample, weights, data_means, data_covs, schedule(t)**2 * torch.eye(2))
    Z_posterior = gaussian_mixture(pos, post_weights, post_means, post_covs)

    # Plot the contour of the posterior PDF
    im2 = ax2.contourf(X.numpy(), Y.numpy(), Z_posterior.detach().numpy(), levels=20, cmap='viridis')
    fig.colorbar(im2, ax=ax2, label='Posterior Probability Density')
    ax2.scatter(sample[0, 0].detach().item(), sample[0, 1].detach().item(), color='red', s=100, label='Sample')
    ax2.scatter(denoiser_mean[0, 0].detach().item(), denoiser_mean[0, 1].detach().item(), color='blue', s=100, label='Denoiser Mean')
    ax2.set_title(f'Posterior Distribution p(x₀|x_t) at t={t:.2f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
def gaussian_mixture_covariance(weights, means, covariances):
    # Ensure inputs are PyTorch tensors
    weights = torch.tensor(weights)
    means = torch.tensor(means)
    covariances = torch.tensor(covariances)

    # Calculate the overall mean
    overall_mean = torch.sum(weights.unsqueeze(1) * means, dim=0)

    # Initialize the covariance matrix
    n_dims = means.shape[1]
    cov = torch.zeros((n_dims, n_dims))

    # Calculate the covariance
    for w, mu, sigma in zip(weights, means, covariances):
        # Add the weighted component covariance
        cov += w * sigma
        
        # Add the weighted outer product of mean differences
        mean_diff = mu - overall_mean
        cov += w * torch.outer(mean_diff, mean_diff)

    return cov

def update_covariance(samples, denoiser_cov, inv_denoiser_cov, hessian, inv_hessian, score_value, denoiser_mean, schedule, t, tnext):
    """
    Update the denoiser covariance, hessian, score function, and denoiser mean for a batch of samples
    at a new time step using a Gaussian approximation of the noisy distribution.

    Args:
        samples (torch.Tensor): Batch of samples, shape (bs, d)
        denoiser_cov (torch.Tensor): Batch of denoiser covariance matrices, shape (bs, d, d)
        inv_denoiser_cov (torch.Tensor): Batch of inverse denoiser covariance matrices, shape (bs, d, d)
        hessian (torch.Tensor): Batch of hessian matrices, shape (bs, d, d)
        inv_hessian (torch.Tensor): Batch of inverse hessian matrices, shape (bs, d, d)
        score_value (torch.Tensor): Batch of score function values, shape (bs, d)
        denoiser_mean (torch.Tensor): Batch of denoiser mean values, shape (bs, d)
        schedule (callable): Function that returns the noise level at a given time
        t (float): Current time step
        tnext (float): Next time step

    Returns:
        tuple: Updated values for denoiser_cov, inv_denoiser_cov, hessian, inv_hessian, score_value, denoiser_mean
    """
    dim = samples.shape[1]
    
    # Update the inverse covariance matrix
    new_inv_denoiser_cov = inv_denoiser_cov + (schedule(tnext)**(-2) - schedule(t)**(-2)) * torch.eye(dim)
    new_denoiser_cov = torch.linalg.inv(new_inv_denoiser_cov)
    
    new_inv_hessian = inv_hessian - (schedule(tnext)**(2) - schedule(t)**(2)) * torch.eye(dim)
    new_hessian = torch.linalg.inv(new_inv_hessian)
    
    # Score function at time t_next
    new_score_value = (new_hessian @ inv_hessian @ score_value[...,None])[...,0]
    # Denoiser mean at time t_next
    new_denoiser_mean = samples + schedule(tnext)**2 * new_score_value
    
    return new_denoiser_cov, new_inv_denoiser_cov, new_hessian, new_inv_hessian, new_score_value, new_denoiser_mean

def update_bfgs(denoiser_cov, inv_denoiser_cov, hessian, inv_hessian, score_at_t, score_at_tnext, denoiser_mean_at_t, denoiser_mean_at_tnext, schedule, t, x, dx):
    """
    Update the BFGS approximation of the Hessian and related quantities.

    This function implements the BFGS (Broyden–Fletcher–Goldfarb–Shanno) update
    for approximating the Hessian matrix and its inverse. It also updates related
    quantities such as the denoiser covariance and the score value.

    Args:
        denoiser_cov (torch.Tensor): Current denoiser covariance matrix.
        inv_denoiser_cov (torch.Tensor): Current inverse of denoiser covariance matrix.
        hessian (torch.Tensor): Current Hessian matrix.
        inv_hessian (torch.Tensor): Current inverse of Hessian matrix.
        score_at_t (torch.Tensor): Score value at current point x and time t.
        score_at_tnext (torch.Tensor): Score value at point x+dx and time t.
        denoiser_mean_at_t (torch.Tensor): Denoiser mean at current point x and time t.
        denoiser_mean_at_tnext (torch.Tensor): Denoiser mean at point x+dx and time t.
        schedule (callable): Function that returns the noise level at a given time.
        t (float): Current time.
        x (torch.Tensor): Current point.
        dx (torch.Tensor): Step taken from x to x+dx.

    Returns:
        tuple: A tuple containing:
            - updated_denoiser_cov (torch.Tensor): Updated denoiser covariance matrix.
            - updated_inv_denoiser_cov (torch.Tensor): Updated inverse of denoiser covariance matrix.
            - updated_hessian (torch.Tensor): Updated Hessian matrix.
            - updated_inv_hessian (torch.Tensor): Updated inverse of Hessian matrix.
            - updated_score_value (torch.Tensor): Updated score value.
            - updated_denoiser_mean (torch.Tensor): Updated denoiser mean.

    Note:
        This function assumes that the score_at_t and score_at_tnext are 
        ∇_x log p(x,t) and ∇_x log p(x+dx, t) respectively, i.e., the diffusion 
        time is the same, but the score is evaluated at two different points.
    """
    bs, d = x.shape
    I = torch.eye(d).unsqueeze(0).repeat(bs, 1, 1)  # shape (bs, d, d)
    de = schedule(t)**2 * (denoiser_mean_at_tnext - denoiser_mean_at_t) # shape (bs, d)
    
    gamma = 1/(dx[...,None,:] @ de[...,:,None])
    
    updated_denoiser_cov = denoiser_cov - denoiser_cov @ dx[...,:,None] @ dx[...,None,:] @ denoiser_cov / (dx[...,None,:] @ denoiser_cov @ dx[...,:,None]) + de[...,:,None] @ de[...,None,:] * gamma
    updated_inv_denoiser_cov = (I - dx[...,:,None] @ de[...,None,:] * gamma) @ inv_denoiser_cov @ (I - de[...,:,None] @ dx[...,None,:] * gamma) + dx[...,:,None] @ dx[...,None,:] * gamma
    
    updated_hessian = (updated_denoiser_cov/schedule(t)**2 - I)/schedule(t)**2
    updated_inv_hessian = torch.linalg.inv(updated_hessian + 1e-10*torch.eye(d).unsqueeze(0).repeat(bs, 1, 1))# add a jitter term to make it invertible
    
    return updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian

# Create the combined update for the sampling process
def sample_ode_with_second_order_bfgs_updates(prior_samples, schedule, schedule_derivative, score_function, num_steps, init_denoiser_cov, y, device, 
                                              weights, data_means, data_covs, cov_y,
                                              rho=1.0,get_sample_paths=False):
    # Sample from the prior
    denoiser_covs = [init_denoiser_cov[None,:,:].repeat(prior_samples.shape[0], 1, 1)]
    inv_denoiser_covs = [torch.inverse(denoiser_covs[0])]
    # The following calculation is unstable with respect to the initial denoiser covariance. It's difficult to recover the initial covariance from this. Maybe should do in log-space?
    # The point is that it is very close to 0, so when we multiply it back again with schedule(1)**2, it goes to exactly 0 due to underflow. 
    # Maybe represent it in log-space or with some other normalisation? Is it necessary at all to have the Hessian? (I guess kind of necessary for E[x_0|x-t] updates? Or maybe those updates can also be framed in a numerically stable way?)
    # We should be able to use the prior information to make the updates more stable for the denoiser covariance as well. 
    d = prior_samples.shape[1]
    hessians = [(denoiser_covs[0]/schedule(1)**2 - torch.eye(d)[None,:,:])/schedule(1)**2]
    inv_hessians = [torch.inverse(hessians[0])]
    
    ts = torch.linspace(1, 0, num_steps + 1, device=device)**rho
    dts = ts[:-1] - ts[1:]
    # Reverse ODE step
    x = prior_samples#torch.randn(n_samples, d, dtype=torch.float, device=device) * schedule(1)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    
    score_evals = [score_function(x, ts[0]).detach()]
    grad_log_p_y_xt = compute_gradient(x, y, weights, data_means, data_covs, schedule(ts[0]), cov_y, denoiser_covs[0])
    score_for_update = score_evals[-1] + grad_log_p_y_xt
    
    xs = [x]
    for i,t in enumerate(ts[:-1]):
        score_eval = score_evals[-1]
        xnew = reverse_ode_step_euler(x, t, dts[i], score_for_update, schedule, schedule_derivative)
        dx = xnew - x
        dt = dts[i]
        score_eval_at_xnext_tnext = score_function(xnew, t-dt)
        score_evals.append(score_eval_at_xnext_tnext.detach())
        
        denoiser_mean = x + schedule(t)**2 * score_eval
        denoiser_mean_at_xnext_tnext = xnew + schedule(t-dt)**2 * score_eval_at_xnext_tnext
        
        # Update covariances etc. in dt
        denoiser_cov_at_tnext, inv_denoiser_cov_at_tnext, hessian_at_tnext, inv_hessian_at_tnext, score_eval_at_tnext, denoiser_mean_at_tnext = update_covariance(x,
                                                        denoiser_covs[-1], inv_denoiser_covs[-1], hessians[-1], inv_hessians[-1], score_eval, denoiser_mean, schedule, t, t-dt)
        
        denoiser_mean_at_tnext = x + schedule(t-dt)**2 * score_function(x, t-dt)
        
        new_denoiser_cov, new_inv_denoiser_cov, new_hessian, new_inv_hessian = update_bfgs(denoiser_cov_at_tnext, inv_denoiser_cov_at_tnext, hessian_at_tnext, inv_hessian_at_tnext, score_eval_at_tnext, score_eval_at_xnext_tnext, denoiser_mean_at_tnext, denoiser_mean_at_xnext_tnext, schedule, t-dt, x, dx)
        # new_denoiser_cov, new_inv_denoiser_cov, new_hessian, new_inv_hessian = denoiser_cov_at_tnext, inv_denoiser_cov_at_tnext, hessian_at_tnext, inv_hessian_at_tnext
        
        denoiser_covs.append(new_denoiser_cov.detach())
        inv_denoiser_covs.append(new_inv_denoiser_cov.detach())
        hessians.append(new_hessian.detach())
        inv_hessians.append(new_inv_hessian.detach())
        
        grad_log_p_y_xt = compute_gradient(xnew, y, weights, data_means, data_covs, schedule(t-dt), cov_y, denoiser_covs[-1])
        score_for_update = score_evals[-1] + grad_log_p_y_xt
        
        # Append to sample path
        x = xnew
        if get_sample_paths:
            xs.append(xnew)
    if get_sample_paths:
        xs = torch.cat([x[:,None,:] for x in xs], dim=1)
        return x, xs, ts, denoiser_covs, inv_denoiser_covs, hessians, inv_hessians
    else:
        return x

    
if __name__ == '__main__':
    gaussian_mixture_log_hessian_autograd_test()