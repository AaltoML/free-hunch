import torch
from torch import nn
from torch.autograd import grad
from torch.fft import fft2, ifft2
from scipy.sparse.linalg import cg, LinearOperator
import numpy as np
from warnings import warn
from abc import abstractmethod
# import gpytorch
# from gpytorch.distributions import MultivariateNormal
from conditioning_utils.utils import OrthoTransform, LazyOTCovariance
import conditioning_utils.diffpir_utils.utils_sisr as sr
import time
import math

def choose_conditioning_mechanism(name):
    if name == 'dps':
        return DPS
    elif name == 'pigdm':
        return PiGDM
    elif name == 'pigdm_videodiff_schedule':
        return PiGDM_Videodiff_schedule
    elif name == 'online_covariance':
        return BFGSOnlineUpdate
    elif name == 'peng_convert':
        return PengConvert
    elif name == 'peng_analytic':
        return PengAnalytic
    elif name == 'tmpd':
        return TMPD
    elif name == 'diffpir':
        return DiffPIR
    elif name == 'ddnm':
        raise ValueError(f"DDNM conditioning mechanism not implemented in this branch of the codebase")
    else:
        raise ValueError(f"Unknown conditioning mechanism: {name}")

class ConditioningMechanism:
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean,
                 init_denoiser_variance=None, init_noise_variance=None, data_dim=None,
                 pigdm_posthoc_scaling=False, **argv):
        self.cond_scaling = cond_scaling
        self.forward_operator = forward_operator
        self.clip_x0_mean = clip_x0_mean
    
    def __call__(self, x_t, x_0_mean, y, sigma):
        x_0_mean_new = self.x0_mean_update(x_t, x_0_mean, y, sigma)
        if self.clip_x0_mean:
            x_0_mean_new = x_0_mean_new.clip(-1,1)
        return x_0_mean_new

class DPS(ConditioningMechanism):
    def x0_mean_update(self, x_t, model, y, sigma):
        # TODO: check whether this works for batch size > 1
        x_t = x_t.requires_grad_()
        x_0_mean, _ = model(x_t, sigma)
        difference = y - self.forward_operator.forward(x_0_mean, noiseless=True)
        norm = torch.linalg.norm(difference)
        norm_squared = norm**2
        zeta = self.cond_scaling
        p_y_xt_grad = -grad(norm, x_t)[0] * zeta
        x_0_mean_new = x_0_mean + p_y_xt_grad * sigma.pow(2)
        return x_0_mean_new

class PengConvert(ConditioningMechanism):
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean,
                 init_denoiser_variance=None, init_noise_variance=None, data_dim=None,
                 pigdm_posthoc_scaling=True, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)
        self.mat_solver = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, 'scipy', argv['max_rtol'], ortho_tf, sigma_t)
        self.pigdm_posthoc_scaling = pigdm_posthoc_scaling
        self.mle_sigma_thres = 0.2

    def x0_mean_update(self, x_t, model, y, sigma):
        x_t = x_t.requires_grad_()
        x_0_mean, x0_var = model(x_t, sigma)
        if sigma < self.mle_sigma_thres:
            x0_var = x0_var
        else:
            x0_var = sigma.pow(2) / (1 + sigma.pow(2))
        mat = self.mat_solver(y, x_0_mean, theta0_var=x0_var)
        p_y_xt_grad = grad((mat.detach() * x_0_mean).sum(), x_t)[0] * self.cond_scaling # cond scaling is optional here (x0_var if self.pigdm_posthoc_scaling else 1)
        x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))
        return x_0_mean_new

class PengAnalytic(ConditioningMechanism):
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean,
                 init_denoiser_variance=None, init_noise_variance=None, data_dim=None,
                 pigdm_posthoc_scaling=True, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)
        self.mat_solver = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, 'scipy', argv['max_rtol'], ortho_tf, sigma_t, use_rtol_func=False)
        self.pigdm_posthoc_scaling = pigdm_posthoc_scaling
        analytic_var_file = "analytic_variance/imagenet/recon_mse.pt"
        self.recon_mse = torch.load(analytic_var_file)
        self.mle_sigma_thres = 0.2

    def x0_mean_update(self, x_t, model, y, sigma):
        x_t = x_t.requires_grad_()
        x_0_mean, x0_var = model(x_t, sigma)
        if sigma < self.mle_sigma_thres:
            idx = (self.recon_mse['sigmas'].to(sigma.device) - sigma).abs().argmin()
            x0_var = self.recon_mse['mse_list'][idx].to(sigma.device)
        else:
            x0_var = sigma.pow(2) / (1 + sigma.pow(2)) 
        mat = self.mat_solver(y, x_0_mean, theta0_var=x0_var)
        p_y_xt_grad = grad((mat.detach() * x_0_mean).sum(), x_t)[0] * self.cond_scaling # cond scaling is optional here (x0_var if self.pigdm_posthoc_scaling else 1)
        x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))
        return x_0_mean_new

class TMPD(ConditioningMechanism):
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean,
                 init_denoiser_variance=None, init_noise_variance=None, data_dim=None,
                 pigdm_posthoc_scaling=True, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)
        self.mat_solver = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, 'scipy', argv['max_rtol'], ortho_tf, sigma_t, use_rtol_func=True)
        self.pigdm_posthoc_scaling = pigdm_posthoc_scaling
        
    def x0_mean_update(self, x_t, model, y, sigma):
        # Implementation from Peng et al.,
        x_t = x_t.requires_grad_()
        x_0_mean_, _ = model(x_t, sigma)
        x0_var = grad(x_0_mean_.sum(), x_t, retain_graph=True)[0] * sigma.pow(2)
        mat = self.mat_solver(y, x_0_mean_, theta0_var=x0_var, sigma_t=sigma.item())
        model.zero_grad()
        x_t = x_t.requires_grad_()
        x_0_mean, _ = model(x_t, sigma)
        p_y_xt_grad = grad((mat.detach() * x_0_mean).sum(), x_t)[0] * self.cond_scaling # cond scaling is optional here (x0_var if self.pigdm_posthoc_scaling else 1)
        x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))
        return x_0_mean_new

class PiGDM(ConditioningMechanism):
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean,
                 init_denoiser_variance=None, init_noise_variance=None, data_dim=None,
                 pigdm_posthoc_scaling=True, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)
        self.mat_solver = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, 'scipy', argv['max_rtol'], ortho_tf, sigma_t)
        # self.mat_solver = __MAT_SOLVER__[forward_operator.name]
        self.pigdm_posthoc_scaling = pigdm_posthoc_scaling

    def x0_mean_update(self, x_t, model, y, sigma):
        x_t = x_t.requires_grad_()
        x_0_mean, _ = model(x_t, sigma)
        x0_var = sigma.pow(2) / (1 + sigma.pow(2))
        mat = self.mat_solver(y, x_0_mean, theta0_var=x0_var)
        p_y_xt_grad = grad((mat.detach() * x_0_mean).sum(), x_t)[0] * (x0_var if self.pigdm_posthoc_scaling else 1) * self.cond_scaling # cond scaling is optional here
        # construct the denoised estimate and clip to [-1,1]
        x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))
        return x_0_mean_new

class PiGDM_Videodiff_schedule(ConditioningMechanism):
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean,
                 init_denoiser_variance=None, init_noise_variance=None, data_dim=None,
                 pigdm_posthoc_scaling=False, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)
        self.mat_solver = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, 'scipy', argv['max_rtol'], ortho_tf, sigma_t)
        # self.mat_solver = __MAT_SOLVER__[forward_operator.name]

    def x0_mean_update(self, x_t, model, y, sigma):
        x_t = x_t.requires_grad_()
        x_0_mean, _ = model(x_t, sigma)
        x0_var = sigma.pow(2)
        mat = self.mat_solver(y, x_0_mean, theta0_var=x0_var)
        # vector = self.forward_operator.forward(difference, noiseless=True)
        p_y_xt_grad = grad((mat.detach() * x_0_mean).sum(), x_t)[0] * self.cond_scaling
        x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))
        return x_0_mean_new

class DiffPIR(ConditioningMechanism):
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)
        self.mat_solver = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, 'scipy', argv['max_rtol'], ortho_tf, sigma_t)
        self.lambda_ = argv['diffpir_lambda']
        
    def x0_mean_update(self, x_t, model, y, sigma):
        assert self.lambda_ is not None, "lambda_ must be specified for DiffPIR guidance"
        x0_mean, _ = model(x_t, sigma)
        x0_var = sigma.pow(2) / self.lambda_
        mat = self.mat_solver(y, x0_mean, x0_var)
        hat_x0 = x0_mean + mat * x0_var
        return hat_x0

from conditioning_utils.online_update_bfgs import CovarianceHessianBFGS, CovarianceHessianBFGSDCT, CovarianceHessianBFGSDCTPCA

class BFGSOnlineUpdate(ConditioningMechanism):
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean, init_denoiser_variance, init_noise_variance, data_dim,
                 pigdm_posthoc_scaling=False, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)
        self.mat_solver = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, argv['solver_type'], argv['max_rtol'], ortho_tf, sigma_t, use_rtol_func=argv['use_rtol_func'])
        self.project_to_diagonal = argv['project_to_diagonal']
        if argv['image_base_covariance'] == 'identity':
            self.covariance_model = CovarianceHessianBFGS(init_denoiser_variance, init_noise_variance.item(), data_dim, dtype=torch.complex128, 
                                                          max_vector_count=argv['max_vector_count'], project_to_diagonal=self.project_to_diagonal)
        elif argv['image_base_covariance'] == 'dct_diagonal':
            self.covariance_model = CovarianceHessianBFGSDCT(argv['data_dir'], init_noise_variance.item(), data_dim, dtype=torch.complex128, 
                                                             max_vector_count=argv['max_vector_count'], project_to_diagonal=self.project_to_diagonal,
                                                             use_precalculated_info=True)
        elif argv['image_base_covariance'] == 'pca_dct_diagonal':
            self.covariance_model = CovarianceHessianBFGSDCTPCA(argv['data_dir'], init_noise_variance.item(), data_dim, dtype=torch.complex128, 
                                                                max_vector_count=argv['max_vector_count'], pca_component_count=argv['pca_component_count'], 
                                                                project_to_diagonal=self.project_to_diagonal)
        elif argv['image_base_covariance'] == 'dct_diagonal_noinfo':
            self.covariance_model = CovarianceHessianBFGSDCT(argv['data_dir'], init_noise_variance.item(), data_dim, dtype=torch.complex128, 
                                                             max_vector_count=argv['max_vector_count'], project_to_diagonal=self.project_to_diagonal, 
                                                             use_precalculated_info=False)
        self.do_space_updates = argv['do_space_updates']
        self.init_denoiser_variance = init_denoiser_variance
        self.init_noise_variance = init_noise_variance
        self.data_dim = data_dim
        self.sigmas = [] # accumulates all the sigmas where the model is evaluated at
        self.xs = []
        self.denoiser_means = []
        self.denoiser_mean_error_threshold = argv['denoiser_mean_error_threshold']
        self.use_analytical_score_time_update = argv['use_analytical_score_time_update']
        self.space_step_update_threshold = argv['space_step_update_threshold']
        self.space_step_update_lower_threshold = argv['space_step_update_lower_threshold']
        self.pigdm_posthoc_scaling = pigdm_posthoc_scaling
        self.use_analytic_var_at_end = argv['use_analytic_var_at_end']
        analytic_var_file = "analytic_variance/imagenet/recon_mse.pt"
        self.recon_mse = torch.load(analytic_var_file)
        self.mle_sigma_thres = 0.2
        self.mat_solver_analytic_cov = lambda y, x0_mean, theta0_var=None, covariance_model=None, ortho_tf=OrthoTransform(), sigma_t=None: choose_solver(
            forward_operator.name, forward_operator, y, x0_mean, theta0_var, covariance_model, 'scipy', argv['max_rtol'], ortho_tf, sigma_t, use_rtol_func=False)

    def update_time_step(self, x_t, sigma_t, sigma_tnext, score_t):
        self.covariance_model.update_time_step(x_t, sigma_t, sigma_tnext, score_t)
        
    def update_space_step(self, denoiser_mean_at_x, denoiser_mean_at_xnext, sigma_t, x, xnext):
        self.covariance_model.update_space_step(denoiser_mean_at_x, denoiser_mean_at_xnext, sigma_t, x, xnext)

    def x0_mean_update(self, x_t, model, y, sigma):
        x_t = x_t.requires_grad_()
        x_0_mean, _ = model(x_t, sigma)
        if self.do_space_updates:
            if len(self.sigmas) != 0 and sigma != self.sigmas[-1]:
                start_time = time.time()
                score_previous = (self.denoiser_means[-1] - self.xs[-1]) / self.sigmas[-1]**2
                x_0_mean_at_previous_x_current_sigma, _ = self.covariance_model.update_time_step(self.xs[-1], self.sigmas[-1], sigma.item(), score_previous)
                # x_0_mean_at_previous_x_current_sigma = x_0_mean_at_previous_x_current_sigma.clip(-1,1)
                end_time = time.time()
                print(f"Time taken for update_time_step: {end_time - start_time:.4f} seconds")
            elif len(self.sigmas) != 0 and sigma == self.sigmas[-1]: # for Heun sampler, in case we don't do time update
                x_0_mean_at_previous_x_current_sigma = self.denoiser_means[-1]
            if len(self.xs) != 0 and not torch.allclose(x_t.detach().cpu(), self.xs[-1]):
                start_time = time.time()
                if not self.use_analytical_score_time_update:
                    with torch.no_grad(): # TODO: try using the time step updates for this calculation instead
                        x_0_mean_at_previous_x_current_sigma, _ = model(self.xs[-1].to(x_t.device), sigma)
                # if np.abs(self.sigmas[-1] - sigma.item()) < self.space_step_update_threshold and np.abs(self.sigmas[-1] - sigma.item()) > self.space_step_update_lower_threshold:
                if sigma.item() > self.space_step_update_lower_threshold and sigma.item() < self.space_step_update_threshold:
                    self.covariance_model.update_space_step(x_0_mean_at_previous_x_current_sigma, x_0_mean, sigma.item(), self.xs[-1], x_t)
                end_time = time.time()
                print(f"Time taken for update_space_step: {end_time - start_time:.4f} seconds")
        else:
            if len(self.sigmas) != 0 and sigma != self.sigmas[-1]:
                score_previous = (self.denoiser_means[-1] - self.xs[-1]) / self.sigmas[-1]**2
                self.covariance_model.update_time_step(self.xs[-1], self.sigmas[-1], sigma.item(), score_previous, only_covariance=True)

        # This should calculate (y - A x0_mean)^T (A C A^T + sigma_y^2 I)^{-1} A
        # for a given C. 
        start_time = time.time()
        mat = self.mat_solver(y, x_0_mean, covariance_model=self.covariance_model, sigma_t=sigma.item())

        end_time = time.time()
        print(f"Time taken for mat_solver: {end_time - start_time:.4f} seconds")
        
        if self.use_analytic_var_at_end and sigma < self.mle_sigma_thres:
            idx = (self.recon_mse['sigmas'].to(sigma.device) - sigma).abs().argmin()
            x0_var = self.recon_mse['mse_list'][idx].to(sigma.device)
            mat = self.mat_solver_analytic_cov(y, x_0_mean, theta0_var=x0_var)
            p_y_xt_grad = grad((mat.detach() * x_0_mean).sum(), x_t)[0] * self.cond_scaling
            x_0_mean_new = x_0_mean + p_y_xt_grad * sigma.pow(2)
        else:
            p_y_xt_grad = grad((mat.detach() * x_0_mean).sum(), x_t)[0]
            # p_y_xt_grad = self.covariance_model.denoiser_cov_vector_dot(mat, use_cuda=True) * self.cond_scaling / sigma.pow(2)
            x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))
            if (p_y_xt_grad * sigma.pow(2)).std() > self.denoiser_mean_error_threshold:
                p_y_xt_grad = self.covariance_model.denoiser_cov_vector_dot(mat.detach(), use_cuda=True) * self.cond_scaling / sigma.pow(2)
                x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))
            else:
                p_y_xt_grad *= self.cond_scaling
                x_0_mean_new = (x_0_mean + p_y_xt_grad * sigma.pow(2))

        self.sigmas.append(sigma.detach().cpu().item())
        self.xs.append(x_t.detach().cpu())
        self.denoiser_means.append(x_0_mean.detach().cpu())
        
        return x_0_mean_new



class DDNMPlus(ConditioningMechanism):
    # DDNM+ works a bit differently from the other conditioning mechanisms
    def __init__(self, cond_scaling, forward_operator, clip_x0_mean, **argv):
        super().__init__(cond_scaling, forward_operator, clip_x0_mean)

#---------------------------------------------
# Implementation of mat solver (computing v)
#---------------------------------------------

def rtol_func(sigma, rtol_max=1e0, rtol_min=1e-14):
    # Define the range of sigma and corresponding rtol values
    sigma_min, sigma_max = 0.1, 80.0
    # rtol_min, rtol_max = 1e-14, 1e0
    
    # Ensure sigma is within the defined range
    sigma = max(min(sigma, sigma_max), max(sigma_min, sigma))
    
    # Calculate the logarithmic interpolation factor with power function
    p = 0.1  # Adjust this value to control the steepness (e.g., 2 or 3)
    log_factor = ((math.log10(sigma) - math.log10(sigma_min)) / (math.log10(sigma_max) - math.log10(sigma_min))) ** p
    
    # Interpolate rtol value logarithmically
    log_rtol = log_factor * (math.log10(rtol_max) - math.log10(rtol_min)) + math.log10(rtol_min)
    rtol = 10 ** log_rtol
    
    return rtol

def rtol_func_2(sigma, rtol_max=1e0, rtol_min=1e-4):
    # This used mainly for TMPD, the large variance values in the beginning make the solver implementation a bit slow
    # This makes the running time more reasonable
    # Define the range of sigma and corresponding rtol values
    sigma_min, sigma_max = 0.1, 80.0
    # rtol_min, rtol_max = 1e-14, 1e0
    
    # Ensure sigma is within the defined range
    sigma = max(min(sigma, sigma_max), max(sigma_min, sigma))
    
    # Calculate the logarithmic interpolation factor with power function
    p = 0.05  # optimising the 
    log_factor = ((math.log10(sigma) - math.log10(sigma_min)) / (math.log10(sigma_max) - math.log10(sigma_min))) ** p
    
    # Interpolate rtol value logarithmically
    log_rtol = log_factor * (math.log10(rtol_max) - math.log10(rtol_min)) + math.log10(rtol_min)
    rtol = 10 ** log_rtol
    
    return rtol
    
__MAT_SOLVER__ = {}

def register_mat_solver(name):
    def wrapper(func):
        __MAT_SOLVER__[name] = func
        return func
    return wrapper

@torch.no_grad()
def _inpainting_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform(), sigma_t=None):
    mask = operator.mask
    sigma_s = operator.sigma_s.clip(min=0.001)
    if theta0_var.numel() == 1:
        mat = (mask * y - mask * x0_mean) / (sigma_s.pow(2) + theta0_var)
    else:
        device = x0_mean.device
        sigma_s, mask, y, x0_mean, theta0_var = \
            sigma_s.cpu(), mask.cpu(), y.cpu(), x0_mean.cpu(), theta0_var.cpu()
        ot = ortho_tf
        iot = ortho_tf.inv

        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (x0_mean.numel(), x0_mean.numel()))

            def _matvec(self, mat):
                mat = torch.Tensor(mat).reshape(x0_mean.shape)
                mat = sigma_s**2 * mat + mask * iot(theta0_var * ot(mat))
                mat = mat.flatten().detach().cpu().numpy()
                return mat
        
        b = (mask * y - mask * x0_mean).flatten().detach().cpu().numpy()
        mat, info = cg(A(), b, tol=1e-4 if sigma_t is None else rtol_func_2(sigma_t), maxiter=1000)
        if info != 0:
            warn('CG not converge.')
        mat = torch.Tensor(mat).reshape(x0_mean.shape).to(device)
   
    return mat

@torch.no_grad()
def _inpainting_mat_generic_customcuda_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol, ortho_tf=OrthoTransform(), sigma_t=None):
    mask = operator.mask
    sigma_s = operator.sigma_s.clip(min=0.001)
    device = x0_mean.device
    
    sigma_s, mask, y, x0_mean = \
        sigma_s.cuda(), mask.cuda(), y.cuda(), x0_mean.cuda()
    ot = ortho_tf
    iot = ortho_tf.inv
    
    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, mat):
            mat = mat.reshape(x0_mean.shape)
            # this multiplies a vector with (A C A^T + sigma_s^2 I)
            mat = sigma_s**2 * mat + mask * iot(covariance_model.denoiser_cov_vector_dot(ot(mask * mat), use_cuda=True))
            mat = mat.flatten().detach()
            return mat

    # the residual (y - A x0_mean)
    b = mask * y - mask * x0_mean
    b = b.flatten().detach().cuda()

    mat, info = torch_cg.cg(A(), b, rtol=rtol_func(sigma_t, max_rtol), maxiter=5000)
    sigma_s = sigma_s.cuda()

    # print(f"rtol: {rtol_func(sigma_t)}")
    
    if info['niter'] == 5000:
        warn('CG not converge.')
    mat = torch.Tensor(mat).reshape(x0_mean.shape)

    return mat.to(device)

@torch.no_grad()
def _inpainting_mat_generic_scipy_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol, ortho_tf=OrthoTransform(), sigma_t=None):
    mask = operator.mask
    sigma_s = operator.sigma_s.clip(min=0.001)
    device = x0_mean.device
    sigma_s, mask, y, x0_mean = \
        sigma_s.cpu(), mask.cpu(), y.cpu(), x0_mean.cpu()
    ot = ortho_tf
    iot = ortho_tf.inv

    class A(LinearOperator):
        def __init__(self):
            super().__init__(np.float32, (x0_mean.numel(), x0_mean.numel()))

        def _matvec(self, mat):
            mat = torch.Tensor(mat).reshape(x0_mean.shape)
            mat = sigma_s**2 * mat + mask * iot(covariance_model.denoiser_cov_vector_dot(ot(mask * mat), use_cuda=False))
            mat = mat.flatten().detach().cpu().numpy()
            return mat
    
    b = (mask * y - mask * x0_mean).flatten().detach().cpu().numpy()
    mat, info = cg(A(), b, tol=1e-4 if sigma_t is None else rtol_func_2(sigma_t), maxiter=1000)
    if info != 0:
        warn('CG not converge.')
    mat = torch.Tensor(mat).reshape(x0_mean.shape).to(device)
   
    return mat

@torch.no_grad()
def _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform(), sigma_t=None):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    if theta0_var.numel() == 1:
        mat = ifft2(fft2(y - ifft2(FB * fft2(x0_mean))) / (sigma_s.pow(2) + theta0_var * F2B) * FBC).real
    
    else:
        device = x0_mean.device
        sigma_s, FB, FBC, F2B, FBFy, y, x0_mean, theta0_var = \
            sigma_s.cpu(), FB.cpu(), FBC.cpu(), F2B.cpu(), FBFy.cpu(), y.cpu(), x0_mean.cpu(), theta0_var.cpu()
        ot = ortho_tf
        iot = ortho_tf.inv

        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (y.numel(), y.numel()))

            def _matvec(self, u):
                u = torch.Tensor(u).reshape(y.shape)
                u = sigma_s**2 * u + ifft2(FB * fft2(iot(theta0_var * ot(ifft2(FBC * fft2(u)).real)))).real
                u = u.flatten().detach().cpu().numpy()
                return u
        
        b = y - ifft2(FB * fft2(x0_mean)).real
        b = b.flatten().detach().cpu().numpy()

        u, info = cg(A(), b, tol=1e-4 if sigma_t is None else rtol_func_2(sigma_t), maxiter=1000)
        if info != 0:
            warn('CG not converge.')
        u = torch.Tensor(u).reshape(y.shape)

        mat = (ifft2(FBC * fft2(u)).real).to(device)
   
    return mat

# from conditioning_utils.cg import cg_batch
import conditioning_utils.cg as torch_cg

@torch.no_grad()
def _deblur_mat_generic_customcuda_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol, ortho_tf=OrthoTransform(), sigma_t=None):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    # else:
    device = x0_mean.device
    # sigma_s, FB, FBC, F2B, FBFy, y, x0_mean = \
    #     sigma_s.cpu(), FB.cpu(), FBC.cpu(), F2B.cpu(), FBFy.cpu(), y.cpu(), x0_mean.cpu()
    sigma_s, FB, FBC, F2B, FBFy, y, x0_mean = \
        sigma_s.cuda(), FB.cuda(), FBC.cuda(), F2B.cuda(), FBFy.cuda(), y.cuda(), x0_mean.cuda()
    ot = ortho_tf
    iot = ortho_tf.inv
    
    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, u):
            u = u.reshape(y.shape)
            # this multiplies a vector with (A C A^T + sigma_s^2 I)
            u = sigma_s**2 * u + torch.fft.ifft2(FB * torch.fft.fft2(iot(covariance_model.denoiser_cov_vector_dot(ot(torch.fft.ifft2(FBC * torch.fft.fft2(u)).real), use_cuda=True)))).real
            u = u.flatten().detach()
            return u

    # the residual (y - A x0_mean)
    b = y - ifft2(FB * fft2(x0_mean)).real
    b = b.flatten().detach().cuda()#.cpu().numpy()

    u, info = torch_cg.cg(A(), b, rtol=rtol_func(sigma_t, max_rtol), maxiter=5000)
    sigma_s = sigma_s.cuda()
    # u, info = torch_cg.cg(A(), b, rtol=1e-4, maxiter=1000)
    if info['niter'] == 2000:
        warn('CG not converge.')
    u = torch.Tensor(u).reshape(y.shape)

    mat = (torch.fft.ifft2(FBC * torch.fft.fft2(u)).real).to(device)
   
    return mat

def _deblur_mat_generic_scipy_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol, ortho_tf=OrthoTransform(), sigma_t=None):
    sigma_s = operator.sigma_s.clip(min=0.001)
    FB, FBC, F2B, FBFy = operator.pre_calculated

    device = x0_mean.device
    sigma_s, FB, FBC, F2B, FBFy, y, x0_mean = \
        sigma_s.cpu(), FB.cpu(), FBC.cpu(), F2B.cpu(), FBFy.cpu(), y.cpu(), x0_mean.cpu()
    ot = ortho_tf
    iot = ortho_tf.inv

    class A(LinearOperator):
        def __init__(self):
            super().__init__(np.float32, (y.numel(), y.numel()))

        def _matvec(self, u):
            u = torch.Tensor(u).reshape(y.shape)
            #u = sigma_s**2 * u + ifft2(FB * fft2(iot(theta0_var * ot(ifft2(FBC * fft2(u)).real)))).real
            u = sigma_s**2 * u + torch.fft.ifft2(FB * torch.fft.fft2(iot(covariance_model.denoiser_cov_vector_dot(ot(torch.fft.ifft2(FBC * torch.fft.fft2(u)).real), use_cuda=False)))).real
            u = u.flatten().detach().cpu().numpy()
            return u
    
    b = y - ifft2(FB * fft2(x0_mean)).real
    b = b.flatten().detach().cpu().numpy()

    u, info = cg(A(), b, tol=1e-4 if sigma_t is None else rtol_func_2(sigma_t), maxiter=1000)
    if info != 0:
        warn('CG not converge.')
    u = torch.Tensor(u).reshape(y.shape)

    mat = (ifft2(FBC * fft2(u)).real).to(device)
   
    return mat

def choose_solver(operator_name, operator, y, x0_mean, theta0_var=None, covariance_model=None, method='customcuda', max_rtol=1, ortho_tf=OrthoTransform(), sigma_t=None, use_rtol_func=False):
    if operator_name == 'gaussian_blur':
        if method == 'customcuda':
            return _deblur_mat_generic_customcuda_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t)
        elif method == 'customscipy':
            return _deblur_mat_generic_scipy_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t if use_rtol_func else None)
        elif use_rtol_func:
            return _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf, sigma_t)
        else:
            return _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf)
    elif operator_name == 'super_resolution':
        if method == 'customcuda':
            return _super_resolution_mat_generic_customcuda_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t)
        elif method == 'customscipy':
            return _super_resolution_mat_generic_scipy_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t if use_rtol_func else None)
        elif use_rtol_func:
            return _super_resolution_mat(operator, y, x0_mean, theta0_var, ortho_tf, sigma_t)
        else:
            return _super_resolution_mat(operator, y, x0_mean, theta0_var, ortho_tf)
    elif operator_name == 'motion_blur':
        if method == 'customcuda':
            return _deblur_mat_generic_customcuda_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t)
        elif method == 'customscipy':
            return _deblur_mat_generic_scipy_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t if use_rtol_func else None)
        elif use_rtol_func:
            return _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf, sigma_t)
        else:
            return _deblur_mat(operator, y, x0_mean, theta0_var, ortho_tf)
    elif operator_name == 'inpainting':
        if method == 'customcuda':
            return _inpainting_mat_generic_customcuda_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t)
        elif method == 'customscipy':
            return _inpainting_mat_generic_scipy_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=max_rtol, ortho_tf=ortho_tf, sigma_t=sigma_t if use_rtol_func else None)
        elif use_rtol_func:
            return _inpainting_mat(operator, y, x0_mean, theta0_var, ortho_tf, sigma_t)
        else:
            return _inpainting_mat(operator, y, x0_mean, theta0_var, ortho_tf)
    else:
        raise ValueError("Invalid operator name. Please choose 'gaussian_blur', 'super_resolution', 'motion_blur', or 'inpainting'.")

@torch.no_grad()
def _super_resolution_mat(operator, y, x0_mean, theta0_var, ortho_tf=OrthoTransform(), sigma_t=None):
    sigma_s = operator.sigma_s.clip(min=0.001).clip(min=1e-2)
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated
    
    if theta0_var.numel() == 1:
        invW = torch.mean(sr.splits(F2B, sf), dim=-1, keepdim=False)
        mat = ifft2(FBC * (fft2(y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)) / (sigma_s.pow(2) + theta0_var * invW)).repeat(1, 1, sf, sf)).real
    
    else:
        device = x0_mean.device
        sigma_s, FB, FBC, F2B, FBFy, y, x0_mean, theta0_var = \
            sigma_s.cpu(), FB.cpu(), FBC.cpu(), F2B.cpu(), FBFy.cpu(), y.cpu(), x0_mean.cpu(), theta0_var.cpu()
        ot = ortho_tf
        iot = ortho_tf.inv

        class A(LinearOperator):
            def __init__(self):
                super().__init__(np.float32, (y.numel(), y.numel()))

            def _matvec(self, u):
                u = torch.Tensor(u).reshape(y.shape)
                u = sigma_s**2 * u + sr.downsample(ifft2(FB * fft2(iot(theta0_var * ot(ifft2(FBC * fft2(sr.upsample(u, sf))).real)))), sf)
                u = u.real.flatten().detach().cpu().numpy()
                return u
        
        b = (y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)).real
        b = b.flatten().detach().cpu().numpy()

        u, info = cg(A(), b, tol=1e-4 if sigma_t is None else rtol_func_2(sigma_t), maxiter=1000)
        if info != 0:
            warn('CG not converge.')
        u = torch.Tensor(u).reshape(y.shape)

        mat = (ifft2(FBC * fft2(sr.upsample(u, sf))).real).to(device)

    return mat

def _super_resolution_mat_generic_customcuda_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=1,  ortho_tf=OrthoTransform(), sigma_t=None):
    sigma_s = operator.sigma_s.clip(min=0.001).clip(min=1e-2)
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated
    device = x0_mean.device
    sigma_s, FB, FBC, F2B, FBFy, y, x0_mean = \
        sigma_s.cuda(), FB.cuda(), FBC.cuda(), F2B.cuda(), FBFy.cuda(), y.cuda(), x0_mean.cuda()
    ot = ortho_tf
    iot = ortho_tf.inv
    
    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, u):
            u = u.reshape(y.shape)
            # this multiplies a vector with (A C A^T + sigma_s^2 I)
            u = sigma_s**2 * u + sr.downsample(torch.fft.ifft2(FB * torch.fft.fft2(iot(covariance_model.denoiser_cov_vector_dot(ot(torch.fft.ifft2(FBC * torch.fft.fft2(sr.upsample(u, sf))).real), use_cuda=True)))).real, sf)
            u = u.flatten().detach()
            return u

    # the residual (y - A x0_mean)
    b = (y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)).real
    b = b.flatten().detach().cuda()#.cpu().numpy()

    u, info = torch_cg.cg(A(), b, rtol=rtol_func(sigma_t, max_rtol), maxiter=5000)
    sigma_s = sigma_s.cuda()
    # u, info = torch_cg.cg(A(), b, rtol=1e-4, maxiter=1000)
    if info['niter'] == 2000:
        warn('CG not converge.')
    u = torch.Tensor(u).reshape(y.shape)

    mat = (torch.fft.ifft2(FBC * torch.fft.fft2(sr.upsample(u, sf))).real).to(device)
   
    return mat

@torch.no_grad()
def _super_resolution_mat_generic_scipy_bfgs_tailored(operator, y, x0_mean, covariance_model, max_rtol=1, ortho_tf=OrthoTransform(), sigma_t=None):
    sigma_s = operator.sigma_s.clip(min=0.001).clip(min=1e-2)
    sf = operator.scale_factor
    FB, FBC, F2B, FBFy = operator.pre_calculated
    
    device = x0_mean.device
    sigma_s, FB, FBC, F2B, FBFy, y, x0_mean = \
        sigma_s.cpu(), FB.cpu(), FBC.cpu(), F2B.cpu(), FBFy.cpu(), y.cpu(), x0_mean.cpu()
    ot = ortho_tf
    iot = ortho_tf.inv

    class A(LinearOperator):
        def __init__(self):
            super().__init__(np.float32, (y.numel(), y.numel()))

        def _matvec(self, u):
            u = torch.Tensor(u).reshape(y.shape)
            u = sigma_s**2 * u + sr.downsample(torch.fft.ifft2(FB * torch.fft.fft2(iot(covariance_model.denoiser_cov_vector_dot(ot(torch.fft.ifft2(FBC * torch.fft.fft2(sr.upsample(u, sf))).real), use_cuda=False)))).real, sf)
            u = u.real.flatten().detach().cpu().numpy()
            return u
    
    b = (y - sr.downsample(ifft2(FB * fft2(x0_mean)), sf)).real
    b = b.flatten().detach().cpu().numpy()

    u, info = cg(A(), b, tol=1e-4 if sigma_t is None else rtol_func_2(sigma_t), maxiter=1000)
    if info != 0:
        warn('CG not converge.')
    u = torch.Tensor(u).reshape(y.shape)

    mat = (ifft2(FBC * fft2(sr.upsample(u, sf))).real).to(device)

    return mat