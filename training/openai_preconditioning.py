import torch
import numpy as np

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

#@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        model,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        # model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = model # globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.double).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32) else torch.float32 # and x.device.type == 'cuda' used to have this condition for using fp16

        # sigma[:] = 70

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        alpha_bar = c_in**2

        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        print(c_noise)

        # Set torch seed and numpy seed
        # torch.manual_seed(0)
        # np.random.seed(0)
        # # resample x (uniform normal noise)
        # eps = torch.randn_like(x, dtype=torch.float)
        # x = eps#(eps * sigma)
        # c_in[:] = 1
        # c_noise[:] = 999
        # c_out = -1

        F_x = self.model((c_in.to(dtype) * x.to(dtype)), c_noise.flatten().to(torch.long).repeat(2), class_labels=class_labels, **model_kwargs)[:, :self.img_channels]
        
        # assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        #(eps - torch.sqrt(1+alpha_bar)*F_x) #

        D_x = torch.clamp(D_x,-1,1)

        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)
    

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models", for the linear schedule

#@persistence.persistent_class
class iDDPMLinearPrecond(torch.nn.Module):
    def __init__(self,
        model,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        beta_min        = 0.0001,
        beta_max        = 0.02,
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        # model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.M = M
        self.model = model # globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        betas = torch.cat([torch.tensor([0]), 
                          torch.linspace(self.beta_min, self.beta_max, M)]) # The zero corresponds to the zero-noise level that we jump to at the very end
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        # reverse
        alpha_bar = alpha_bar.flip(dims=[0])
        u = torch.sqrt((1-alpha_bar) / alpha_bar)

        # u = torch.zeros(M + 1)
        # for j in range(M, 0, -1): # M, ..., 1
        #     u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

        # The following copy-pasted from OpenAI diffusion code
        betas = betas.numpy()
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.num_timesteps = M+1
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.double).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32) else torch.float32 # and x.device.type == 'cuda' used to have this condition for using fp16

        # sigma[:] = 70

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        alpha_bar = c_in**2

        c_noise = self.M - self.round_sigma(sigma, return_index=True).to(torch.float32)

        x_var = self.model((c_in.to(dtype) * x.to(dtype)), c_noise.flatten().to(torch.long).repeat(x.shape[0]), class_labels=class_labels, **model_kwargs)
        F_x = x_var[:, :self.img_channels]
        vars = x_var[:, self.img_channels:]

        x0_var = ((vars - _extract_into_tensor(self.posterior_variance, c_noise.to(torch.long), x.shape)) \
                    / _extract_into_tensor(self.posterior_mean_coef1, c_noise.to(torch.long), x.shape).pow(2)
                ).clip(min=1e-6) # Eq. (22) from Peng et al.
        
        # assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        #(eps - torch.sqrt(1+alpha_bar)*F_x) #

        D_x = torch.clamp(D_x,-1,1)

        return D_x, x0_var

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)
    

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
