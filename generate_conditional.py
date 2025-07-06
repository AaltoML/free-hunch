# Script is based on the EDM2 codebase: https://github.com/NVlabs/edm2. 

"""Generate random images using the given model."""
import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch.utils.data import Subset
from torch_utils import distributed as dist
from torch_utils import misc
from torch.utils.data import DistributedSampler
# from training.openai_unet import UNetModel
from training.openai_loading_utils import load_model
from training.openai_preconditioning import iDDPMPrecond, iDDPMLinearPrecond
from measurement_utils.measurements import get_operator, get_noise
from torch.autograd import grad
from conditioning_utils.conditioning_mechanisms import choose_conditioning_mechanism
import skimage.metrics
import lpips
from omegaconf import DictConfig, OmegaConf
import hydra
import time
from ddnm_functions.custom_ddnm_sampling import ddnm_conditional_sampler

# import torch.distributed as dist
from log_utils import setup_logger
logger = setup_logger()
warnings.filterwarnings('ignore', '`resume_download` is deprecated')

#----------------------------------------------------------------------------
# Sampler for generation conditional on another image
def conditional_sampler(
    net, noise, cond_images, operator_kwargs, noise_kwargs, labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, **other_args
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm', 'ddpm_linear']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']
    
    logger.info(f"in conditional_sampler")

    # Setup operator and noise
    forward_operator = get_operator(**operator_kwargs)
    # forward_noise = get_noise(**noise_kwargs) # the noise is included in forward_operator
    cond_images = forward_operator.forward(cond_images, noiseless=False)

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=noise.device)

    sigma_steps = get_sigma_steps(discretization, num_steps, sigma_min, sigma_max, vp_beta_d, vp_beta_min, rho, step_indices, M, C_1, C_2, noise, vp_sigma, ve_sigma, epsilon_s)

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = noise.to(torch.float64) * (sigma(t_next) * s(t_next))

    x_all = [x_next.detach()]
    
    cond_mechanism = choose_conditioning_mechanism(other_args['conditioning_mechanism'])(other_args['cond_scaling'], 
                            forward_operator, other_args['clip_x0_mean'], init_denoiser_variance=1, init_noise_variance=sigma(t_next)**2, data_dim=x_next.shape[1:].numel(),
                            pigdm_posthoc_scaling=other_args['pigdm_posthoc_scaling'], max_vector_count=other_args['max_vector_count'],
                            data_dir=other_args['dataset_path'], image_base_covariance=other_args['image_base_covariance'], pca_component_count=other_args['pca_component_count'],
                            denoiser_mean_error_threshold=other_args['denoiser_mean_error_threshold'], use_analytical_score_time_update=other_args['use_analytical_score_time_update'],
                            project_to_diagonal=other_args['project_to_diagonal'], space_step_update_threshold=other_args['space_step_update_threshold'],
                            space_step_update_lower_threshold=other_args['space_step_update_lower_threshold'], max_rtol=other_args['max_rtol'],  do_space_updates=other_args['do_space_updates'],
                            use_analytic_var_at_end=other_args['use_analytic_var_at_end'], solver_type=other_args['solver_type'], use_rtol_func=other_args['use_rtol_func'],
                            diffpir_lambda=other_args['diffpir_lambda'])

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, None..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
            t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))

            x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

            # Euler step.
            with torch.enable_grad():
                x_hat = x_hat.requires_grad_()
                h = t_next - t_hat
                # forget about other than VE for now
                denoised_updated = cond_mechanism(x_hat, net, cond_images.to(noise.device), sigma(t_hat))
                score = -(x_hat - denoised_updated) / sigma(t_hat)**2
                d_cur = -score * sigma(t_hat)

            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h

            # Apply 2nd order correction.
            if solver == 'euler' or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == 'heun'
                with torch.enable_grad():
                    x_prime = x_prime.requires_grad_()
                    # denoised = net(x_prime, sigma(t_prime), labels).to(torch.float64)
                    denoised_updated = cond_mechanism(x_prime, net, cond_images.to(noise.device), sigma(t_prime))
                    # init_score = (denoised.to(torch.float32) - x_prime) / sigma(t_prime)**2
                    # score = init_score + p_y_xt_grad
                    # denoised_updated = x_prime + sigma(t_prime)**2 * score
                    # denoised_updated = denoised.to(torch.float32) + sigma(t_cur)**2 * p_y_xt_grad

                d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised_updated
                x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next, x_all, cond_images


def get_sigma_steps(discretization, num_steps, sigma_min, sigma_max, vp_beta_d, vp_beta_min, rho, step_indices, M, C_1, C_2, noise, vp_sigma, ve_sigma, epsilon_s):
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=noise.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=noise.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    elif discretization == 'ddpm_linear':
        # u = torch.zeros(M + 1, dtype=torch.float64, device=noise.device)
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, M, device=noise.device)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        # reverse
        alpha_bar = alpha_bar.flip(dims=[0])
        u = torch.sqrt((1-alpha_bar) / alpha_bar)
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return sigma_steps
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

def load_network(net, architecture='default', openai_state_dict_path=None, openai_setup_path=None, device=torch.device('cuda'), verbose=True, **sampler_kwargs):
    if isinstance(net, str):
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if architecture == 'default':
            net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    elif openai_state_dict_path is not None:
        net, model_args = load_model(openai_state_dict_path, openai_setup_path)
        net.eval()
        net = net.to(device)
        if sampler_kwargs['iddpm_preconditioning'] == 'cosine':
            net = iDDPMPrecond(net, img_resolution=net.img_resolution, img_channels=net.img_channels, label_dim=net.label_dim, **model_args)
        elif sampler_kwargs['iddpm_preconditioning'] == 'linear':
            net = iDDPMLinearPrecond(net, img_resolution=net.img_resolution, img_channels=net.img_channels, label_dim=net.label_dim, **model_args)
        else:
            raise ValueError(f'Preconditioning {sampler_kwargs["iddpm_preconditioning"]} not supported')
        encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None
    return net, encoder

def load_guidance_network(gnet, net, device=torch.device('cuda'), verbose=True):
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guidance network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net
    return gnet

def init_encoder(encoder, encoder_batch_size, device, verbose):
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size
    return encoder


def save_videos(images_all, video_paths):
    for i in range(images_all[0].shape[0]):
        import cv2
        dir_path = os.path.dirname(video_paths[i])
        os.makedirs(dir_path, exist_ok=True)
        _, height, width = images_all[0][i].shape
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try 'avc1' codec
        video = cv2.VideoWriter(video_paths[i], fourcc, 10, (width, height))
        if not video.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback to 'mp4v' if 'avc1' fails
            video = cv2.VideoWriter(video_paths[i], fourcc, 10, (width, height))
        for image in images_all:
            im = image[i].permute(1, 2, 0).cpu().numpy()
            frame = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            video.write(frame)
        video.release()

def save_images(images, image_paths):
    for i, image in enumerate(images.permute(0, 2, 3, 1).cpu().numpy()):
        image_dir = os.path.dirname(image_paths[i])
        os.makedirs(image_dir, exist_ok=True)
        PIL.Image.fromarray(image, 'RGB').save(image_paths[i])

def generate_conditional_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 17),        # List of random seeds. (tells the number of images per condition to generate, and their seeds)
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = conditional_sampler,          # Which sampler function to use.
    architecture        = 'default',            # Architecture to use for the network.
    openai_state_dict_path = None,              # Path to the state dict for the OpenAI model
    openai_setup_path = None,                   # Path to the setup file for the OpenAI model
    dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset', path=None),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=1, prefetch_factor=2, shuffle=False),
    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    total_images        = None,     # Total number of images to process. None = all images in dataset.
    operator_kwargs     = dict(name='noise', device=torch.device('cuda')),  # Operator to apply
    noise_kwargs        = dict(name='clean'),  # Noise to apply
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    
    logger.info(f"in generate_conditional_images")
    
    # Initialize
    misc.set_random_seed(seed, dist.get_rank())
    # Setup dataset
    dist.print0('Loading dataset...')
    dataset_kwargs['return_idx'] = True # we need to know which index are we dealing with in order to do distributed sampling naming correctly
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    if total_images is not None:
        # Take into account the dist.get_rank() and how many images each rank should generate (total_images // dist.get_world_size())
        # range_ = range(dist.get_rank() * (total_images // dist.get_world_size()), (dist.get_rank() + 1) * (total_images // dist.get_world_size()))
        range_ = range(0, total_images)
        dataset_obj = Subset(dataset_obj, range_)

    if dist.get_world_size() > 1:
        sampler = DistributedSampler(dataset_obj, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        sampler = None#misc.CustomDistributedSampler(dataset_obj, num_replicas=1, rank=0)

    # Setup data loader
    # batch_size_different_conditions = max_batch_size // len(seeds)
    data_loader = dnnlib.util.construct_class_by_name(
        dataset=dataset_obj, 
        batch_size=1,
        sampler=sampler,
        **data_loader_kwargs
    )
        
    print(f"dist rank: {dist.get_rank()}, number of images: {len(data_loader)}")
    
    print(f"dist rank: {dist.get_rank()}, number of images: {len(data_loader)}")
    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    net, encoder = load_network(net, architecture, openai_state_dict_path, openai_setup_path, device, verbose, **sampler_kwargs)
    # Load guidance network.
    gnet = load_guidance_network(gnet, net, device, verbose)
    # Initialize encoder.
    encoder = init_encoder(encoder, encoder_batch_size, device, verbose)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    images_per_conditional = len(seeds)
    num_batches = len(dataset_obj) * images_per_conditional #max((len(dataset_obj)*images_per_conditional - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    # rank_batches = np.arange(len(seeds))
    # rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        logger.info(f'Generating conditional images for {len(dataset_obj)} images.... {len(data_loader)} batches')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(data_loader)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, (global_indices, cond_images, labels) in enumerate(data_loader):

                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(data_loader), indices=global_indices)
                r.seeds = [seeds[idx] for _ in global_indices for idx in range(images_per_conditional)] # this should be roughly batch_size_different_conditions * images_per_conditional ~= total_images

                if len(r.seeds) > 0:
                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    # r.noise = r.noise.repeat(images_per_conditional, 1, 1, 1)
                    r.labels = None

                    r.images = []
                    images_all = []

                    cond_images = cond_images.repeat(images_per_conditional, 1, 1, 1)
                    cond_image_latents = encoder.encode(cond_images).to(device)
                    latents, latents_all, cond_images_forward = sampler_fn(net=net, cond_images=cond_image_latents, gnet=gnet, encoder=encoder, noise=r.noise, labels=r.labels, 
                                                               **{**sampler_kwargs, "operator_kwargs": operator_kwargs, "noise_kwargs": noise_kwargs})
                    decoded_images = encoder.decode(latents)
                    r.images = decoded_images

                    for latent in latents_all:
                        images_all.append(encoder.decode(latent))
                    
                    # forward_operator = get_operator(**operator_kwargs)
                    # # forward_noise = get_noise(**noise_kwargs)
                    # cond_images_forward = forward_operator.forward(cond_image_latents, noiseless=False)
                    cond_images_forward_decoded = encoder.decode(cond_images_forward)
                    
                    r.images_all = images_all
                    r.cond_images = cond_images[::images_per_conditional]
                    r.cond_images_forward = cond_images_forward_decoded

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

from config_utils import load_config

def cmdline():
    opts = load_config()
    outdir = opts.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # Validate options.
    if opts.net is None and opts.openai_state_dict_path is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    # Set up distributed environment
    dist.init()
    
    # Set device
    opts.device = torch.device(opts.device)
    
    if dist.get_rank() == 0:
        import sys
        log_file = open(os.path.join(opts.outdir, 'output.log'), 'w')
        # Only redirect if not in interactive mode
        if not sys.stdin.isatty():  # or use: if not hasattr(sys, 'ps1')
            sys.stdout = sys.stderr = log_file
    
    if opts.conditional:
        opts.operator_kwargs = {'name': opts.operator_name}
        opts.operator_kwargs['kernel_size'] = opts.kernel_size
        opts.operator_kwargs['intensity'] = opts.intensity
        opts.operator_kwargs['device'] = opts.device
        opts.operator_kwargs['sigma_s'] = opts.noise_sigma
        if opts.inpainting_type == 'box':
            opts.operator_kwargs['mask_opt'] = {'mask_type': 'box', 'mask_len_range': (64, 156), 'mask_prob_range': (0.1, 0.3)}
        elif opts.inpainting_type == 'random':
            opts.operator_kwargs['mask_opt'] = {'mask_type': 'random', 'mask_len_range': (64, 156), 'mask_prob_range': (opts.inpainting_prob_lower, opts.inpainting_prob_upper)}
        else:
            raise ValueError(f"Inpainting type {opts.inpainting_type} is not supported")
        opts.operator_kwargs['scale_factor'] = opts.scale_factor
        if opts.dataset == 'imagenet':
            opts.operator_kwargs['in_shape'] = (1, 3, 256, 256)
            opts.operator_kwargs['mask_opt']['image_size'] = 256
        elif opts.dataset == 'ffhq':
            opts.operator_kwargs['in_shape'] = (1, 3, 256, 256)
            opts.operator_kwargs['mask_opt']['image_size'] = 256
        else:
            raise ValueError(f"Dataset {opts.dataset} is not supported")
        # opts.sampler_kwargs['dps_zeta'] = opts.dps_zeta
        opts.noise_kwargs = {'name': opts.noise_name}
        opts.noise_kwargs['sigma'] = opts.noise_sigma
        opts.dataset_kwargs = {'class_name': opts.dataset_name}
        opts.dataset_kwargs['path'] = opts.dataset_path
        if opts.conditioning_mechanism == 'ddnm':
            sampler_fn = ddnm_conditional_sampler
            if opts.solver == 'heun':
                opts.num_steps *= 2 # to match the number of steps that we would use with the Heun solver
            logger.info(f"Using DDNM conditional sampler with {opts.num_steps} steps")
        else:
            sampler_fn = conditional_sampler
        opts.sampler_fn = sampler_fn
        image_iter = generate_conditional_images(**opts)
    else:
        assert False
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(opts.device)
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    
    t0 = time.time()
    
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        outdir = opts.outdir
        global_indices = _r.indices
        images_per_conditional = len(_r.seeds) // len(global_indices)
        images_all = _r.images_all
        cond_images = _r.cond_images
        cond_images_forward = _r.cond_images_forward
        gen_images = _r.images
        seeds = _r.seeds
                
        # save as a video
        if opts.save_videos:
            video_dir = os.path.join(outdir, 'videos')
            os.makedirs(video_dir, exist_ok=True)
            save_videos(images_all, [os.path.join(video_dir, f'{global_indices[idx // images_per_conditional]:06d}_{seed:06d}.mp4') for idx, seed in enumerate(seeds)])

        # save the generated images
        if True:
            image_dir = os.path.join(outdir, 'images')
            os.makedirs(image_dir, exist_ok=True)
            save_images(gen_images, [os.path.join(image_dir, f'{global_indices[idx // images_per_conditional]:06d}_{seed:06d}.png') for idx, seed in enumerate(seeds)])

        # save the conditioning images (not the duplicated ones)
        if opts.num_other_images_to_save > 0:
            cond_image_dir = os.path.join(outdir, 'cond_images')
            os.makedirs(cond_image_dir, exist_ok=True)
            # save_images(cond_images[::images_per_conditional], [os.path.join(cond_image_dir, f'cond_{idx:06d}.png') for idx in global_indices if idx <= opts.num_other_images_to_save])
            save_images(cond_images, [os.path.join(cond_image_dir, f'{global_indices[idx // images_per_conditional]:06d}_{seed:06d}.png') for idx, seed in enumerate(seeds)])
            forward_image_dir = os.path.join(outdir, 'forward_images')
            os.makedirs(forward_image_dir, exist_ok=True)
            # save_images(cond_images_forward, [os.path.join(forward_image_dir, f'forward_{idx:06d}.png') for idx in global_indices if idx <= opts.num_other_images_to_save])
            save_images(cond_images_forward, [os.path.join(forward_image_dir, f'{global_indices[idx // images_per_conditional]:06d}_{seed:06d}.png') for idx, seed in enumerate(seeds)])
        
        def to_eval(x):
            return (x / 255 - 0.5) * 2 # normalize to [-1, 1]
        
        # loop over the generated images:
        psnr, ssim, lpips_ = 0, 0, 0
        for (cond_img, img) in zip(cond_images.repeat(images_per_conditional, 1, 1, 1).cpu().numpy(), gen_images.cpu().numpy()):
            psnr += skimage.metrics.peak_signal_noise_ratio(cond_img, img, data_range=255)
            ssim += skimage.metrics.structural_similarity(cond_img, img, data_range=255, channel_axis=0)
        lpips_ = loss_fn_vgg(to_eval(cond_images.repeat(images_per_conditional, 1, 1, 1).to(opts.device)), to_eval(gen_images.to(opts.device))).detach().cpu().numpy()

        total_psnr += psnr / len(image_iter)
        total_ssim += ssim / len(image_iter)
        total_lpips += lpips_.item() / len(image_iter)
    
    t1 = time.time()
    print(f"Time taken: {t1 - t0}")
    print(f"PSNR: {total_psnr}, SSIM: {total_ssim}, LPIPS: {total_lpips}")
    
    def all_reduce(x):
        x = x.clone()
        torch.distributed.all_reduce(x)
        return x
    
    # Aggregate PSNR, SSIM, and LPIPS scores from all ranks
    total_psnr = torch.tensor([total_psnr], dtype=torch.float32, device='cuda')
    total_ssim = torch.tensor([total_ssim], dtype=torch.float32, device='cuda')
    total_lpips = torch.tensor([total_lpips], dtype=torch.float32, device='cuda')
    
    total_psnr = all_reduce(total_psnr) / dist.get_world_size()
    total_ssim = all_reduce(total_ssim) / dist.get_world_size()
    total_lpips = all_reduce(total_lpips) / dist.get_world_size()
    
    # Convert back to Python floats
    total_psnr = total_psnr.item()
    total_ssim = total_ssim.item()
    total_lpips = total_lpips.item()
    
    if dist.get_rank() == 0:
        print(f"Aggregated PSNR: {total_psnr}, SSIM: {total_ssim}, LPIPS: {total_lpips}")
    
    # Save the results to a file
    with open(os.path.join(opts.outdir, 'results.txt'), 'w') as f:
        f.write(f"PSNR: {total_psnr}\n")
        f.write(f"SSIM: {total_ssim}\n")
        f.write(f"LPIPS: {total_lpips}\n")

    # Delete excess generated images
    if opts.num_other_images_to_save is not None and opts.num_other_images_to_save >= 0 and (dist.get_rank() == 0):
        image_dirs = [os.path.join(opts.outdir, 'images'), os.path.join(opts.outdir, 'cond_images'), os.path.join(opts.outdir, 'forward_images')]
        for image_dir in image_dirs:
            all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
            images_to_delete = all_images[opts.num_other_images_to_save:]
            for image_name in images_to_delete:
                os.remove(os.path.join(image_dir, image_name))
            print(f"Deleted {len(images_to_delete)} excess images from {image_dir}. Kept {opts.num_other_images_to_save} images.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
