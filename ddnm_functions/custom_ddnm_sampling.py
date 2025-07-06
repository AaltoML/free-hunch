import torch
import torchvision.utils as tvu
import torchvision.transforms as transforms
import numpy as np
import random
import os
import tqdm
from measurement_utils.measurements import get_operator, get_noise

def ddnm_conditional_sampler(
    net, noise, cond_images, operator_kwargs, noise_kwargs, labels=None, randn_like=torch.randn_like,
    num_steps=18, simplified=False, **other_args
):
    """Wrapper around DDNM+ SVD sampler to match conditional_sampler interface"""

    # 1. Create appropriate SVD operator based on operator_kwargs
    if operator_kwargs['name'] == 'gaussian_blur':
        from ddnm_functions.svd_operators import Deblurring
        # Load the pre-computed kernel instead of generating it
        kernel = torch.Tensor(np.load('./measurement_utils/kernels/gaussian_ks61_std3.0.npy')).to(noise.device)
        A_funcs = Deblurring(kernel, 3, noise.shape[-1], noise.device, use_ddnm_kernel_params=other_args['use_ddnm_kernel_params'])
        # 2. Get measurement from conditional image
        y = A_funcs.A(cond_images.reshape(cond_images.shape[0], -1))
        y = y + torch.randn_like(y) * noise_kwargs['sigma']
        y_for_output = y.clone().reshape(cond_images.shape)
    elif operator_kwargs['name'] == 'motion_blur':
        raise NotImplementedError("Motion blur not implemented for DDNM")
        from ddnm_functions.svd_operators import Deblurring
        # Load the pre-computed motion blur kernel
        kernel = torch.Tensor(np.load('./measurement_utils/kernels/motion_ks61_std0.5.npy')).to(noise.device)
        A_funcs = Deblurring(kernel, 3, noise.shape[-1]**2, noise.device)
    elif operator_kwargs['name'] == 'inpainting':
        from ddnm_functions.svd_operators import create_inpainting_operator
        A_funcs = create_inpainting_operator(3, noise.shape[-1], operator_kwargs['mask_opt'], noise.device)
        # 2. Get measurement from conditional image
        y = A_funcs.A(cond_images.reshape(cond_images.shape[0], -1))
        y = y + torch.randn_like(y) * noise_kwargs['sigma']
        y_for_output = A_funcs.A_with_zeros(cond_images.reshape(cond_images.shape[0], -1)).reshape(cond_images.shape)
    elif operator_kwargs['name'] == 'super_resolution':
        from ddnm_functions.svd_operators import SuperResolution
        A_funcs = SuperResolution(3, noise.shape[-1], operator_kwargs['scale_factor'], noise.device)
        y = A_funcs.A(cond_images.reshape(cond_images.shape[0], -1))
        y = y + torch.randn_like(y) * noise_kwargs['sigma']
        y_for_output = y.clone().reshape(1, 3, noise.shape[-1]//operator_kwargs['scale_factor'], noise.shape[-1]//operator_kwargs['scale_factor'])
    else:
        raise ValueError(f"Operator {operator_kwargs['name']} not supported yet")
    
    # 3. Get noise level from noise_kwargs
    sigma_y = noise_kwargs['sigma']
    
    # 4. Run DDNM+ with SVD
    with torch.no_grad():
        if simplified:
            raise NotImplementedError("Simplified DDNM+ not implemented")
            x, x_all = simplified_ddnm_plus(
                noise, net, net.betas, other_args.get('eta', 1.0),
                A_funcs, y, cls_fn=None, classes=labels,
                num_steps=num_steps
            )
        else:
            x, x_all = svd_based_ddnm_plus(
                noise, net, net.betas, other_args.get('eta', 1.0),
                A_funcs, y, sigma_y, cls_fn=None, classes=labels,
                num_steps=num_steps
            )
    
    return x, x_all, y_for_output

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def svd_based_ddnm_plus(noise, net, betas, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, num_steps=18):
    with torch.no_grad():
        # setup iteration variables
        betas = torch.tensor(betas).to(noise.device)
        skip = len(betas) // num_steps
        n = noise.size(0)
        x0_preds = []
        xs = [noise]

        # generate time schedule
        times = get_schedule_jump(num_steps, 
                                travel_length=1,  # Default values since they're not provided
                                travel_repeat=1)
        time_pairs = list(zip(times[:-1], times[1:]))        
        
        # reverse diffusion sampling
        for i, j in time_pairs:
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(noise.device)
                next_t = (torch.ones(n) * j).to(noise.device)
                at = compute_alpha(betas, t.long())
                at_next = compute_alpha(betas, next_t.long())
                xt = xs[-1]
                if cls_fn == None:
                    et = net.model(xt, t)
                else:
                    et = net.model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(noise, t, classes)

                if et.size(1) == 6: # removes the variance prediction output
                    et = et[:, :3]

                # Eq. 12
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                # Eq. 17
                x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                # Eq. 51
                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                x0_preds.append(x0_t)
                xs.append(xt_next)
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(noise.device)
                at_next = compute_alpha(betas, next_t.long())
                x0_t = x0_preds[-1]
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next)

    return xs[-1], [x0_preds[-1]]

def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)