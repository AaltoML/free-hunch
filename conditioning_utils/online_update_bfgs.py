import torch
import numpy as np
from scipy.linalg import sqrtm
import os
import torch_dct

class CovarianceHessianBFGS:
    """A class for storing the covariance and hessian and their inverses in a memory-efficient format, and for updating them when moving in x or diffusion time t.
    Uses the BFGS update for the covariance, and derives the Hessian update from that. The BFGS update necessitates a data representation where
    the covariance matrix is represented as diag + UU^T - VV^T. All the linear algebra operators are then implemented for this data representation."""

    def __init__(self, init_denoiser_variance, init_noise_variance, data_dim, dtype=torch.complex128,
                 max_vector_count=None, init_denoiser_cov_u=None, project_to_diagonal=False, use_precalculated_info=True):
        #self.denoiser_variance = init_variance # the variance of the diagonal denoiser covariance matrix
        # init_noise_variance is either a number or a (N,) vector, specifying the noise variance for each dimension
        if init_denoiser_cov_u is None:
            init_denoiser_cov_u = torch.zeros(data_dim, 0, dtype=dtype)
        self.vectors_denoiser_cov_u = init_denoiser_cov_u
        self.vectors_inv_denoiser_cov_u = torch.zeros(data_dim, 0, dtype=dtype)
        self.vectors_hessian_u = torch.zeros(data_dim, 0, dtype=dtype)
        self.vectors_inv_hessian_u = torch.zeros(data_dim, 0, dtype=dtype)
        # The u-v distinction is relevant for the BFGS data representation. U-vectors are the positive low-rank updates, V-vectors are the negative low-rank updates
        self.vectors_denoiser_cov_v = torch.zeros(data_dim, 0, dtype=dtype)
        self.vectors_inv_denoiser_cov_v = torch.zeros(data_dim, 0, dtype=dtype)
        self.vectors_hessian_v = torch.zeros(data_dim, 0, dtype=dtype)
        self.vectors_inv_hessian_v = torch.zeros(data_dim, 0, dtype=dtype)
        self.diagonal_denoiser_cov = torch.ones(data_dim, dtype=dtype) * init_denoiser_variance
        self.diagonal_inv_denoiser_cov = 1/self.diagonal_denoiser_cov
        self.diagonal_hessian = (init_denoiser_variance/init_noise_variance - 1)/init_noise_variance * torch.ones(data_dim, dtype=dtype)
        self.diagonal_inv_hessian = 1/self.diagonal_hessian

        # these are used to store the raw information from the recent steps, so that we can drop out older vectors
        self.raw_added_vectors_denoiser_cov_u = torch.zeros(data_dim, 0, dtype=dtype)
        self.raw_added_vectors_denoiser_cov_v = torch.zeros(data_dim, 0, dtype=dtype)

        # this in case the init_denoiser_cov_u is not None
        self.set_others_corresponding_to_current_denoiser_cov(np.sqrt(init_noise_variance))

        # Also store these vectors on CUDA (even though main calculations are on CPU)
        self.cuda_diagonal_denoiser_cov = self.diagonal_denoiser_cov.cuda()
        self.cuda_diagonal_inv_denoiser_cov = self.diagonal_inv_denoiser_cov.cuda()
        self.cuda_diagonal_hessian = self.diagonal_hessian.cuda()
        self.cuda_diagonal_inv_hessian = self.diagonal_inv_hessian.cuda()
        self.cuda_vectors_denoiser_cov_u = self.vectors_denoiser_cov_u.cuda()
        self.cuda_vectors_inv_denoiser_cov_u = self.vectors_inv_denoiser_cov_u.cuda()
        self.cuda_vectors_hessian_u = self.vectors_hessian_u.cuda()
        self.cuda_vectors_inv_hessian_u = self.vectors_inv_hessian_u.cuda()
        self.cuda_vectors_denoiser_cov_v = self.vectors_denoiser_cov_v.cuda()
        self.cuda_vectors_inv_denoiser_cov_v = self.vectors_inv_denoiser_cov_v.cuda()
        self.cuda_vectors_hessian_v = self.vectors_hessian_v.cuda()
        self.cuda_vectors_inv_hessian_v = self.vectors_inv_hessian_v.cuda()

        self.data_dim = data_dim
        self.dtype = dtype
        self.max_vector_count = max_vector_count
        self.project_to_diagonal = project_to_diagonal

    def to_complex(self, x):
        return x.to(self.dtype)
        # if x.dtype == torch.float32:
        #     return x.to(torch.complex64)
        # elif x.dtype == torch.float64:
        #     return x.to(torch.complex128)
        # else:
        #     return x

    def sqrtm(self, A):
        if A.shape[0] != 0 and A.shape[1] != 0:
            return torch.from_numpy(sqrtm(A.numpy()).astype(np.complex128)).to(A.dtype)
        else:
            return torch.zeros_like(A)

    def update_cuda_vectors(self):
        self.cuda_diagonal_denoiser_cov = self.diagonal_denoiser_cov.cuda()
        self.cuda_diagonal_inv_denoiser_cov = self.diagonal_inv_denoiser_cov.cuda()
        self.cuda_diagonal_hessian = self.diagonal_hessian.cuda()
        self.cuda_diagonal_inv_hessian = self.diagonal_inv_hessian.cuda()
        self.cuda_vectors_denoiser_cov_u = self.vectors_denoiser_cov_u.cuda()
        self.cuda_vectors_inv_denoiser_cov_u = self.vectors_inv_denoiser_cov_u.cuda()
        self.cuda_vectors_hessian_u = self.vectors_hessian_u.cuda()
        self.cuda_vectors_inv_hessian_u = self.vectors_inv_hessian_u.cuda()
        self.cuda_vectors_denoiser_cov_v = self.vectors_denoiser_cov_v.cuda()
        self.cuda_vectors_inv_denoiser_cov_v = self.vectors_inv_denoiser_cov_v.cuda()
        self.cuda_vectors_hessian_v = self.vectors_hessian_v.cuda()
        self.cuda_vectors_inv_hessian_v = self.vectors_inv_hessian_v.cuda()

    def woodbury_inverse_from_diag_plus_lowrank(self, diag_inv, U):
        # calculate (diag + W W^T)^-1 = diag_inv - diag_inv W (I + W^T diag_inv W)^-1 W^T diag_inv
        # .. assuming that we have the diagonal inverse. 
        # returns V_inv, because of the negative sign of the update
        
        # shapes: 
        # diag_inv: (d) (corresponds to a (d,d) diagonal matrix)
        # U: (d, k) (corresponds to a (d,k) matrix)
        k = U.shape[-1]
        inner_inv = torch.linalg.inv(torch.eye(k) + U.T @ (diag_inv[...,None] * U))
        # make sure that symmetric
        inner_inv = (inner_inv + inner_inv.T) / 2
        
        # There is no guarantee that inner_inv is positive definite, so we cannot use Cholesky decomposition
        inner_inv_sqrt = self.sqrtm(inner_inv)
        
        V_inv = diag_inv[...,None] * (U @ inner_inv_sqrt)
        return V_inv

    def woodbury_inverse_from_diag_plus_lowrank_minus_lowrank(self, U, V, diag):
        diag_inv = 1/diag
        # Calculate A=(diag + UU^T)^-1 in format diag_inv - V_inv V_inv^T
        V_inv = self.woodbury_inverse_from_diag_plus_lowrank(diag_inv, U)
        # calculate (A - VV^T)^-1 in format A_inv + U_inv U_inv^T
        # (A - VV^T)^-1 = A^-1 + A^-1 V (I - V^T A^-1 V)^-1 V^T A^-1 
        # first calculate (I - V^T A^-1 V)^-1 = (I - V^T A_diag^-1 V + V^T V_inv V_inv^T V)^-1
        k = V.shape[-1]
        K = V_inv.T @ V
        inner_inv = torch.linalg.inv(torch.eye(k) - V.T @ (diag_inv[...,None] * V) + K.T @ K)
        inner_inv_sqrt = self.sqrtm(inner_inv)
        V_inner_inv_sqrt = V @ inner_inv_sqrt
        U_inv = diag_inv[...,None] * (V_inner_inv_sqrt) - V_inv @ (V_inv.T @ V_inner_inv_sqrt)
        return diag_inv, U_inv, V_inv
    
    def sherman_morrison_update(self, U, V, diag, v, pos):
        """ pos = True if we are adding a positive update, False if we are adding a negative update
         implements the Sherman-Morrison formula for computing (A+vv^T)^-1 or (A-vv^T)^-1, given that we have A^{-1}. 
         A^{-1}= diag + UU^T - VV^T, wiht the update being (A+vv^T)^-1 or (A-vv^T)^-1. 
         diag is the diagonal of the matrix, U and V are the U and V vectors of the original inverse matrix, 
         v is the update vector, and pos is a boolean indicating whether the update is positive or negative"""
        if pos:
            denominator = 1 + v.T @ (U @ (U.T @ v) - V @ (V.T @ v) + diag * v)
            if denominator <= 0:
                u_update = (U @ (U.T @ v) - V @ (V.T @ v) + diag * v) / (-denominator).sqrt()
                return torch.cat((U, u_update[:,None]), dim=-1), V, diag
            else:
                v_update = (U @ (U.T @ v) - V @ (V.T @ v) + diag * v) / denominator.sqrt()
                return U, torch.cat((V, v_update[:,None]), dim=-1), diag
        else:
            denominator = 1 - v.T @ (U @ (U.T @ v) - V @ (V.T @ v) + diag * v)
            if denominator <= 0:
                v_update = (U @ (U.T @ v) - V @ (V.T @ v) + diag * v) / (-denominator).sqrt()
                return U, torch.cat((V, v_update[:,None]), dim=-1), diag
            else:
                u_update = (U @ (U.T @ v) - V @ (V.T @ v) + diag * v) / denominator.sqrt()
                return torch.cat((U, u_update[:,None]), dim=-1), V, diag

    def sherman_morrison_double_update(self, U, V, diag, u, v):
        """ u is the positive update, v is the negative update
         (A + uu^T - vv^T)^-1
         first calculates (A + uu^T)^-1
         then calculates (A + uu^T - vv^T)^-1 based on that"""
        U_updated, V_updated, diag_updated = self.sherman_morrison_update(U, V, diag, u, True)
        U_updated, V_updated, diag_updated = self.sherman_morrison_update(U_updated, V_updated, diag_updated, v, False)
        return U_updated, V_updated, diag_updated

    def update_time_step(self, x_t, sigma_t, sigma_tnext, score_t, **kwargs):
        new_denoiser_mean, new_score_value = self.update_time_step_(x_t, sigma_t, sigma_tnext, score_t, **kwargs)
        return new_denoiser_mean.real, new_score_value.real

    def update_time_step_(self, x_t, sigma_t, sigma_tnext, score_t, only_covariance=False):
        """Assumes that score_t has batch size 1, and there is no batch dimension"""
        # update the inverse denoiser covariance (only diagonal term necessary to change)        
        shape = x_t.shape # e.g., (bs, C, H, W)
        assert shape[0] == 1, "Batch size must be 1"
        x_t = self.to_complex(x_t.detach().cpu()).reshape(-1) # flatten to (C*H*W)
        score_t = self.to_complex(score_t.detach().cpu()).reshape(-1) # flatten to (C*H*W)
        
        k = self.vectors_denoiser_cov_u.shape[-1]
        self.diagonal_inv_denoiser_cov = self.diagonal_inv_denoiser_cov + (sigma_tnext**(-2) - sigma_t**(-2)) * torch.ones(self.data_dim)
        self.diagonal_denoiser_cov, self.vectors_denoiser_cov_u, self.vectors_denoiser_cov_v = self.woodbury_inverse_from_diag_plus_lowrank_minus_lowrank(self.vectors_inv_denoiser_cov_u, 
                                                                                                                                self.vectors_inv_denoiser_cov_v, self.diagonal_inv_denoiser_cov)
        
        if not only_covariance: # optimization in case we don't need the denoiser mean updates
            # Then the hessian...
            new_diagonal_inv_hessian = self.diagonal_inv_hessian - (sigma_tnext**(2) - sigma_t**(2)) * torch.ones(self.data_dim)
            new_diag_hessian, new_u_hessian, new_v_hessian = self.woodbury_inverse_from_diag_plus_lowrank_minus_lowrank(self.vectors_inv_hessian_u, self.vectors_inv_hessian_v, new_diagonal_inv_hessian)

            # Score function at time t_next (new_hessian @ old_inv_hessian @ score_t)
            old_inv_hessian_score_t = self.diagonal_inv_hessian * score_t + self.vectors_inv_hessian_u @ (self.vectors_inv_hessian_u.T @ score_t) - self.vectors_inv_hessian_v @ (self.vectors_inv_hessian_v.T @ score_t)
            new_score_value = (new_diag_hessian * old_inv_hessian_score_t + new_u_hessian @ (new_u_hessian.T @ old_inv_hessian_score_t) - new_v_hessian @ (new_v_hessian.T @ old_inv_hessian_score_t)).real + 0j
            # Denoiser mean at time t_next
            new_denoiser_mean = (x_t + sigma_tnext**2 * new_score_value).real + 0j

            new_denoiser_mean = new_denoiser_mean.reshape(shape)
            new_score_value = new_score_value.reshape(shape)

            self.diagonal_inv_hessian = new_diagonal_inv_hessian
            self.diagonal_hessian, self.vectors_hessian_u, self.vectors_hessian_v  = new_diag_hessian, new_u_hessian, new_v_hessian
        else:
            new_denoiser_mean = (x_t + 0j).reshape(shape)
            new_score_value = (x_t + 0j).reshape(shape)

        self.update_cuda_vectors()

        return new_denoiser_mean, new_score_value
    
    def _denoiser_cov_vector_dot(self, v, use_cuda=False):
        if use_cuda:
            return (self.cuda_diagonal_denoiser_cov * v + self.cuda_vectors_denoiser_cov_u @ (self.cuda_vectors_denoiser_cov_u.T @ v) - self.cuda_vectors_denoiser_cov_v @ (self.cuda_vectors_denoiser_cov_v.T @ v))
        else:
            return (self.diagonal_denoiser_cov * v + self.vectors_denoiser_cov_u @ (self.vectors_denoiser_cov_u.T @ v) - self.vectors_denoiser_cov_v @ (self.vectors_denoiser_cov_v.T @ v))

    def denoiser_cov_vector_dot(self, v, use_cuda=False):
        dtype = v.dtype
        shape = v.shape # e.g., (bs, C, H, W)
        v = self.to_complex(v).reshape(-1) # flatten to (C*H*W)
        return self._denoiser_cov_vector_dot(v, use_cuda).real.reshape(shape).to(dtype)
    
    def _inv_denoiser_cov_vector_dot(self, v):
        return (self.diagonal_inv_denoiser_cov * v + self.vectors_inv_denoiser_cov_u @ (self.vectors_inv_denoiser_cov_u.T @ v) - self.vectors_inv_denoiser_cov_v @ (self.vectors_inv_denoiser_cov_v.T @ v))

    def inv_denoiser_cov_vector_dot(self, v):
        dtype = v.dtype
        shape = v.shape # e.g., (bs, C, H, W)   
        v = self.to_complex(v).reshape(-1) # flatten to (C*H*W)
        return self._inv_denoiser_cov_vector_dot(v).real.reshape(shape).to(dtype)
    
    def _hessian_vector_dot(self, v):
        return (self.diagonal_hessian * v + self.vectors_hessian_u @ (self.vectors_hessian_u.T @ v) - self.vectors_hessian_v @ (self.vectors_hessian_v.T @ v))

    def hessian_vector_dot(self, v):
        dtype = v.dtype
        shape = v.shape # e.g., (bs, C, H, W)
        v = self.to_complex(v).reshape(-1) # flatten to (C*H*W)
        return self._hessian_vector_dot(v).real.reshape(shape).to(dtype)
    
    def _inv_hessian_vector_dot(self, v):
        return (self.diagonal_inv_hessian * v + self.vectors_inv_hessian_u @ (self.vectors_inv_hessian_u.T @ v) - self.vectors_inv_hessian_v @ (self.vectors_inv_hessian_v.T @ v))

    def inv_hessian_vector_dot(self, v):
        dtype = v.dtype
        shape = v.shape # e.g., (bs, C, H, W)
        v = self.to_complex(v).reshape(-1) # flatten to (C*H*W)
        return self._inv_hessian_vector_dot(v).real.reshape(shape).to(dtype)
    
    def drop_vectors(self, max_vector_count, sigma):
        # only keep the last max_vector_count vectors
        if max_vector_count == 0:
            dtype, device = self.vectors_denoiser_cov_u.dtype, self.vectors_denoiser_cov_u.device
            self.vectors_denoiser_cov_u = torch.zeros(self.data_dim, 0, dtype=dtype, device=device)
            self.vectors_denoiser_cov_v = torch.zeros(self.data_dim, 0, dtype=dtype, device=device)
            self.set_others_corresponding_to_current_denoiser_cov(sigma)
        else:
            if self.vectors_denoiser_cov_u.shape[-1] > max_vector_count:
                total_added_vectors = self.vectors_denoiser_cov_u.shape[-1]
                self.vectors_denoiser_cov_u = self.vectors_denoiser_cov_u[:,-min(total_added_vectors, max_vector_count):]
                self.vectors_denoiser_cov_v = self.vectors_denoiser_cov_v[:,-min(total_added_vectors, max_vector_count):]
                self.set_others_corresponding_to_current_denoiser_cov(sigma)
        
    def update_space_step(self, denoiser_mean_at_x, denoiser_mean_at_xnext, sigma_t, x, xnext):
        return self.update_space_step_(denoiser_mean_at_x, denoiser_mean_at_xnext, sigma_t, x, xnext)

    def update_space_step_(self, denoiser_mean_at_x, denoiser_mean_at_xnext, sigma_t, x, xnext):
        """BFGS update of the denoiser covariance and hessian and the inverses"""
        # update the denoiser covariance and hessian

        shape = x.shape # e.g., (bs, C, H, W)
        assert shape[0] == 1, "Batch size must be 1"
        x = self.to_complex(x.detach().cpu()).reshape(-1) # flatten to (C*H*W)
        xnext = self.to_complex(xnext.detach().cpu()).reshape(-1) # flatten to (C*H*W)
        denoiser_mean_at_x = self.to_complex(denoiser_mean_at_x.detach().cpu()).reshape(-1) # flatten to (C*H*W)
        denoiser_mean_at_xnext = self.to_complex(denoiser_mean_at_xnext.detach().cpu()).reshape(-1) # flatten to (C*H*W)

        dx = xnext - x
        de = sigma_t**2 * (denoiser_mean_at_xnext - denoiser_mean_at_x)
        gamma = 1/(dx @ de)

        # Update the denoiser covariance
        # The maths: Dcov -> DCov - DCov @ dx @ dx.T @ DCov / (dx.T @ DCov @ dx) + de @ de.T * gamma
        # need to calculate DCov @ dx (in the form of diag + UU^T - VV^T)
        denoiser_cov_dot_dx = self._denoiser_cov_vector_dot(dx)
        # then dx_dot_denoiser_cov_dot_dx = dx_dot_denoiser_cov @ dx
        dx_dot_denoiser_cov_dot_dx = denoiser_cov_dot_dx @ dx
        # then we're ready for the denoiser covariance update
        v = denoiser_cov_dot_dx / torch.sqrt(dx_dot_denoiser_cov_dot_dx)
        u = de * torch.sqrt(gamma)
        if self.project_to_diagonal:
            new_diagonal_denoiser_cov = self.diagonal_denoiser_cov + u * u - v * v
            new_vectors_denoiser_cov_u = self.vectors_denoiser_cov_u
            new_vectors_denoiser_cov_v = self.vectors_denoiser_cov_v
        else:
            new_diagonal_denoiser_cov = self.diagonal_denoiser_cov
            new_vectors_denoiser_cov_u = torch.cat((self.vectors_denoiser_cov_u, u[:,None]), dim=-1)
            new_vectors_denoiser_cov_v = torch.cat((self.vectors_denoiser_cov_v, v[:,None]), dim=-1)
            # Keeping track of these in case we need to drop vectors
            self.raw_added_vectors_denoiser_cov_u = torch.cat((self.raw_added_vectors_denoiser_cov_u, u[:,None]), dim=-1)
            self.raw_added_vectors_denoiser_cov_v = torch.cat((self.raw_added_vectors_denoiser_cov_v, v[:,None]), dim=-1)

        
        # Update the inverse denoiser covariance
        # Use the Woodbury identity for simplicity, could get more efficiency with Sherman-Morrison updates
        new_diagonal_inv_denoiser_cov, new_vectors_inv_denoiser_cov_u, new_vectors_inv_denoiser_cov_v = self.woodbury_inverse_from_diag_plus_lowrank_minus_lowrank(U=new_vectors_denoiser_cov_u, 
                                                                                V=new_vectors_denoiser_cov_v, diag=new_diagonal_denoiser_cov)
        
        # Update the Hessian based on the denoiser covariance update
        # H = (Dcov/sigma^2 - I)/sigma^2
        new_diagonal_hessian = (new_diagonal_denoiser_cov / sigma_t**2 - torch.ones(self.data_dim)) / sigma_t**2
        u_hessian = u / sigma_t**2
        v_hessian = v / sigma_t**2
        new_vectors_hessian_u = torch.cat((self.vectors_hessian_u, u_hessian[:,None]), dim=-1)
        new_vectors_hessian_v = torch.cat((self.vectors_hessian_v, v_hessian[:,None]), dim=-1)
        
        new_diagonal_inv_hessian, new_vectors_inv_hessian_u, new_vectors_inv_hessian_v = self.woodbury_inverse_from_diag_plus_lowrank_minus_lowrank(U=new_vectors_hessian_u, 
                                                                                V=new_vectors_hessian_v, diag=new_diagonal_hessian)
        
        # Apply all the updates
        self.diagonal_denoiser_cov, self.vectors_denoiser_cov_u, self.vectors_denoiser_cov_v = new_diagonal_denoiser_cov, new_vectors_denoiser_cov_u, new_vectors_denoiser_cov_v
        self.diagonal_inv_denoiser_cov, self.vectors_inv_denoiser_cov_u, self.vectors_inv_denoiser_cov_v = new_diagonal_inv_denoiser_cov, new_vectors_inv_denoiser_cov_u, new_vectors_inv_denoiser_cov_v
        self.diagonal_hessian, self.vectors_hessian_u, self.vectors_hessian_v  = new_diagonal_hessian, new_vectors_hessian_u, new_vectors_hessian_v
        self.diagonal_inv_hessian, self.vectors_inv_hessian_u, self.vectors_inv_hessian_v = new_diagonal_inv_hessian, new_vectors_inv_hessian_u, new_vectors_inv_hessian_v

        if self.max_vector_count is not None:
            self.drop_vectors(self.max_vector_count, sigma_t)

        self.update_cuda_vectors()

    def transform(self,x):
        return x
    
    def inverse_transform(self,x):
        return x

    def get_dense_matrices(self):
        denoiser_cov = self.diagonal_denoiser_cov[:,None] * torch.eye(self.data_dim) + self.vectors_denoiser_cov_u @ self.vectors_denoiser_cov_u.T - self.vectors_denoiser_cov_v @ self.vectors_denoiser_cov_v.T
        inv_denoiser_cov = self.diagonal_inv_denoiser_cov[:,None] * torch.eye(self.data_dim) + self.vectors_inv_denoiser_cov_u @ self.vectors_inv_denoiser_cov_u.T - self.vectors_inv_denoiser_cov_v @ self.vectors_inv_denoiser_cov_v.T
        hessian = self.diagonal_hessian[:,None] * torch.eye(self.data_dim) + self.vectors_hessian_u @ self.vectors_hessian_u.T - self.vectors_hessian_v @ self.vectors_hessian_v.T
        inv_hessian = self.diagonal_inv_hessian[:,None] * torch.eye(self.data_dim) + self.vectors_inv_hessian_u @ self.vectors_inv_hessian_u.T - self.vectors_inv_hessian_v @ self.vectors_inv_hessian_v.T
        return denoiser_cov, inv_denoiser_cov, hessian, inv_hessian

    def set_others_corresponding_to_current_denoiser_cov(self, sigma):
        self.diagonal_inv_denoiser_cov, self.vectors_inv_denoiser_cov_u, self.vectors_inv_denoiser_cov_v = self.woodbury_inverse_from_diag_plus_lowrank_minus_lowrank(self.vectors_denoiser_cov_u, self.vectors_denoiser_cov_v, self.diagonal_denoiser_cov)
        self.diagonal_hessian, self.vectors_hessian_u, self.vectors_hessian_v = (self.diagonal_denoiser_cov/sigma**2 - 1)/sigma**2, self.vectors_denoiser_cov_u/sigma**2, self.vectors_denoiser_cov_v/sigma**2
        self.diagonal_inv_hessian, self.vectors_inv_hessian_u, self.vectors_inv_hessian_v = self.woodbury_inverse_from_diag_plus_lowrank_minus_lowrank(self.vectors_hessian_u, self.vectors_hessian_v, self.diagonal_hessian)
    
    def zero_other_channels(self, x, channel_to_keep):
        x = x.clone()
        dims_not_to_keep = [i for i in range(x.shape[1]) if i != channel_to_keep]
        x[:, dims_not_to_keep] = 0
        return x


class CovarianceHessianBFGSDCT(CovarianceHessianBFGS):
    # A wrapper around CovarianceHessianBFGS that makes it perform all the operations in the DCT basis
    def __init__(self, data_dir, init_noise_variance, data_dim,
                 dtype=torch.complex128, max_vector_count=None, **kwargs):
        dct_variance = torch.load(os.path.join(data_dir, 'dct_variance.pt'))
        # dct_variance = dict_['dct_variance']
        if kwargs['use_precalculated_info']:
            self.dct_variance = dct_variance
        else:
            self.dct_variance = torch.ones(data_dim)
        super().__init__(self.dct_variance.reshape(-1), init_noise_variance, data_dim, dtype, max_vector_count, **kwargs)
    
    def transform(self,x):
        return torch_dct.dct_2d(x, norm='ortho')
    
    def inverse_transform(self,x):
        return torch_dct.idct_2d(x, norm='ortho')

    def update_time_step(self, x_t, sigma_t, sigma_tnext, score_t, **kwargs):
        new_denoiser_mean, new_score_value = self.update_time_step_(torch_dct.dct_2d(x_t, norm='ortho'), sigma_t, sigma_tnext, torch_dct.dct_2d(score_t, norm='ortho'), **kwargs)
        new_denoiser_mean, new_score_value = new_denoiser_mean.real, new_score_value.real
        return torch_dct.idct_2d(new_denoiser_mean, norm='ortho'), torch_dct.idct_2d(new_score_value, norm='ortho')
    
    def update_space_step(self, denoiser_mean_at_x, denoiser_mean_at_xnext, sigma_t, x, xnext):
        meanx = torch_dct.dct_2d(denoiser_mean_at_x, norm='ortho')
        meanxnext = torch_dct.dct_2d(denoiser_mean_at_xnext, norm='ortho')
        x = torch_dct.dct_2d(x, norm='ortho')
        xnext = torch_dct.dct_2d(xnext, norm='ortho')

        self.update_space_step_(meanx, meanxnext, sigma_t, x, xnext)
    
    def denoiser_cov_vector_dot(self, v, use_cuda=False):
        dtype = v.dtype
        shape = v.shape # e.g., (bs, C, H, W)
        v = self.to_complex(torch_dct.dct_2d(v, norm='ortho')).reshape(-1) # flatten to (C*H*W)
        return torch_dct.idct_2d(self._denoiser_cov_vector_dot(v, use_cuda).real.reshape(shape).to(dtype), norm='ortho')


def update_covariance(samples, denoiser_cov, inv_denoiser_cov, hessian, inv_hessian, score_value, denoiser_mean, schedule, t, tnext):
    """
    Update the dense forms of the denoiser covariance, hessian, score function, 
    and denoiser mean for a batch of samples at a new time step using a 
    Gaussian approximation of the noisy distribution.

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
    dim = samples.shape[-1]
    
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

def update_bfgs(denoiser_cov, inv_denoiser_cov, denoiser_mean_at_x, denoiser_mean_at_xnext, schedule, t, x, dx):
    """
    Do the "space" update step of the BFGS approximation of the Hessian 
    and related quantities, for dense matrix formats. 

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
    de = schedule(t)**2 * (denoiser_mean_at_xnext - denoiser_mean_at_x) # shape (bs, d)
    
    gamma = 1/(dx[...,None,:] @ de[...,:,None])
    
    updated_denoiser_cov = denoiser_cov - denoiser_cov @ dx[...,:,None] @ dx[...,None,:] @ denoiser_cov / (dx[...,None,:] @ denoiser_cov @ dx[...,:,None]) + de[...,:,None] @ de[...,None,:] * gamma
    updated_inv_denoiser_cov = (I - dx[...,:,None] @ de[...,None,:] * gamma) @ inv_denoiser_cov @ (I - de[...,:,None] @ dx[...,None,:] * gamma) + dx[...,:,None] @ dx[...,None,:] * gamma
    
    updated_hessian = (updated_denoiser_cov/schedule(t)**2 - I)/schedule(t)**2
    updated_inv_hessian = torch.linalg.inv(updated_hessian + 1e-10*torch.eye(d).unsqueeze(0).repeat(bs, 1, 1))# add a jitter term to make it invertible
    
    return updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian
