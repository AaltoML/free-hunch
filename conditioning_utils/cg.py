import torch
import time


def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves a batch of matrix linear systems of the form

        A_i X_i = B_i,  i=1,...,K,

    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, torch.max(residual_norm-stopping_matrix),
                    1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))


    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info

def cg(A_mm, b, M_mm=None, x0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a PD matrix linear system using the preconditioned CG algorithm.

    This function solves a matrix linear system of the form

        A x = b,

    where A is an n x n positive definite matrix and b is an n-dimensional vector,
    and x is the n-dimensional vector representing the solution.

    Args:
        A_mm: A callable that performs a matrix multiply of A and an n-dimensional vector.
        b: An n-dimensional vector representing the right hand side.
        M_mm: (optional) A callable that performs a matrix multiply of the preconditioning
            matrix M and an n-dimensional vector. (default=identity matrix)
        x0: (optional) Initial guess for x, defaults to M_mm(b). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    n = b.shape[0]

    if M_mm is None:
        M_mm = lambda x: x
    if x0 is None:
        x0 = M_mm(b)
    if maxiter is None:
        maxiter = 5 * n

    assert b.shape == (n,)
    assert x0.shape == (n,)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    # x_k = x0
    # r_k = b - A_mm(x_k)
    # z_k = M_mm(r_k)

    # p_k = torch.zeros_like(z_k)

    # b_norm = torch.norm(b)
    # stopping_criterion = max(rtol * b_norm, atol)

    # if verbose:
    #     print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    # optimal = False
    # start = time.perf_counter()
    # for k in range(1, maxiter + 1):
    #     start_iter = time.perf_counter()
    #     z_k = M_mm(r_k)

    #     if k == 1:
    #         p_k = z_k
    #     else:
    #         beta = torch.dot(r_k, z_k) / torch.dot(r_k_prev, z_k_prev)
    #         p_k = z_k + beta * p_k

    #     Ap_k = A_mm(p_k)
    #     alpha = torch.dot(r_k, z_k) / torch.dot(p_k, Ap_k)
    #     x_k = x_k + alpha * p_k
    #     r_k_prev, z_k_prev = r_k, z_k
    #     r_k = r_k - alpha * Ap_k
    #     end_iter = time.perf_counter()

    #     residual_norm = torch.norm(r_k)

    #     if verbose:
    #         print("%03d | %8.4e %4.2f" %
    #               (k, residual_norm - stopping_criterion,
    #                1. / (end_iter - start_iter)))

    # x_k = x0
    # r_k = b - A_mm(x_k)
    # z_k = M_mm(r_k)
    # p_k = z_k.clone()  # Initialize p_k to z_k

    # b_norm = torch.norm(b)
    # stopping_criterion = max(rtol * b_norm, atol)

    # if verbose:
    #     print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    # optimal = False
    # start = time.perf_counter()
    # for k in range(1, maxiter + 1):
    #     start_iter = time.perf_counter()

    #     Ap_k = A_mm(p_k)
    #     alpha = torch.dot(r_k, z_k) / torch.dot(p_k, Ap_k)
    #     x_k = x_k + alpha * p_k
    #     r_k_new = r_k - alpha * Ap_k
        
    #     residual_norm = torch.norm(r_k_new)
        
    #     if residual_norm <= stopping_criterion:
    #         optimal = True
    #         break

    #     z_k_new = M_mm(r_k_new)
    #     beta = torch.dot(r_k_new, z_k_new) / torch.dot(r_k, z_k)
    #     p_k = z_k_new + beta * p_k

    #     r_k, z_k = r_k_new, z_k_new
    #     end_iter = time.perf_counter()

    #     if verbose:
    #         print("%03d | %8.4e %4.2f" %
    #               (k, residual_norm - stopping_criterion,
    #                1. / (end_iter - start_iter)))

    x_k = x0
    r_k = b - A_mm(x_k)
    z_k = M_mm(r_k)
    p_k = z_k.clone()  # Initialize p_k to z_k

    r_norm = torch.norm(r_k)
    b_norm = torch.norm(b)
    stopping_criterion = max(rtol * b_norm, atol)

    if verbose:
        print("%03s | %010s %06s" % ("it", "residual", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()

        Ap_k = A_mm(p_k)
        pAp = torch.dot(p_k, Ap_k)
        
        if pAp <= 1e-16:  # Check for numerical issues
            break
        
        alpha = torch.dot(r_k, z_k) / pAp
        x_k = x_k + alpha * p_k
        r_k_new = r_k - alpha * Ap_k
        
        r_norm_new = torch.norm(r_k_new)
        
        if r_norm_new <= stopping_criterion:
            optimal = True
            r_k = r_k_new
            r_norm = r_norm_new
            break

        z_k_new = M_mm(r_k_new)
        beta = torch.dot(r_k_new, z_k_new) / torch.dot(r_k, z_k)
        p_k = z_k_new + beta * p_k

        r_k, z_k, r_norm = r_k_new, z_k_new, r_norm_new
        end_iter = time.perf_counter()

        if verbose:
            print("%03d | %8.4e %4.2f" %
                  (k, r_norm, 1. / (end_iter - start_iter)))

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))

    info = {
        "niter": k,
        "optimal": optimal,
        "residual_norm": r_norm
    }

    return x_k, info

class CG(torch.autograd.Function):

    def __init__(self, A_bmm, M_bmm=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
        self.A_bmm = A_bmm
        self.M_bmm = M_bmm
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.verbose = verbose

    def forward(self, B, X0=None):
        X, _ = cg_batch(self.A_bmm, B, M_bmm=self.M_bmm, X0=X0, rtol=self.rtol,
                     atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return X

    def backward(self, dX):
        dB, _ = cg_batch(self.A_bmm, dX, M_bmm=self.M_bmm, rtol=self.rtol,
                      atol=self.atol, maxiter=self.maxiter, verbose=self.verbose)
        return dB