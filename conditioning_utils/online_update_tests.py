from conditioning_utils.online_update_bfgs import *

def test_covariance_hessian_time_update():
    """Test the time update step of the covariance and Hessian matrices.
    
    This test verifies that the dense matrix update equations match the memory-efficient 
    BFGS update formulation when performing a time update step. Specifically, it:

    1. Initializes dense covariance/Hessian matrices and their memory-efficient BFGS representations
    2. Performs a time update step using both formulations
    3. Compares the results to ensure they match within numerical precision

    The test uses a simple linear score function and checks:
    - Denoiser covariance matrix updates
    - Inverse denoiser covariance matrix updates  
    - Hessian matrix updates
    - Inverse Hessian matrix updates

    A small error in the matrix norms indicates the two formulations are equivalent.
    """
    # Define dimensions and batch size
    d = 5
    bs = 1

    # Define a simple score function (e.g., linear function)
    def score_fn(x, t):
        return -x / (t ** 2)

    # Define schedule
    def schedule(t):
        return t

    # Initialize parameters
    t = 20.0
    tnext = 18.0
    x = torch.randn(d)

    # Initialize dense matrices
    denoiser_cov = torch.eye(d)
    inv_denoiser_cov = torch.eye(d)
    hessian = (denoiser_cov/schedule(t)**2 - torch.eye(d))/schedule(t)**2
    inv_hessian = torch.linalg.inv(hessian)

    # Initialize CovarianceHessianBFGS
    bfgs = CovarianceHessianBFGS(init_denoiser_variance=1, init_noise_variance=schedule(t)**2, data_dim=d)

    # Compute scores and denoiser means
    score_at_t = score_fn(x, t)
    score_at_tnext = score_fn(x, tnext)
    denoiser_mean_at_t = x + (schedule(t) ** 2) * score_at_t
    denoiser_mean_at_tnext = x + (schedule(tnext) ** 2) * score_at_tnext

    # Update dense matrices
    updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian, new_score_value, new_denoiser_mean = update_covariance(
        x, denoiser_cov, inv_denoiser_cov, hessian, inv_hessian,
        score_at_t, denoiser_mean_at_t,
        schedule, t, tnext
    )

    # Update BFGS representation
    bfgs.update_time_step(x, schedule(t), schedule(tnext), score_at_t)
    
    # Compare results
    bfgs_denoiser_cov, bfgs_inv_denoiser_cov, bfgs_hessian, bfgs_inv_hessian = bfgs.get_dense_matrices()

    print("Denoiser Covariance Error:", torch.norm(updated_denoiser_cov - bfgs_denoiser_cov).item())
    print("Inverse Denoiser Covariance Error:", torch.norm(updated_inv_denoiser_cov - bfgs_inv_denoiser_cov).item())
    print("Hessian Error:", torch.norm(updated_hessian - bfgs_hessian).item())
    print("Inverse Hessian Error:", torch.norm(updated_inv_hessian - bfgs_inv_hessian).item())

def test_covariance_hessian_time_update_with_u_and_v():
    """
    Test the time update of covariance and Hessian matrices with rank-k updates.
    
    This test verifies that the covariance and Hessian updates work correctly when the 
    initial covariance matrix is modified by adding/subtracting rank-1 updates of the form
    uu^T - vv^T. This simulates a more complex covariance structure than just the identity
    matrix.

    The test:
    1. Creates initial covariance matrices with varying numbers of rank-1 updates (1,2,4,8 pairs)
    2. Performs time updates on both the dense matrix representation and the BFGS representation
    3. Compares the results to ensure both approaches give matching outputs
    4. Ensures the positive definiteness of matrices is maintained throughout

    The u,v vectors are chosen to be nearly orthogonal and properly scaled to maintain
    positive definiteness of the covariance matrix.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    for num_u_v_pairs in [1,2,4,8]:
        # Define dimensions
        d = 15

        # Define a simple score function (e.g., linear function)
        def score_fn(x, t):
            return -x / (t ** 2)

        # Define schedule
        def schedule(t):
            return t

        dtype = torch.float32

        # Initialize parameters
        t = 80.0
        tnext = 79.0
        x = torch.randn(d, dtype=dtype)

        # Initialize dense matrices
        denoiser_cov = torch.eye(d, dtype=dtype)
        inv_denoiser_cov = torch.eye(d, dtype=dtype)
        hessian = (denoiser_cov/schedule(t)**2 - torch.eye(d, dtype=dtype))/schedule(t)**2
        inv_hessian = torch.linalg.inv(hessian)

        U = []
        V = []
        for _ in range(num_u_v_pairs):
            # Add vector outer products to denoiser covariance
            u = torch.randn(d, dtype=dtype)
            v = torch.randn(d, dtype=dtype)
            u = u / torch.norm(u)  # Normalize u
            v = v / torch.norm(v)  # Normalize v
        
            # Ensure positive definiteness by making u and v nearly orthogonal
            v = v - torch.dot(u, v) * u
            v = v / torch.norm(v) * np.sqrt(0.5)  # Renormalize v
        
            # Add uu^T - vv^T to denoiser_cov
            denoiser_cov += torch.outer(u, u) - torch.outer(v, v) 
            U.append(u[:,None])
            V.append(v[:,None])

        U = torch.cat(U, dim=1)
        V = torch.cat(V, dim=1)

        # assert that the denoiser covariance is positive definite
        # Check positive definiteness using Cholesky decomposition
        try:
            torch.linalg.cholesky(denoiser_cov)
        except RuntimeError:
            raise ValueError("The denoiser covariance matrix is not positive definite.")

        # Update inv_denoiser_cov and hessian accordingly
        inv_denoiser_cov = torch.linalg.inv(denoiser_cov)
        hessian = (denoiser_cov/schedule(t)**2 - torch.eye(d))/schedule(t)**2
        inv_hessian = torch.linalg.inv(hessian)

        # Initialize CovarianceHessianBFGS
        bfgs = CovarianceHessianBFGS(init_denoiser_variance=1, init_noise_variance=schedule(t)**2, data_dim=d)

        bfgs.vectors_denoiser_cov_u = U
        bfgs.vectors_denoiser_cov_v = V
        bfgs.set_others_corresponding_to_current_denoiser_cov(schedule(t))

        bfgs_denoiser_cov, bfgs_inv_denoiser_cov, bfgs_hessian, bfgs_inv_hessian = bfgs.get_dense_matrices()
        assert torch.norm(bfgs_denoiser_cov - denoiser_cov).item()/d**2 < 1e-8, "Reconstructed denoiser covariance does not match the original."
        assert torch.norm(bfgs_inv_denoiser_cov - inv_denoiser_cov).item()/d**2 < 1e-7, "Reconstructed inverse denoiser covariance does not match the original."
        assert torch.norm(bfgs_hessian - hessian).item()/d**2 < 1e-10, "Reconstructed Hessian does not match the original."
        assert torch.norm(bfgs_inv_hessian - inv_hessian).item()/d**2 < 1e-4, "Reconstructed inverse Hessian does not match the original."

        # Compute scores and denoiser means
        score_at_t = score_fn(x, t)
        score_at_tnext = score_fn(x, tnext)
        denoiser_mean_at_t = x + (schedule(t) ** 2) * score_at_t
        denoiser_mean_at_tnext = x + (schedule(tnext) ** 2) * score_at_tnext

        # Update dense matrices
        updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian, new_score_value, new_denoiser_mean = update_covariance(
            x, denoiser_cov, inv_denoiser_cov, hessian, inv_hessian,
            score_at_t, denoiser_mean_at_t,
            schedule, t, tnext
        )

        # Update BFGS representation
        bfgs.update_time_step(x, schedule(t), schedule(tnext), score_at_t)
        
        # Compare results
        bfgs_denoiser_cov, bfgs_inv_denoiser_cov, bfgs_hessian, bfgs_inv_hessian = bfgs.get_dense_matrices()

        print("----------------------------------")
        print("Results for num_u_v_pairs =", num_u_v_pairs)
        print("Denoiser Covariance Error:", torch.norm(updated_denoiser_cov - bfgs_denoiser_cov).item() / d**2)
        print("Inverse Denoiser Covariance Error:", torch.norm(updated_inv_denoiser_cov - bfgs_inv_denoiser_cov).item() / d**2)
        print("Hessian Error:", torch.norm(updated_hessian - bfgs_hessian).item() / d**2)
        print("Inverse Hessian Error:", torch.norm(updated_inv_hessian - bfgs_inv_hessian).item() / d**2)

def test_bfgs_update():
    """Test the BFGS update step of the covariance and Hessian matrices.
    
    This test verifies that the dense matrix update equations match the memory-efficient 
    BFGS update formulation when performing a BFGS update step. Specifically, it:

    1. Initializes dense covariance/Hessian matrices and their memory-efficient BFGS representations
    2. Takes multiple random steps and performs BFGS updates using both formulations
    3. Compares the results to ensure they match within numerical precision

    The test uses a mixture of linear score functions and checks:
    - Denoiser covariance matrix updates
    - Inverse denoiser covariance matrix updates  
    - Hessian matrix updates
    - Inverse Hessian matrix updates

    The test takes multiple steps to verify stability of the updates over time.
    A small error in the matrix norms indicates the two formulations are equivalent.
    """
    # Define dimensions
    d = 15

    # Define a simple score function (e.g., linear function)
    def score_fn(x, t):
        term1 = -x / (t ** 2)
        term2 = -0.5 * (x-torch.ones_like(x)) / (t ** 2)
        return 0.7 * term1 + 0.3 * term2

    # Define schedule
    def schedule(t):
        return t

    dtype = torch.float32

    # set random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # Initialize parameters
    t = 50.0
    x = torch.randn(d, dtype=dtype)

    # Initialize dense matrices
    denoiser_cov = torch.eye(d, dtype=dtype)
    inv_denoiser_cov = torch.eye(d, dtype=dtype)
    hessian = (denoiser_cov/schedule(t)**2 - torch.eye(d, dtype=dtype))/schedule(t)**2
    inv_hessian = torch.linalg.inv(hessian)

    # Initialize CovarianceHessianBFGS
    bfgs = CovarianceHessianBFGS(init_denoiser_variance=1, init_noise_variance=schedule(t)**2, data_dim=d)
    bfgs_denoiser_cov, bfgs_inv_denoiser_cov, bfgs_hessian, bfgs_inv_hessian = bfgs.get_dense_matrices()
    assert torch.norm(bfgs_denoiser_cov - denoiser_cov).item()/d**2 < 1e-8, "Reconstructed denoiser covariance does not match the original."
    assert torch.norm(bfgs_inv_denoiser_cov - inv_denoiser_cov).item()/d**2 < 1e-7, "Reconstructed inverse denoiser covariance does not match the original."
    assert torch.norm(bfgs_hessian - hessian).item()/d**2 < 1e-10, "Reconstructed Hessian does not match the original."
    assert torch.norm(bfgs_inv_hessian - inv_hessian).item()/d**2 < 1e-4, "Reconstructed inverse Hessian does not match the original."

    steps = 10

    for _ in range(steps):
        dx = torch.randn(d, dtype=dtype) * 0.1
        xnext = x + dx

        # Compute scores and denoiser means
        score_at_x = score_fn(x, t)
        score_at_xnext = score_fn(xnext, t)
        denoiser_mean_at_x = x + (schedule(t) ** 2) * score_at_x
        denoiser_mean_at_xnext = xnext + (schedule(t) ** 2) * score_at_xnext
        
        # compute bfgs update using the dense matrices
        updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian = update_bfgs(denoiser_cov, inv_denoiser_cov, denoiser_mean_at_x[None,:], denoiser_mean_at_xnext[None,:], schedule, t, x[None,:], dx[None,:])
        denoiser_cov, inv_denoiser_cov, hessian, inv_hessian = updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian

        # compute bfgs update using the bfgs representation
        bfgs.update_space_step(denoiser_mean_at_x, denoiser_mean_at_xnext, schedule(t), x, xnext)
        bfgs_denoiser_cov, bfgs_inv_denoiser_cov, bfgs_hessian, bfgs_inv_hessian = bfgs.get_dense_matrices()

        print("---------------BFGS update results-----------------")
        print("Denoiser Covariance Error:", torch.norm(updated_denoiser_cov - bfgs_denoiser_cov).item() / d**2)
        print("Inverse Denoiser Covariance Error:", torch.norm(updated_inv_denoiser_cov - bfgs_inv_denoiser_cov).item() / d**2)
        print("Hessian Error:", torch.norm(updated_hessian - bfgs_hessian).item() / d**2)
        print("Inverse Hessian Error:", torch.norm(updated_inv_hessian - bfgs_inv_hessian).item() / d**2)

        x = xnext

def test_time_and_space_updates():
    """Test both time and space updates of the covariance and Hessian matrices.
    
    This test verifies that both time updates and space updates work correctly when applied
    sequentially. It performs the following steps:

    1. Initializes matrices and BFGS representation
    2. For each time step:
        a. Performs a time update to move to the next timestep
        b. Performs a space update by taking a small step in the state space
        c. Verifies that dense and BFGS representations remain consistent
    
    The test uses a mixture of two Gaussian score functions and checks:
    - Score value updates through time
    - Denoiser mean updates through time  
    - Consistency between dense and BFGS matrix representations
    - Numerical stability across multiple update steps

    The test uses complex128 dtype for precision and stability.
    """
    # Set up parameters
    d = 5  # dimension
    dtype = torch.complex128
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    # Initialize parameters
    x = torch.randn(d, dtype=torch.float32).to(dtype)

    # Define score function and schedule
    def score_fn(x, t):
        term1 = -x / (t ** 2)
        term2 = -0.5 * (x-torch.ones_like(x)) / (t ** 2)
        return 0.7 * term1 + 0.3 * term2

    def schedule(t):
        return t

    ts = [50.0, 48.0, 46.0, 44.0, 42.0, 40.0, 38.0, 36.0, 34.0, 32.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.1]
    # ts = [50.0, 30.0, 10.0]
    t = ts[0]

    # Initialize dense matrices
    denoiser_cov = torch.eye(d, dtype=dtype)
    inv_denoiser_cov = torch.eye(d, dtype=dtype)
    hessian = (denoiser_cov/schedule(t)**2 - torch.eye(d, dtype=dtype))/schedule(t)**2
    inv_hessian = torch.linalg.inv(hessian)

    # Initialize CovarianceHessianBFGS
    bfgs = CovarianceHessianBFGS(init_denoiser_variance=1, init_noise_variance=schedule(t)**2, data_dim=d, dtype=dtype)

    for i in range(len(ts)-1):
        t = ts[i]
        tnext = ts[i+1]
        dx = torch.real(torch.randn(d, dtype=dtype) * 0.1).to(dtype)
        xnext = x + dx

        # Perform time update
        score_at_t = score_fn(x, t)
        denoiser_mean_at_t = x + (schedule(t) ** 2) * score_at_t

        updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian, new_score_value, new_denoiser_mean = update_covariance(
            x[None, :], denoiser_cov[None, :, :], inv_denoiser_cov[None, :, :], hessian[None, :, :], inv_hessian[None, :, :], 
            score_at_t[None, :], denoiser_mean_at_t[None, :], schedule, t, tnext
        )

        bfgs_new_denoiser_mean, bfgs_new_score_value = bfgs.update_time_step(x, schedule(t), schedule(tnext), score_at_t)

        print("Round ", i)
        print("---------------Time update results-----------------")
        print("Score Value Error:", torch.norm(new_score_value - bfgs_new_score_value).item() / d)
        print("Denoiser Mean Error:", torch.norm(new_denoiser_mean - bfgs_new_denoiser_mean).item() / d)
        bfgs_denoiser_cov, bfgs_inv_denoiser_cov, bfgs_hessian, bfgs_inv_hessian = bfgs.get_dense_matrices()
        print("Denoiser Covariance Error:", torch.norm(updated_denoiser_cov - bfgs_denoiser_cov).item() / d**2)
        print("Inverse Denoiser Covariance Error:", torch.norm(updated_inv_denoiser_cov - bfgs_inv_denoiser_cov).item() / d**2)
        print("Hessian Error:", torch.norm(updated_hessian - bfgs_hessian).item() / d**2)
        print("Inverse Hessian Error:", torch.norm(updated_inv_hessian - bfgs_inv_hessian).item() / d**2)


        score_at_xnext = score_fn(xnext, tnext)
        denoiser_mean_at_xnext = xnext + (schedule(tnext) ** 2) * score_at_xnext

        updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian = update_bfgs(
            updated_denoiser_cov[0], updated_inv_denoiser_cov[0], denoiser_mean_at_t, denoiser_mean_at_xnext[None,:], 
            schedule, tnext, x[None,:], dx[None,:]
        )

        bfgs.update_space_step(denoiser_mean_at_t, denoiser_mean_at_xnext, schedule(tnext), x, xnext)
        bfgs_denoiser_cov, bfgs_inv_denoiser_cov, bfgs_hessian, bfgs_inv_hessian = bfgs.get_dense_matrices()

        print("---------------Space update results-----------------")
        print("Denoiser Covariance Error:", torch.norm(updated_denoiser_cov - bfgs_denoiser_cov).item() / d**2)
        print("Inverse Denoiser Covariance Error:", torch.norm(updated_inv_denoiser_cov - bfgs_inv_denoiser_cov).item() / d**2)
        print("Hessian Error:", torch.norm(updated_hessian - bfgs_hessian).item() / d**2)
        print("Inverse Hessian Error:", torch.norm(updated_inv_hessian - bfgs_inv_hessian).item() / d**2)

        # State updates for the dense matrices
        denoiser_cov, inv_denoiser_cov, hessian, inv_hessian = updated_denoiser_cov, updated_inv_denoiser_cov, updated_hessian, updated_inv_hessian

        x = xnext

if __name__ == "__main__":
    test_covariance_hessian_time_update()
    test_covariance_hessian_time_update_with_u_and_v()
    test_bfgs_update()
    test_time_and_space_updates()