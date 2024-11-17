import numpy as np


data = np.load('test.npz')
Z_test = data['Z_test']    # M,C,(T-1)
A_test = data['A_test']     # M,M
print('Z_test:', Z_test.shape)
print('A_test:', A_test.shape)
M=Z_test.shape[0]
C=Z_test.shape[1]
T=Z_test.shape[2]
K = int(M*(M-1)/2)

def compute_A(H, beta):
    """
    Parameters:
        beta (np.ndarray): A 1D array of shape [K].
        H (np.ndarray): A 3D array of shape [K, M, M].

    Returns:
        A (np.ndarray): A 2D array of shape [M, M].

    Raises:
        ValueError: If the shapes of beta and H are incompatible.
    """
    # Check dimensions
    if len(beta.shape) != 1:
        raise ValueError(f"Invalid shape for beta. Expected 1D array of shape [K], but got {beta.shape}.")

    if len(H.shape) != 3:
        raise ValueError(f"Invalid shape for H. Expected 3D array of shape [K, M, M], but got {H.shape}.")

    if beta.shape[0] != K:
        raise ValueError(f"Incompatible shapes for beta and H. Got beta: {beta.shape}, H: {H.shape}.")

    if H.shape[0] != K or H.shape[1] != M or H.shape[2] != M:
        raise ValueError(f"Incompatible shapes for beta and H. Got beta: {beta.shape}, H: {H.shape}.")
    
    # Compute A
    A = np.sum(H * beta[:, None, None], axis=0)
    return A

def compute_W(H, Z):
    """
    Computes W = H @ Z with shape checks.
    
    Parameters:
        H (np.ndarray): A 3D array of shape [K, M, M].
        Z (np.ndarray): A 2D array of shape [M, C(T-1)].
    
    Returns:
        W (np.ndarray): A 3D array of shape [K, M, C(T-1)].
    
    Raises:
        ValueError: If the shapes of H and Z_short are incompatible.
    """
    # Check input shapes
    if len(H.shape) != 3 or len(Z.shape) != 2:
        raise ValueError(f"Invalid shapes: H must be [K, M, M] and Z_short must be [M, C(T-1)]. Got H: {H.shape}, Z_short: {Z_short.shape}.")
        
    if H.shape[0] != K or H.shape[1] != M or H.shape[2] != M:
        raise ValueError(f"Incompatible shapes for matrix multiplication. Got H: {H.shape}.")

    if Z.shape[0] != M or Z.shape[1] != C*(T-1):
        raise ValueError(f"Incompatible shapes for matrix multiplication. Got Z: {Z.shape}.")
    
    # Perform the multiplication
    W = np.einsum('kmn,nc->kmc', H, Z)

    # Verify the output shape
    expected_shape = (K, M, C*(T-1))
    if W.shape != expected_shape:
        raise ValueError(f"Unexpected output shape. Expected {expected_shape}, but got {W.shape}.")
    
    return W

def compute_beta(W, dZ):
    """
    Computes beta using the formula β = (WWT)^(-1)W∆Z.
    
    Parameters:
        W (np.ndarray): A 2D array of shape [K, M, C*(T-1)].
        dZ (np.ndarray): A 2D array of shape [M, C*(T-1)].
    
    Returns:
        beta (np.ndarray): A 1D array of shape [K].

    Raises:
        ValueError: If the input dimensions are incompatible.
    """
    # Input dimension checks
    if len(W.shape) != 3 or len(dZ.shape) != 2:
        raise ValueError("Invalid input shapes. H must be [K, M, M], Z and delta_Z must be [M, C*(T-1)].")
    
    if W.shape[0] != K or W.shape[1] != M or W.shape[2] != C*(T-1):
        raise ValueError(f"Incompatible shapes for matrix multiplication. Got H: {H.shape}.")
    
    if dZ.shape[0] != M or dZ.shape[1] != C*(T-1):
        raise ValueError(f"Incompatible shapes for matrix multiplication. Got Z: {Z.shape}.")

    # Reshape W to [K, M*C*(T-1)]
    W_flat = W.reshape(K, -1)  # Flatten along the last two dimensions

    # Reshape delta_Z to [M*C*(T-1)]
    delta_Z_flat = dZ.flatten()  # Flatten delta_Z to a 1D vector

    # Compute β = (W W⊤)^(-1) W ∆Z
    WW_t = np.dot(W_flat, W_flat.T)  # W W⊤, shape [K, K]
    WW_t_inv = np.linalg.inv(WW_t)  # Inverse of W W⊤, shape [K, K]
    beta = np.dot(WW_t_inv, np.dot(W_flat, delta_Z_flat))  # Compute β, shape [K]

    return beta

def estimate_A(Z):

    H = np.zeros((K, M, M), dtype=int)
    k = 0
    for i in range(M):
        for j in range(M):
            if j > i:
                H[k, i, j] = 1
                H[k, j, i] = -1
                k+=1

    Z_3d = Z.reshape(Z.shape[0],C,-1) # (M=2, C, T)
    dZ_3d = np.diff(Z_3d, axis=2)
    Z_3d_short = Z_3d[:,:,:-1]

    Z_short = Z_3d_short.reshape(dZ_3d.shape[0],-1)
    dZ = dZ_3d.reshape(dZ_3d.shape[0],-1)

    W = compute_W(H, Z_short)  # Shape will be [K, M, C(T-1)]
    beta_estim = compute_beta(W, dZ)
    A = compute_A(H,beta_estim)

    return A


A_hat = estimate_A(Z_test.reshape(M,-1))
# print(A_hat.shape)
print(np.max( (A_hat-A_test) ))
# print(A_hat)
