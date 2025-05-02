import torch

all = [
    'skew',
    'se3_log',
    'se3_exp',
    'standard_lbs_alexa'
]

def skew(w):
    """Convert batched 3D vector(s) to skew-symmetric matrices. Shape: (..., 3) → (..., 3, 3)"""
    zeros = torch.zeros_like(w[..., 0])  # Match input batch dimensions
    return torch.stack([
        torch.stack([zeros, -w[..., 2], w[..., 1]], dim=-1),
        torch.stack([w[..., 2], zeros, -w[..., 0]], dim=-1),
        torch.stack([-w[..., 1], w[..., 0], zeros], dim=-1),
    ], dim=-2)

def se3_log(T):
    """Convert SE(3) transformation matrix to se(3) twist (6D vector)."""
    # T shape: (..., 4, 4)
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    # Compute rotation axis and angle
    theta = torch.acos((torch.diagonal(R, dim1=-2, dim2=-1).sum(-1) - 1) / 2)
    theta = theta.clamp(min=1e-6)  # Avoid division by zero

    # Compute omega (axis-angle)
    omega = (1 / (2 * torch.sin(theta.unsqueeze(-1))) * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]
    ], dim=-1)
    )
    # Compute translation part (v)
    A = torch.sin(theta) / theta
    B = (1 - torch.cos(theta)) / theta**2
    C = (1 - A) / theta**2
    omega_skew = skew(omega)
    invV = A.unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=T.device) + \
            B.unsqueeze(-1).unsqueeze(-1) * omega_skew + \
            C.unsqueeze(-1).unsqueeze(-1) * omega_skew @ omega_skew
    v = torch.linalg.solve(invV, t.unsqueeze(-1)).squeeze(-1)

    # Combine into twist (6D)
    twist = torch.cat([omega * theta.unsqueeze(-1), v], dim=-1)
    return twist

def se3_exp(twist):
    """Convert se(3) twist (6D vector) to SE(3) matrix."""
    # Input twist shape: (B, H, 6) e.g., (6000, 10, 6)
    omega = twist[..., :3]  # Shape: (B, H, 3)
    v = twist[..., 3:]      # Shape: (B, H, 3)

    # Remove singleton dimensions if present
    if omega.dim() == 4 and omega.size(-2) == 1:
        omega = omega.squeeze(-2)  # Fix shape to (B, H, 3)
    if v.dim() == 4 and v.size(-2) == 1:
        v = v.squeeze(-2)  # Fix shape to (B, H, 3)

    # Compute rotation parameters
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # (B, H, 1)
    theta = theta.clamp(min=1e-6)
    omega_norm = omega / theta  # (B, H, 3)

    # Handle edge case: zero rotation (pure translation)
    omega_norm = torch.where(theta < 1e-6, torch.zeros_like(omega_norm), omega_norm)

    # Compute skew-symmetric matrix
    omega_skew = skew(omega_norm)  # Output shape: (B, H, 3, 3)

    # Compute coefficients with broadcasting
    A = torch.sin(theta) / theta  # (B, H, 1)
    B = (1 - torch.cos(theta)) / theta**2
    C = (1 - A) / theta**2

    # Explicit broadcasting for tensors
    A_exp = A.unsqueeze(-1)  # (B, H, 1, 1)
    B_exp = B.unsqueeze(-1)  # (B, H, 1, 1)
    C_exp = C.unsqueeze(-1)  # (B, H, 1, 1)

    # Compute terms with einsum (batched matrix-vector ops)
    term1 = torch.einsum('bhij,bhj->bhi', omega_skew, v)          # (B, H, 3)
    term2 = torch.einsum('bhij,bhjk,bhk->bhi', omega_skew, omega_skew, v)  # (B, H, 3)

    # Combine terms with aligned dimensions
    t = (
        A_exp.squeeze(-1) * v +  # (B, H, 3)
        B_exp.squeeze(-1) * term1 + 
        C_exp.squeeze(-1) * term2
    )

    # Build SE(3) matrix
    eye = torch.eye(3, device=twist.device)
    R = eye + torch.sin(theta.unsqueeze(-1)) * omega_skew + \
        (1 - torch.cos(theta.unsqueeze(-1))) * (omega_skew @ omega_skew)
    
    T = torch.zeros((*twist.shape[:-1], 4, 4), device=twist.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    
    return T  # Output shape: (B, H, 4, 4)

def standard_lbs_alexa(x0, tfms, w_x0):
    """Applies Alexa's logarithmic/exponential blend skinning."""
    N = x0.shape[0]  # Number of points (e.g., 6000)
    B, H = tfms.shape[0], tfms.shape[1]  # Batch size (e.g., 6000), handles (e.g., 10)

    # Convert 3x4 affine matrices to 4x4 homogeneous matrices
    ones = torch.zeros((B, H, 1, 4), dtype=tfms.dtype, device=tfms.device)
    ones[..., 3] = 1.0
    tfms_4x4 = torch.cat([tfms, ones], dim=2)  # Shape: (B, H, 4, 4)

    # Compute se(3) twists for all transformations
    twists = se3_log(tfms_4x4)  # Shape: (B, H, 6)

    # Expand weights to (N, B, H)
    w_x0_expanded = w_x0.unsqueeze(1).expand(N, B, H)  # (N, B, H)

    # Blend twists: sum_j (w_j * twist_j)
    blended_twists = torch.einsum('nbh,bhk->nbk', w_x0_expanded, twists)  # (N, B, 6)

    # Convert blended twists back to SE(3)
    T_blend = se3_exp(blended_twists)  # Shape: (N, B, 4, 4)

    # Apply blended transformations to rest points (homogeneous coordinates)
    x0_homo = torch.cat([x0, torch.ones_like(x0[:, :1])], dim=1)  # (N, 4)
    x0_homo = x0_homo.unsqueeze(1).unsqueeze(-1).expand(-1, B, -1, -1)  # (N, B, 4, 1)

    # Transform points and retain singleton dimension
    x_transformed = torch.matmul(T_blend[..., :3, :], x0_homo)  # Shape: (N, B, 3, 1)
    x_transformed = x_transformed.permute(0, 1, 3, 2)  # Shape: (N, B, 1, 3)

    return x_transformed  # Shape: [6000, 10, 1, 3]