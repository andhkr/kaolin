import torch
import torch.nn.functional as F
from functools import partial # For wrapper function example

# --- Quaternion Helper Functions (Using robust versions from previous answer) ---
# Note: Using wxyz convention for quaternions [w, x, y, z]

def robust_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions more robustly.
    Handles potential variations in leading batch dimensions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    m = matrix.reshape(-1, 3, 3) # Flatten leading dimensions for processing
    num_matrices = m.shape[0]
    if num_matrices == 0: # Handle empty input
         original_shape = matrix.shape[:-2] + (4,)
         return torch.empty(original_shape, dtype=matrix.dtype, device=matrix.device)

    q = torch.empty((num_matrices, 4), dtype=matrix.dtype, device=matrix.device)

    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    trace = m00 + m11 + m22

    # Use masks for efficient selection - adapted for clarity
    mask_trace = trace > 0
    mask_m00 = (m00 > m11) & (m00 > m22) & ~mask_trace
    mask_m11 = (m11 > m22) & ~mask_trace & ~mask_m00 # Ensure mutually exclusive masks
    mask_m22 = ~mask_trace & ~mask_m00 & ~mask_m11 # The rest

    # Case 1: trace > 0
    if torch.any(mask_trace):
        s1 = torch.sqrt(trace[mask_trace] + 1.0) * 2.0
        q[mask_trace, 0] = 0.25 * s1                   # w
        q[mask_trace, 1] = (m21[mask_trace] - m12[mask_trace]) / s1 # x
        q[mask_trace, 2] = (m02[mask_trace] - m20[mask_trace]) / s1 # y
        q[mask_trace, 3] = (m10[mask_trace] - m01[mask_trace]) / s1 # z

    # Case 2: m00 is largest diagonal element
    if torch.any(mask_m00):
        s2 = torch.sqrt(1.0 + m00[mask_m00] - m11[mask_m00] - m22[mask_m00]) * 2.0
        q[mask_m00, 0] = (m21[mask_m00] - m12[mask_m00]) / s2 # w
        q[mask_m00, 1] = 0.25 * s2                   # x
        q[mask_m00, 2] = (m01[mask_m00] + m10[mask_m00]) / s2 # y
        q[mask_m00, 3] = (m02[mask_m00] + m20[mask_m00]) / s2 # z

    # Case 3: m11 is largest diagonal element
    if torch.any(mask_m11):
        s3 = torch.sqrt(1.0 + m11[mask_m11] - m00[mask_m11] - m22[mask_m11]) * 2.0
        q[mask_m11, 0] = (m02[mask_m11] - m20[mask_m11]) / s3 # w
        q[mask_m11, 1] = (m01[mask_m11] + m10[mask_m11]) / s3 # x
        q[mask_m11, 2] = 0.25 * s3                   # y
        q[mask_m11, 3] = (m12[mask_m11] + m21[mask_m11]) / s3 # z

    # Case 4: m22 is largest diagonal element
    if torch.any(mask_m22):
        s4 = torch.sqrt(1.0 + m22[mask_m22] - m00[mask_m22] - m11[mask_m22]) * 2.0
        q[mask_m22, 0] = (m10[mask_m22] - m01[mask_m22]) / s4 # w
        q[mask_m22, 1] = (m02[mask_m22] + m20[mask_m22]) / s4 # x
        q[mask_m22, 2] = (m12[mask_m22] + m21[mask_m22]) / s4 # y
        q[mask_m22, 3] = 0.25 * s4                   # z

    # Normalize the quaternion
    q_normalized = F.normalize(q, p=2, dim=-1)

    # Reshape back to original leading dimensions
    original_shape = matrix.shape[:-2] + (4,)
    return q_normalized.reshape(original_shape)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ Multiply two quaternions (a * b). Assumes [w, x, y, z] format. """
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]

    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return torch.cat([ow, ox, oy, oz], dim=-1)

def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """ Compute the conjugate ([w, -x, -y, -z]). """
    return q * q.new_tensor([1, -1, -1, -1])

def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """ Convert rotations given as quaternions [w, x, y, z] to rotation matrices. """
    q = F.normalize(q, p=2, dim=-1) # Ensure unit quaternion input
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Calculate elements directly (slightly more efficient than paper's notation)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    matrix = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    matrix[..., 0, 0] = 1 - 2 * (yy + zz)
    matrix[..., 0, 1] = 2 * (xy - wz)
    matrix[..., 0, 2] = 2 * (xz + wy)
    matrix[..., 1, 0] = 2 * (xy + wz)
    matrix[..., 1, 1] = 1 - 2 * (xx + zz)
    matrix[..., 1, 2] = 2 * (yz - wx)
    matrix[..., 2, 0] = 2 * (xz - wy)
    matrix[..., 2, 1] = 2 * (yz + wx)
    matrix[..., 2, 2] = 1 - 2 * (xx + yy)

    return matrix

# --- Optimized Dual Quaternion Skinning (Kavan et al. 2007, Sec 3.4) ---

def dual_quaternion_nlbs(x0: torch.Tensor, tfms: torch.Tensor, w_x0: torch.Tensor) -> torch.Tensor:
    """
    Applies Dual Quaternion Linear Blend Skinning (DQLBS) using the
    optimized formulation from Kavan et al. 2007, Section 3.4.

    Args:
        x0 (torch.Tensor): Rest state points, shape (num_pts, 3).
        tfms (torch.Tensor): Handle transformations (affine matrices),
                           shape (batch_size, num_handles, 3, 4).
        w_x0 (torch.Tensor): Skinning weights, shape (num_pts, num_handles).

    Returns:
        torch.Tensor: Transformed points, shape (num_pts, batch_size, 1, 3),
                      matching the output format of the provided standard_lbs.
    """
    N = x0.shape[0]  # Number of points
    if tfms.ndim != 4 or tfms.shape[-2:] != (3, 4):
         raise ValueError(f"Expected tfms shape (B, H, 3, 4), got {tfms.shape}")
    if w_x0.ndim != 2 or w_x0.shape[0] != N:
        raise ValueError(f"Expected w_x0 shape (N, H) with N={N}, got {w_x0.shape}")

    B = tfms.shape[0]  # Batch size of transforms
    H = tfms.shape[1]  # Number of handles
    if w_x0.shape[1] != H:
        raise ValueError(f"Mismatch between w_x0 num_handles ({w_x0.shape[1]}) and tfms num_handles ({H})")

    _device = x0.device
    _dtype = x0.dtype

    # --- Step 1: Convert Matrices to Dual Quaternions (CPU/Precompute ideally) ---
    # This part is the same as before, representing tfms[b, h] as q_rot[b, h] + eps * q_dual[b, h]

    rot_matrices = tfms[..., :3, :3]  # (B, H, 3, 3)
    translations = tfms[..., :3, 3]   # (B, H, 3)

    q_rot = robust_matrix_to_quaternion(rot_matrices)  # (B, H, 4)

    # Translation part as pure quaternion: [0, tx, ty, tz]
    q_trans = torch.cat(
        [torch.zeros(B, H, 1, device=_device, dtype=_dtype), translations], dim=-1
    )  # (B, H, 4)

    # Dual part: q_dual = 0.5 * q_trans * q_rot (Eq. 4 in paper)
    q_dual = 0.5 * quaternion_multiply(q_trans, q_rot)  # (B, H, 4)

    # --- Step 2: Linear Blend (GPU/Vertex Shader in paper) ---
    # Prepare for blending
    # w_x0: (N, H) -> (N, 1, H, 1)
    weights = w_x0.unsqueeze(1).unsqueeze(-1)
    # q_rot: (B, H, 4) -> (1, B, H, 4)
    q_rot_expanded = q_rot.unsqueeze(0)
    # q_dual: (B, H, 4) -> (1, B, H, 4)
    q_dual_expanded = q_dual.unsqueeze(0)

    # Handle quaternion flipping for shortest path (same as before)
    if H > 0:
        # Use first handle's rotation quaternion as reference
        q_rot_ref = q_rot_expanded[:, :, 0:1, :] # (1, B, 1, 4)
        # Calculate dot product (sum over last dim)
        dot_prod = torch.sum(q_rot_expanded * q_rot_ref, dim=-1, keepdim=True) # (1, B, H, 1)
        # Determine sign flip for quaternions in opposite hemisphere
        sign_flip = torch.sign(dot_prod)
        sign_flip[sign_flip == 0.0] = 1.0 # Avoid flipping if dot product is zero
        # Apply flip to both real and dual parts
        q_rot_flipped = q_rot_expanded * sign_flip # (1, B, H, 4)
        q_dual_flipped = q_dual_expanded * sign_flip # (1, B, H, 4)
    else: # No handles
        q_rot_flipped = q_rot_expanded
        q_dual_flipped = q_dual_expanded


    # Perform linear blend: b = sum(w_i * q_i) = bo + eps * be
    # weights: (N, 1, H, 1) * q_..._flipped: (1, B, H, 4) -> Blend: (N, B, H, 4) -> Sum(dim=2): (N, B, 4)
    bo = torch.sum(weights * q_rot_flipped, dim=2)  # Real part of blended DQ (N, B, 4)
    be = torch.sum(weights * q_dual_flipped, dim=2) # Dual part of blended DQ (N, B, 4)

    # --- Step 3: Optimized Conversion to Matrix M (GPU/Vertex Shader) ---
    # From Section 3.4 - avoids full DQ normalization

    # Normalize the real part 'bo' to get the rotation quaternion 'co'
    bo_norm = torch.linalg.norm(bo, dim=-1, keepdim=True)
    bo_norm_safe = torch.clamp(bo_norm, min=1e-9) # Avoid division by zero
    co = bo / bo_norm_safe # This is the final rotation quaternion (N, B, 4)

    # Rotation Matrix from co
    rot_matrix = quaternion_to_matrix(co) # (N, B, 3, 3)

    # Calculate the translation vector directly using the optimized formula
    # t = vector_part(2 * be * co_conjugate) / ||bo||^2 - simplified in paper
    # Equivalent to: t = vector_part( 2 * (be / ||bo||) * (bo / ||bo||)_conjugate )
    # Let ce = be / ||bo||
    ce = be / bo_norm_safe # Scaled dual part (N, B, 4)

    # Paper's formula for translation t = (t0, t1, t2) from components of
    # co = (wo, xo, yo, zo) and ce = (we, xe, ye, ze)
    wo, xo, yo, zo = co[..., 0], co[..., 1], co[..., 2], co[..., 3]
    we, xe, ye, ze = ce[..., 0], ce[..., 1], ce[..., 2], ce[..., 3]

    # t = vector_part(2 * ce * conjugate(co))
    t0 = 2 * (-we*xo + xe*wo - ye*zo + ze*yo)
    t1 = 2 * (-we*yo + xe*zo + ye*wo - ze*xo)
    t2 = 2 * (-we*zo - xe*yo + ye*xo + ze*wo)

    translation_vector = torch.stack([t0, t1, t2], dim=-1) # (N, B, 3)

    # --- Step 4: Apply Transformation (GPU/Vertex Shader) ---
    # Apply R * x0 + t

    # Reshape x0 for matmul: (N, 3) -> (N, 1, 3, 1)
    x0_expanded = x0.unsqueeze(1).unsqueeze(-1) # (N, 1, 3, 1)

    # Apply rotation: R * x0
    # Matmul: (N, B, 3, 3) @ (N, 1, 3, 1) -> (N, B, 3, 1)
    rotated_x0 = torch.matmul(rot_matrix, x0_expanded)

    # Apply translation: R * x0 + t
    # translation_vector: (N, B, 3) -> (N, B, 3, 1)
    translation_expanded = translation_vector.unsqueeze(-1)

    transformed_x = rotated_x0 + translation_expanded  # (N, B, 3, 1)

    # --- Final Output Shape Adjustment ---
    # Return result with shape (N, B, 1, 3) to match standard_lbs
    return transformed_x.permute(0, 1, 3, 2) # Output shape (N, B, 1, 3)