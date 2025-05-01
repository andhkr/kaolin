import torch

def transform_to_dq(transforms):
    """
    Convert batch of affine transforms to dual quaternions
    
    Args:
        transforms (torch.Tensor): Tensor of affine handles, shape (batch_size, num_handles, dim, dim+1)
        
    Returns:
        (torch.Tensor): Batch of dual quaternions, shape (batch_size, num_handles, 8)
    """
    B = transforms.shape[0]  # Batch size
    H = transforms.shape[1]  # Number of handles
    
    # Reshape for processing
    transforms = transforms.reshape(B * H, 3, 4)
    
    # Extract rotation matrix (3x3) and translation vector (3x1)
    rot_matrices = transforms[:, :, :3]  # (B*H, 3, 3)
    translations = transforms[:, :, 3:]  # (B*H, 3, 1)
    
    # Convert rotation matrices to quaternions
    # Based on method from Ken Shoemake's paper
    q = torch.zeros((B * H, 4), device=transforms.device)
    
    # Computing trace of rotation matrix
    trace = torch.diagonal(rot_matrices, dim1=1, dim2=2).sum(dim=1)
    
    # Handle different cases based on trace value
    mask = trace > 0
    
    # Case 1: trace > 0
    if mask.any():
        s = torch.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * s
        q[mask, 1] = (rot_matrices[mask, 2, 1] - rot_matrices[mask, 1, 2]) / s
        q[mask, 2] = (rot_matrices[mask, 0, 2] - rot_matrices[mask, 2, 0]) / s
        q[mask, 3] = (rot_matrices[mask, 1, 0] - rot_matrices[mask, 0, 1]) / s
    
    # Case 2: If trace is not > 0, find largest diagonal element
    not_mask = ~mask
    if not_mask.any():
        # Find which diagonal element is largest for each matrix
        diag = torch.diagonal(rot_matrices, dim1=1, dim2=2)  # (B*H, 3)
        max_diag_idx = torch.argmax(diag[not_mask], dim=1)  # (B*H,)
        
        for i in range(3):
            idx_mask = not_mask.clone()
            idx_mask[not_mask] = (max_diag_idx == i)
            if idx_mask.any():
                j = (i + 1) % 3
                k = (i + 2) % 3
                
                s = torch.sqrt(1.0 + rot_matrices[idx_mask, i, i] - 
                               rot_matrices[idx_mask, j, j] - 
                               rot_matrices[idx_mask, k, k]) * 2.0
                
                q[idx_mask, i+1] = 0.25 * s
                q[idx_mask, j+1] = (rot_matrices[idx_mask, j, i] + rot_matrices[idx_mask, i, j]) / s
                q[idx_mask, k+1] = (rot_matrices[idx_mask, k, i] + rot_matrices[idx_mask, i, k]) / s
                q[idx_mask, 0] = (rot_matrices[idx_mask, k, j] - rot_matrices[idx_mask, j, k]) / s
    
    # Normalize quaternions
    q_norm = torch.norm(q, dim=1, keepdim=True)
    q = q / q_norm
    
    # Compute dual part quaternions
    # dual_q = 0.5 * [t, 0, 0, 0] * q
    # where * is quaternion multiplication
    
    # Dual part quaternion calculation
    # Prepare translation quaternion [0, tx, ty, tz]
    t_quat = torch.zeros((B * H, 4), device=transforms.device)
    t_quat[:, 1:] = translations.squeeze(-1)
    
    # Quaternion multiplication (0.5 * t_quat * q)
    dual_q = torch.zeros((B * H, 4), device=transforms.device)
    dual_q[:, 0] = -0.5 * (t_quat[:, 1] * q[:, 1] + t_quat[:, 2] * q[:, 2] + t_quat[:, 3] * q[:, 3])
    dual_q[:, 1] = 0.5 * (t_quat[:, 0] * q[:, 1] + t_quat[:, 2] * q[:, 3] - t_quat[:, 3] * q[:, 2])
    dual_q[:, 2] = 0.5 * (t_quat[:, 0] * q[:, 2] + t_quat[:, 3] * q[:, 1] - t_quat[:, 1] * q[:, 3])
    dual_q[:, 3] = 0.5 * (t_quat[:, 0] * q[:, 3] + t_quat[:, 1] * q[:, 2] - t_quat[:, 2] * q[:, 1])
    
    # Combine into dual quaternion [real_part, dual_part]
    dq = torch.cat([q, dual_q], dim=1)  # (B*H, 8)
    
    # Reshape to match input format
    dq = dq.reshape(B, H, 8)
    
    return dq

def normalize_dq(dq):
    """
    Normalize dual quaternions
    
    Args:
        dq (torch.Tensor): Dual quaternions, shape (..., 8)
        
    Returns:
        (torch.Tensor): Normalized dual quaternions, shape (..., 8)
    """
    # Extract real and dual parts
    real = dq[..., :4]
    dual = dq[..., 4:]
    
    # Normalize real part
    real_norm = torch.norm(real, dim=-1, keepdim=True)
    real_normalized = real / real_norm
    
    # Normalize dual part (ensuring orthogonality constraint)
    real_dot_dual = torch.sum(real * dual, dim=-1, keepdim=True)
    dual_normalized = (dual - real_normalized * real_dot_dual) / real_norm
    
    # Combine normalized parts
    return torch.cat([real_normalized, dual_normalized], dim=-1)

def dq_to_transformation(dq):
    """
    Convert dual quaternion to transformation matrix
    
    Args:
        dq (torch.Tensor): Dual quaternion, shape (..., 8)
        
    Returns:
        (torch.Tensor): Transformation matrix, shape (..., 3, 4)
    """
    # Extract normalized real and dual parts
    normalized_dq = normalize_dq(dq)
    q_r = normalized_dq[..., :4]  # Real part
    q_d = normalized_dq[..., 4:]  # Dual part
    
    # Extract quaternion components
    w, x, y, z = q_r[..., 0], q_r[..., 1], q_r[..., 2], q_r[..., 3]
    
    # Calculate rotation matrix
    rot = torch.zeros((*q_r.shape[:-1], 3, 3), device=dq.device)
    
    # First row
    rot[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rot[..., 0, 1] = 2 * (x * y - w * z)
    rot[..., 0, 2] = 2 * (x * z + w * y)
    
    # Second row
    rot[..., 1, 0] = 2 * (x * y + w * z)
    rot[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rot[..., 1, 2] = 2 * (y * z - w * x)
    
    # Third row
    rot[..., 2, 0] = 2 * (x * z - w * y)
    rot[..., 2, 1] = 2 * (y * z + w * x)
    rot[..., 2, 2] = 1 - 2 * (x * x + y * y)
    
    # Extract translation using: t = 2 * q_d * q_r_conj
    q_r_conj = q_r.clone()
    q_r_conj[..., 1:] = -q_r_conj[..., 1:]  # Conjugate of real quaternion
    
    # Quaternion multiplication
    t_quat = torch.zeros_like(q_r)
    t_quat[..., 0] = -q_d[..., 1] * q_r_conj[..., 1] - q_d[..., 2] * q_r_conj[..., 2] - q_d[..., 3] * q_r_conj[..., 3]
    t_quat[..., 1] = q_d[..., 0] * q_r_conj[..., 1] + q_d[..., 2] * q_r_conj[..., 3] - q_d[..., 3] * q_r_conj[..., 2]
    t_quat[..., 2] = q_d[..., 0] * q_r_conj[..., 2] + q_d[..., 3] * q_r_conj[..., 1] - q_d[..., 1] * q_r_conj[..., 3]
    t_quat[..., 3] = q_d[..., 0] * q_r_conj[..., 3] + q_d[..., 1] * q_r_conj[..., 2] - q_d[..., 2] * q_r_conj[..., 1]
    
    # Extract translation (multiply by 2)
    trans = 2 * t_quat[..., 1:]
    
    # Combine rotation and translation into transformation matrix
    transform = torch.cat([rot, trans.unsqueeze(-1)], dim=-1)
    
    return transform

def blend_dq(dqs, weights):
    """
    Blend dual quaternions based on weights
    
    Args:
        dqs (torch.Tensor): Dual quaternions, shape (N, BH, 8)
        weights (torch.Tensor): Blend weights, shape (N, H)
        
    Returns:
        (torch.Tensor): Blended dual quaternions, shape (N, B, 8)
    """
    N = weights.shape[0]  # Number of points
    H = weights.shape[1]  # Number of handles
    BH = dqs.shape[1]     # Batch size * Number of handles
    B = BH // H          # Batch size
    
    # Reshape dqs to (N, B, H, 8)
    dqs_reshaped = dqs.reshape(N, B, H, 8)
    
    # Adjust weights for broadcasting
    # Weights shape: (N, 1, H, 1)
    weights_expanded = weights.unsqueeze(1).unsqueeze(-1)
    
    # Handle quaternion antipodality
    # Ensure all quaternions are in the same hemisphere
    # We use the first handle as reference
    ref_dq = dqs_reshaped[:, :, 0:1, :]  # (N, B, 1, 8)
    
    # Calculate dot product of real parts to determine hemisphere
    real_dot = torch.sum(dqs_reshaped[..., :4] * ref_dq[..., :4], dim=-1, keepdim=True)
    
    # Flip quaternions that are in opposite hemisphere
    flip_mask = real_dot < 0
    dqs_adjusted = dqs_reshaped.clone()
    dqs_adjusted[flip_mask.expand_as(dqs_reshaped)] = -dqs_reshaped[flip_mask.expand_as(dqs_reshaped)]
    
    # Weighted blend of adjusted dual quaternions
    blended_dqs = torch.sum(weights_expanded * dqs_adjusted, dim=2)  # (N, B, 8)
    
    # Normalize the blended dual quaternions
    normalized_blended_dqs = normalize_dq(blended_dqs)
    
    return normalized_blended_dqs

def dual_quaternion_skinning(x0, tfms, w_x0):
    r""" 
    Applies dual quaternion skinning transform batched over all pts: x0 for a batch of transforms, given skinning weights w_x0
    
    Args:
        x0 (torch.Tensor): Rest state points, of shape :math:`(\text{num_pts}, \text{dim})`
        tfms (torch.Tensor): Tensor of affine handles, of shape :math:`(\text{batch_size}, \text{num_handles}, \text{dim}, \text{dim}+1)`
        w_x0 (torch.Tensor): Matrix of skinning weights, of shape :math:`(\text{num_pts}, \text{num_handles})` 

    Returns:
        (torch.Tensor): Transformed points, of shape :math:`(\text{batch_size}, \text{num_pts}, 1, \text{dim})`
    """
    N = x0.shape[0]  # Number of sampled points
    B = tfms.shape[0]  # Sample transform batch size
    H = tfms.shape[1]  # Number of handles
    BH = B * H
    
    # Convert transforms to dual quaternions
    dqs = transform_to_dq(tfms)  # (B, H, 8)
    
    # Expand and reshape dual quaternions for all points
    dqs_expanded = dqs.reshape(1, BH, 8).expand(N, BH, 8)
    
    # Blend dual quaternions according to weights
    blended_dqs = blend_dq(dqs_expanded, w_x0)  # (N, B, 8)
    
    # Convert blended dual quaternions back to transformation matrices
    transforms = dq_to_transformation(blended_dqs)  # (N, B, 3, 4)
    
    # Apply transformations to points
    # (N, 1, 3)
    x0_i = x0.unsqueeze(1)
    # (N, 4, 1)
    x03 = torch.cat((x0_i, x0_i.new_ones(N, 1, 1)), dim=2).transpose(1, 2)
    
    # Expand x03 to (N, B, 4, 1)
    x03_expanded = x03.unsqueeze(1).expand(N, B, 4, 1)
    
    # Apply transformation: (N, B, 3, 4) @ (N, B, 4, 1) -> (N, B, 3, 1)
    transformed_points = transforms @ x03_expanded
    
    # Reshape to match standard_lbs output: (N, B, 1, 3)
    transformed_points = transformed_points.transpose(2, 3)
    
    return transformed_points
