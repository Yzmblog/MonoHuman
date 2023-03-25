
def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.
    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])

    return torch.stack((o0, o1, o2, o3), -1)

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )

    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(d6):
    """ 
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix 
    using Gram--Schmidt orthogonalisation per Section B of [1]. 
    Args: 
        d6: 6D rotation representation, of size (*, 6) 

    Returns: 
        batch of rotation matrices of size (*, 3, 3) 

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. 
    On the Continuity of Rotation Representations in Neural Networks. 
    IEEE Conference on Computer Vision and Pattern Recognition, 2019. 
    Retrieved from http://arxiv.org/abs/1812.07035 
    """ 

    a1, a2 = d6[..., :3], d6[..., 3:] 
    b1 = F.normalize(a1, dim=-1) 
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1 
    b2 = F.normalize(b2, dim=-1) 
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2) 

def matrix_to_axis_angle(matrix):
    """ 
    Convert rotations given as rotation matrices to axis/angle. 

    Args: 
        matrix: Rotation matrices as tensor of shape (..., 3, 3). 

    Returns: 
        Rotations given as a vector in axis angle form, as a tensor 
            of shape (..., 3), where the magnitude is the angle 
            turned anticlockwise in radians around the vector's 
            direction. 
    """

    return quaternion_to_axis_angle(matrix_to_quaternion(matrix)) 

###
params = np.load('xxx.npy', allow_pickle=True).item()['thetas']
poses = np.array(matrix_to_axis_angle(rotation_6d_to_matrix(torch.tensor(params[:, :, idx])))).reshape(-1)

