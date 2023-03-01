import numpy as np
import numpy.linalg as la

def normalize(seq, frame: int, jid_left=6, jid_right=12, return_transform=False):
    """
    :param seq: {n_frames x 19 x 3}
    :param frame:
    """
    #  0  1  2  3  4
    #  5  6  7  8  9 10
    # 11 12 13 14 15 16
    assert len(seq) > frame
    left3d = seq[frame, jid_left]
    right3d = seq[frame, jid_right]

    if np.isclose(left3d[2], 0.0):
        raise ValueError("Left seems to be zero!" + left3d)
    if np.isclose(right3d[2], 0.0):
        raise ValueError("Right seems to be zero!" + right3d)

    mu, R = get_normalization(left3d, right3d)
    if return_transform:
        return apply_normalization_to_seq(seq, mu, R), (mu, R)
    else:
        return apply_normalization_to_seq(seq, mu, R)


def undo_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x 19 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    seq = unknown_pose_shape_to_known_shape(seq)
    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    R_T = np.transpose(R)
    seq = apply_rotation_to_seq(seq, R_T)
    seq = seq + mu
    return seq


def apply_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x 19 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    seq = seq - mu
    return apply_rotation_to_seq(seq, R)


def get_normalization(left3d, right3d):
    """
    Get rotation + translation to center and face along the x-axis
    """
    mu = (left3d + right3d) / 2
    mu[2] = 0
    left2d = left3d[:2]
    right2d = right3d[:2]
    y = right2d - left2d
    y = y / la.norm(y)
    angle = np.arctan2(y[1], y[0])
    R = rot3d(0, 0, angle)
    return mu, R


def rot3d(a, b, c):
    """"""
    Rx = np.array(
        [[1.0, 0.0, 0.0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]],
        np.float32,
    )
    Ry = np.array(
        [[np.cos(b), 0, np.sin(b)], [0.0, 1.0, 0.0], [-np.sin(b), 0, np.cos(b)]],
        np.float32,
    )
    Rz = np.array(
        [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0.0, 0.0, 1.0]],
        np.float32,
    )
    return np.ascontiguousarray(Rx @ Ry @ Rz)


def apply_rotation_to_seq(seq, R):
    """
    :param seq: {n_frames x 18 x 3}
    :param R: {3x3}
    """
    R = np.expand_dims(R, axis=0)
    return np.ascontiguousarray(seq @ R)
