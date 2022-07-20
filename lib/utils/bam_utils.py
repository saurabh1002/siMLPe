import numpy as np
import numpy.linalg as la

def unknown_pose_shape_to_known_shape(seq):
    """
    Sometimes we don't know what the pose is shaped like:
    We could get either a single pose: (18*3) OR (18x3) OR (17*3) OR (17x3)
    OR it could be a sequence: (nx18*3) OR (nx18x3) OR (nx17*3) OR (nx17x3)

    This function takes any of those pose(s) and transforms
    into (nx18x3) OR (nx17x3)
    """
    if isinstance(seq, list):
        seq = np.array(seq, dtype=np.float32)
    if len(seq.shape) == 1:
        # has to be dimension 18*3 or 17*3
        if seq.shape[0] == 18 * 3:
            seq = seq.reshape(1, 18, 3)
        else:
            seq = seq.reshape(1, 17, 3)
    elif len(seq.shape) == 2:
        if seq.shape[0] == 18:
            if seq.shape[1] != 3:
                raise ValueError("(1) Incorrect shape:" + str(seq.shape))
            seq = seq.reshape(1, 18, 3)
        elif seq.shape[0] == 17:
            if seq.shape[1] != 3:
                raise ValueError("(1) Incorrect shape:" + str(seq.shape))
            seq = seq.reshape(1, 17, 3)
        else:
            if seq.shape[1] != 18 * 3 and seq.shape[1] != 17 * 3:
                raise ValueError("(2) Incorrect shape:" + str(seq.shape))
            n = len(seq)

            if seq.shape[1] == 18 * 3:
                seq = seq.reshape(n, 18, 3)
            else:
                seq = seq.reshape(n, 17, 3)
    else:
        if len(seq.shape) != 3:
            raise ValueError("(3) Incorrect shape:" + str(seq.shape))
        if (seq.shape[1] != 18 and seq.shape[1] != 17) or seq.shape[2] != 3:
            raise ValueError("(4) Incorrect shape:" + str(seq.shape))
    return seq


def normalize(seq, frame: int, jid_left=13, jid_right=14, return_transform=False):
    """
    :param seq: {n_frames x 18 x 3}
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
    :param seq: {n_frames x 18 x 3}
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
    :param seq: {n_frames x 18 x 3}
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
