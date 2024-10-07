import numpy as np
import random


def cRt_to_trf_mat(c, R, t):
    c, R, t = np.asarray(c), np.asarray(R), np.asarray(t)
    trf_mat = c * R
    trf_mat = np.hstack((trf_mat, np.reshape(t, (3, 1))))
    trf_mat = np.vstack((trf_mat, [0, 0, 0, 1]))

    return trf_mat


def trf_mat_to_cRt(trf_mat):
    trf_mat = np.asarray(trf_mat)
    t = trf_mat[:3, 3]
    R = trf_mat[:3, :3]
    c = np.cbrt(np.linalg.det(R))
    R = (1.0 / c) * R
    
    return c, R, t


def apply_transform(pts, c, R, t, is_colvec):
    pts, c, R, t = np.asarray(pts), np.asarray(c), np.asarray(R), np.asarray(t)
    if is_colvec:
        return  c * np.matmul(R, pts) + t.reshape((3, 1))
    else:
        return  c * np.matmul(pts, R.T) + t.reshape((1, 3))


def apply_transform_trf_mat(pts, trf_mat, is_colvec):
    pts, trf_mat = np.asarray(pts), np.asarray(trf_mat)
    if is_colvec:
        pts = np.pad(pts, [(0, 1), (0, 0)], mode='constant', constant_values=1.0)
        pts = np.matmul(trf_mat, pts)
        pts = pts[:-1, :]
        return pts
    else:
        pts = np.pad(pts, [(0, 0), (0, 1)], mode='constant', constant_values=1.0)
        pts = np.matmul(pts, trf_mat.T)
        pts = pts[:, :-1]
        return pts


# def camera_pose_direction(extrinsic_rotation):
#     return np.asarray(extrinsic_rotation)[2, :]

# def camera_pose_direction_list(extrinsic_rotation_list):
#     return np.asarray(extrinsic_rotation_list)[:, 2, :]


def similarity_transform(from_pts, to_pts, is_colvec):
    '''
    Umeyama tranform: https://pdfs.semanticscholar.org/d107/231cce2676dbeea87e00bb0c587c280b9c53.pdf?_ga=2.96067861.842228479.1608107486-40852687.1608107486
    Code adapted from: https://gist.github.com/dboyliao/f7f862172ed811032ba7cc368701b1e8
    
    Input: from_pts, to_pts 
    Output: Similarity transformation (scale, rotation matrix, offset)
    '''

    if is_colvec:
        from_points = from_pts.T
        to_points = to_pts.T
    else:
        from_points = from_pts
        to_points = to_pts

    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis = 0)
    mean_to = to_points.mean(axis = 0)

    delta_from = from_points - mean_from # N x m
    delta_to = to_points - mean_to       # N x m

    sigma_from = (delta_from * delta_from).sum(axis = 1).mean()
    sigma_to = (delta_to * delta_to).sum(axis = 1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m-1, m-1] = -1
    elif cov_rank < m-1:
        raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)

    return c, R, t


def ransac_similarity_transform(from_pts, to_pts, is_colvec, sample_ratio, n_iteration, inlier_ratio, is_group=False):
    if is_group:
        # from_pts, to_pts should be group_num x N x 3 or group_num x 3 x N 
        group_num = from_pts.shape[0]
        # to (group_num x N) x 3
        if is_colvec:
            from_points = from_pts.transpose(0, 2, 1) 
            to_points = to_pts.transpose(0, 2, 1) 
        else:
            from_points = from_pts
            to_points = to_pts
        from_points = np.reshape(from_points, (from_points.shape[0]*from_points.shape[1], from_points.shape[2]))
        to_points = np.reshape(to_points, (to_points.shape[0]*to_points.shape[1], to_points.shape[2]))
    else:
        # to N x 3
        if is_colvec:
            from_points = from_pts.T
            to_points = to_pts.T
        else:
            from_points = from_pts
            to_points = to_pts

    assert len(from_points.shape) == 2, \
        "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, \
        "from_points and to_points must have the same shape"
    assert len(from_points) == len(to_points), \
        "from_points and to_points must have the same number of points"

    from_points_col, to_points_col = from_points.T, to_points.T
    
    N, m = from_points.shape
    if is_group:
        N //= group_num
    sample_N = int(np.ceil(N * sample_ratio))
    inlier_N = int(np.ceil(N * inlier_ratio))

    best_error = None
    best_model = None
    index = list(range(N))
    for _ in range(n_iteration):
        random.shuffle(index)
        sample_index = index[:sample_N]
        if is_group:
            sample_index_tmp = np.array(sample_index)
            sample_index = np.array(sample_index)
            for i in range(1, group_num):
                sample_index = np.concatenate((sample_index, sample_index_tmp + (i * N)))
        sample_from_points, sample_to_points = from_points[sample_index], to_points[sample_index]
        c, R, t = similarity_transform(from_pts=sample_from_points, to_pts=sample_to_points, is_colvec=False)

        error = np.sum(np.square(apply_transform(from_points_col, c, R, t, is_colvec=True) - to_points_col), axis=0)
        if is_group:
            error = np.sum(error.reshape((group_num, N)), axis=0)

        error_compare = np.sum(np.sort(error)[:inlier_N])
        if best_error is None or error_compare < best_error:
            best_error = error_compare
            best_model = (c, R, t)
    
    return best_model