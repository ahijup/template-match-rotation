import numpy as np

def least_squared_fitting(x, y_vals):
    A = np.zeros((x.shape[0], 3), dtype=np.float64)
    A[:, 0] = x * x
    A[:, 1] = x
    A[:, 2] = 1
    P = np.linalg.lstsq(A, y_vals, rcond=None)
    a0, a1, a2 = P[0]
    # peak position: -a1 / 2a0
    x0 = -a1 / (2 * a0)
    # peak value: a0 * x0 * x0 + a1 * x0 + a2
    max_val = a0 * x0 * x0 + a1 * x0 + a2
    return x0, max_val

def least_squared_fitting_3D(x, y, t, w):
    A = np.zeros((27, 10), dtype=np.float64)
    S = np.zeros((27, 1), dtype=np.float64)

    row_idx = 0
    rot_idx = 0
    for r in t:
        rad = np.deg2rad(r)
        y_idx = 0
        for yy in y:
            x_idx = 0
            for xx in x:
                A[row_idx, 0] = xx * xx
                A[row_idx, 1] = yy * yy
                A[row_idx, 2] = rad * rad
                A[row_idx, 3] = xx * yy
                A[row_idx, 4] = xx * rad
                A[row_idx, 5] = yy * rad
                A[row_idx, 6] = xx
                A[row_idx, 7] = yy
                A[row_idx, 8] = rad
                A[row_idx, 9] = 1
                S[row_idx] = w[rot_idx][y_idx][x_idx]
                row_idx += 1
                x_idx += 1
            y_idx += 1
        rot_idx += 1

    A_t = np.transpose(A)
    A_squared = np.matmul(A_t, A)
    A_inv = np.linalg.inv(A_squared)
    A_inv_A_t = np.matmul(A_inv, A_t)
    P = np.matmul(A_inv_A_t, S)
    a, b, c, d, e, f, g, h, i, j = P.flatten()


    # do partial different
    # f(x, y, r) / dx = 2 * a * x + d * y + e * r + g = 0
    # f(x, y, r) / dy = 2 * b * y + d * x + f * r + h = 0
    # f(x, y, r) / dr = 2 * c * r + e * x + f * y + i = 0
    # [[2 * a, d, e], [d, 2 * b, f], [e, f, c]] * [x, y, r] = [-g, -h, -i]
    MM = np.array([[2 * a, d, e], [d, 2 * b, f], [e, f, 2 * c]], dtype=np.float64)
    N = np.array([-g, -h, -i], dtype=np.float64)

    MM_inv = np.linalg.inv(MM)
    Sol = np.matmul(MM_inv, N)

    # Sol = np.linalg.solve(MM, N)
    x0, y0, r0 = Sol.flatten()
    max_value = a * x0 * x0 + b * y0 * y0 + c * r0 * r0 + d * x0 * y0 + e * x0 * r0 + f * y0 * r0 + g * x0 + h * y0 + i * r0 + j
    r0 = np.rad2deg(r0)
    return x0, y0, r0, max_value

def find_subpix_1D_peak(loc, cur_angle, angle_step, ncc_scores, ncc_scores_l, ncc_scores_r):
    a0, a_max = least_squared_fitting(np.array([-angle_step + cur_angle, cur_angle, angle_step + cur_angle], dtype=np.float64), 
                          np.array([ncc_scores_l[1, 1], ncc_scores[1, 1], ncc_scores_r[1, 1]], dtype=np.float64))
    x0, x_max = least_squared_fitting(np.array([loc[0] - 1, loc[0], loc[0] + 1], dtype=np.float64),
                                      np.array([ncc_scores[1, 0], ncc_scores[1, 1], ncc_scores[1, 2]], dtype=np.float64))
    y0, y_max = least_squared_fitting(np.array([loc[1] - 1, loc[1], loc[1] + 1], dtype=np.float64),
                                      np.array([ncc_scores[0, 1], ncc_scores[1, 1], ncc_scores[2, 1]], dtype=np.float64))
    
    scores = np.array([a_max, x_max, y_max], dtype=np.float64)
    s = np.mean(scores)
    return a0, (x0, y0), s
def find_subpix_3D_peak(loc, cur_angle, angle_step, ncc_scores, ncc_scores_l, ncc_scores_r):
    if ncc_scores[1, 1] > ncc_scores_l[1, 1] and ncc_scores[1, 1] > ncc_scores_r[1, 1]:
        x0, y0, r, s = least_squared_fitting_3D(
                             np.array([loc[0] - 1, loc[0], loc[0] + 1], dtype=np.float64), 
                             np.array([loc[1] - 1, loc[1], loc[1] + 1], dtype=np.float64), 
                             np.array([-angle_step + cur_angle, cur_angle, angle_step + cur_angle], dtype=np.float64),
                             (ncc_scores_l, ncc_scores, ncc_scores_r))
        # print('x0, y0, r:', x0, y0, r)
    else:
        x0, y0, r, s = loc[0], loc[1], cur_angle, ncc_scores[1, 1]
    return r, (x0, y0), s