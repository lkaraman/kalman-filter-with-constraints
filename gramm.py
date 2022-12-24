import numpy as np
import scipy.linalg


def mod_gramm_schmit(A: np.ndarray) -> (np.ndarray, np.ndarray):
    eps = 1e-12

    m, n = A.shape
    if m < n:
        raise ValueError('Error - input matrix cannot have more cols than rows')

    W = np.zeros((n, n))
    T = np.zeros((m+10, m+10))

    for k in range(0, n):
        sigma = scipy.linalg.sqrtm(A[:, k].T @ A[:, k])
        #
        if abs(sigma) < 100 * eps:
            raise ValueError
        W[k, k] = sigma

        for jj in range(k+1, n):
            W[k, jj] = A[:, k].T @ A[:, jj] / sigma

        T[k, 0:m] = A[:, k].T / sigma

        for jj in range(k+1, n):
            A[:, jj] = A[:, jj] - W[k, jj] @ A[:, k] / sigma

    T[n:n+m, 0:m] = np.eye((m))
    index = n

    for k in range(n, n+m):
        temp = T[k, :]

        for i in range(0, k):
            temp = temp - T[k, :] @ T[i, :].T * T[i, :]

        if np.linalg.norm(temp) > 100 * eps:
            T[index,:] = temp / np.linalg.norm(temp)
            index = index + 1

    T = T[0:m, 0:m]

    return W, T