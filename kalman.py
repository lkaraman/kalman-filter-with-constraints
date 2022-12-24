from math import exp
from typing import Any

import numpy as np
import scipy
from filterpy.stats import logpdf
from scipy.special import erf

from gramm import mod_gramm_schmit
from structs import KalmanOutput, RtsOutput



def apply_2d_constant_acceleration_filter(measured_data: Any):
    T = 0.1

    A = np.matrix([
        [1, T, T**2/2, 0, 0],
        [0, 1, T, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, T],
        [0, 0, 0, 0, 1]
    ])

    # we can only measure sd coordinates
    C = np.matrix([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ])

    Q = np.matrix([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]) * 0.1

    R = np.matrix([
        [10, 0],
        [0, 10]
    ])

    P = np.matrix([
        [100, 0, 0, 0, 0],
        [0, 100, 0, 0, 0],
        [0, 0, 100, 0, 0],
        [0, 0, 0, 100, 0],
        [0, 0, 0, 0, 100]
    ])

    # Initial conditions
    x = np.matrix([
        [measured_data.s[0]],
        [0],
        [0],
        [measured_data.d[0]],
        [0]
    ])

    P_plus = P.copy()
    x_est_plus = x.copy()

    nx = 5
    ny = 2
    N = len(measured_data.t)

    PminusArr = np.zeros((nx, nx, N))
    PplusArr = np.zeros((nx, nx, N))
    xhatminusArr = np.zeros((nx, N))
    xhatplusArr = np.zeros((nx, N))
    KArr = np.zeros((nx, ny, N))

    for k, t in enumerate(measured_data.t):
        y_meas = np.matrix([
            [measured_data.s[k]],
            [measured_data.d[k]]
        ])

        # Standard forward Kalman filter
        P_minus = A @ P_plus @ A.T + Q
        K = P_minus @ C.T @ np.linalg.inv(C @ P_minus @ C.T + R)
        x_est_minus = A @ x_est_plus
        x_est_plus = x_est_minus + K @ (y_meas - C @ x_est_minus)
        P_plus = P_minus - K @ C @ P_minus

        KArr[:, :, k] = K
        PminusArr[:, :, k] = P_minus
        PplusArr[:, :, k] = P_plus

        xhatminusArr[:, k] = x_est_minus.flatten()
        xhatplusArr[:, k] = x_est_plus.flatten()

    return KalmanOutput(
        P_minus_arr=PminusArr,
        P_plus_arr=PplusArr,
        x_minus_arr=xhatminusArr,
        x_plus_arr=xhatplusArr,
        K_arr=KArr,
        A=A

    )


def apply_rts(ko: KalmanOutput):
    nx = ko.A.shape[0]
    N = ko.K_arr.shape[2]
    xhatSmooth = ko.x_plus_arr[:, -1]
    PSmooth = ko.P_plus_arr[:, :, -1]
    xhatSmoothArr = np.zeros((nx, N))
    xhatSmoothArr[:, N - 1] = xhatSmooth
    KSmootherArr = np.zeros((nx, nx, N - 1))
    PSmootherArr = np.zeros((nx, nx, N))
    PSmootherArr[:, :, N-1] = PSmooth

    xhatSmooth = np.atleast_2d(xhatSmooth).T

    for k in range(N - 1, 0, -1):
        print(k)
        K = ko.P_plus_arr[:, :, k - 1] @ ko.A.T @ np.linalg.inv((ko.P_minus_arr[:, :, k]))
        PSmooth = ko.P_plus_arr[:, :, k - 1] - K @ (ko.P_minus_arr[:, :, k] - PSmooth) @ K.T
        xhatSmooth = np.array(
            np.atleast_2d(ko.x_plus_arr[:, k - 1]).T + K @ (xhatSmooth - np.atleast_2d(ko.x_minus_arr[:, k]).T))

        xhatSmoothArr[:, k - 1] = xhatSmooth.flatten()
        KSmootherArr[:, :, k - 1] = K
        PSmootherArr[:, :, k - 1] = PSmooth

    return RtsOutput(
        x_smooth=xhatSmoothArr,
        K_smooth=KSmootherArr,
        P_smooth=PSmootherArr
    )



def strategy_acceleration(rts_output: RtsOutput):
    N = rts_output.x_smooth.shape[1]
    # assert N == 200

    # Constraint matrices
    ak = np.matrix([
        [-0.5],
        [0.5],
        [-5]
    ])

    bk = np.matrix([
        [0.5],
        [5],
        [-0.5]
    ])

    D = np.matrix([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])

    pdf_lane_keep = []
    pdf_lane_change_left = []
    pdf_lane_change_right = []

    for i in range(N):
        P_plus = rts_output.P_smooth[:, :, i]
        x_est_plus = rts_output.x_smooth[:, i]

        for jj in range(3):
            PTrunc = np.copy(P_plus)
            xTrunc = np.copy(x_est_plus)
            Utrunc, Wtrunc, Vtrunc = np.linalg.svd(PTrunc)
            Ttrunc = np.copy(Utrunc)
            Wtrunc = np.diag(Wtrunc)
            Amgs = np.sqrt(Wtrunc) @ Utrunc.T @ D[0, :].T

            Wmgs, S = mod_gramm_schmit(A=Amgs)

            PTrunc = 0.5 * (PTrunc + PTrunc.T)
            S = S * float(np.sqrt(D[0, :] @ PTrunc @ D[0, :].T)) / float(Wmgs)
            cTrunc = (ak[jj] - D[jj, :] @ xTrunc) / np.sqrt(D[jj, :] @ PTrunc @ D[jj, :].T)
            dTrunc = (bk[jj] - D[jj, :] @ xTrunc) / np.sqrt(D[jj, :] @ PTrunc @ D[jj, :].T)


            alpha = np.sqrt(2 / np.pi) / (erf(dTrunc / np.sqrt(2)) - erf(cTrunc / np.sqrt(2)))
            mu = alpha * (np.exp(-cTrunc ** 2 / 2) - np.exp(-dTrunc ** 2 / 2))
            sigma2 = alpha * (np.exp(-cTrunc ** 2 / 2) * (cTrunc - 2 * mu) - np.exp(
                -dTrunc ** 2 / 2) * (dTrunc - 2 * mu)) + mu ** 2 + 1

            # if np.isinf(alpha):
            #     mu = cTrunc
            #     sigma2 = 0

            zTrunc = np.zeros_like(xTrunc)
            zTrunc[0] = mu
            CovZ = np.eye(zTrunc.size)
            CovZ[0, 0] = sigma2
            xTrunc = Ttrunc @ scipy.linalg.sqrtm(Wtrunc) @ S.T @ zTrunc + xTrunc
            PTruncnew = Ttrunc @ scipy.linalg.sqrtm(Wtrunc) @ S.T @ CovZ @ S @ scipy.linalg.sqrtm(Wtrunc) @ Ttrunc.T
            print(f'step: {i}\talpha {mu}\tmi: {mu}\tsigma2: {sigma2}')
            # print(f'step: {k}\tP={PTruncnew}')
            PTrunc = PTruncnew
            # P_plus = PTrunc
            # x_est_plus = xTrunc

            r = x_est_plus[2] - xTrunc[2]
            S_s = PTruncnew[2, 2]
            pdf = exp(logpdf(x=r, cov=S_s))

            if jj == 0:
                pdf_lane_keep.append(pdf)
            elif jj == 1:
                pdf_lane_change_left.append(pdf)
            elif jj == 2:
                pdf_lane_change_right.append(pdf)
            else:
                raise ValueError

    return pdf_lane_keep, pdf_lane_change_left, pdf_lane_change_right