from math import exp
from typing import Optional

import numpy as np
import scipy
from filterpy.stats import logpdf
from scipy.special import erf

from gramm import mod_gramm_schmit
from structs import KalmanOutput, RtsOutput, Vehicle, BehaviorStrategy


class KalmanFilterWithConstraints:
    """
    Implements all relevant Kalman filter methods
        - forward Kalman filter
        - RTS smoother
        - probability truncation approach for constraints
    """

    def __init__(self):
        # Kalman matrices
        self.A = None
        self.C = None
        self.Q = None
        self.R = None
        self.P = None

        self._behavior_strategy: Optional[BehaviorStrategy] = None
        self._kalman_output = None
        self._initialize_kalman_matrices()

    @property
    def behavior_strategy(self):
        return self._behavior_strategy

    @behavior_strategy.setter
    def behavior_strategy(self, value: BehaviorStrategy):
        self._behavior_strategy = value

    def _initialize_kalman_matrices(self) -> None:
        T = 0.1

        self.A = np.matrix([
            [1, T, T ** 2 / 2, 0, 0],
            [0, 1, T, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, T],
            [0, 0, 0, 0, 1]
        ])

        # we can only measure sd coordinates
        self.C = np.matrix([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0]
        ])

        self.Q = np.matrix([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ]) * 0.1

        self.R = np.matrix([
            [1, 0],
            [0, 1]
        ])

        self.P = np.matrix([
            [100, 0, 0, 0, 0],
            [0, 100, 0, 0, 0],
            [0, 0, 100, 0, 0],
            [0, 0, 0, 100, 0],
            [0, 0, 0, 0, 100]
        ])

    def compute_probabilities_from_constraint_filtering(self, vehicle_measurements: Vehicle):

        kalman_output = self.apply_2d_ca_cv_filter(vehicle_measurements=vehicle_measurements)
        rts_output = self.apply_rts_filter(kalman_output)
        probabilities = self.apply_constraint_filter(rts_output)

        return probabilities

    def apply_2d_ca_cv_filter(self, vehicle_measurements: Vehicle) -> KalmanOutput:
        """
        Uses forward ConstantAccelerationConstantVelocity Kalman filter on measurement data
        :param vehicle_measurements: vehicle object containing long/lat measurements + time values
        :return:
        """

        # Initial conditions
        x = np.matrix([
            [vehicle_measurements.s[0]],
            [0],
            [0],
            [vehicle_measurements.d[0]],
            [0]
        ])

        P_plus = self.P.copy()
        x_est_plus = x.copy()

        nx = 5
        ny = 2
        N = len(vehicle_measurements.t)

        PminusArr = np.zeros((nx, nx, N))
        PplusArr = np.zeros((nx, nx, N))
        xhatminusArr = np.zeros((nx, N))
        xhatplusArr = np.zeros((nx, N))
        KArr = np.zeros((nx, ny, N))

        for k, t in enumerate(vehicle_measurements.t):
            y_meas = np.matrix([
                [vehicle_measurements.s[k]],
                [vehicle_measurements.d[k]]
            ])

            # Standard forward Kalman filter
            P_minus = self.A @ P_plus @ self.A.T + self.Q
            K = P_minus @ self.C.T @ np.linalg.inv(self.C @ P_minus @ self.C.T + self.R)
            x_est_minus = self.A @ x_est_plus
            x_est_plus = x_est_minus + K @ (y_meas - self.C @ x_est_minus)
            P_plus = P_minus - K @ self.C @ P_minus

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
            A=self.A

        )

    def apply_rts_filter(self, ko: KalmanOutput) -> RtsOutput:
        nx = ko.A.shape[0]
        N = ko.K_arr.shape[2]
        xhatSmooth = ko.x_plus_arr[:, -1]
        PSmooth = ko.P_plus_arr[:, :, -1]
        xhatSmoothArr = np.zeros((nx, N))
        xhatSmoothArr[:, N - 1] = xhatSmooth
        KSmootherArr = np.zeros((nx, nx, N - 1))
        PSmootherArr = np.zeros((nx, nx, N))
        PSmootherArr[:, :, N - 1] = PSmooth

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

    def apply_constraint_filter(self, rts_output: RtsOutput):
        N = rts_output.x_smooth.shape[1]
        # assert N == 200

        # Constraint matrices
        ak = self._behavior_strategy.ak
        bk = self._behavior_strategy.bk
        D = self._behavior_strategy.D

        num_of_sets = len(self._behavior_strategy.group_names)

        pdfs = [[] for _ in range(num_of_sets)]

        for i in range(N):
            P_plus = rts_output.P_smooth[:, :, i]
            x_est_plus = rts_output.x_smooth[:, i]

            for jj in range(num_of_sets):
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
                # print(f'step: {i}\talpha {mu}\tmi: {mu}\tsigma2: {sigma2}')

                r = x_est_plus - xTrunc
                S_s = PTruncnew
                pdf = exp(logpdf(x=r, cov=S_s))

                pdfs[jj].append(pdf)

        return pdfs
