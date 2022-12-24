import numpy as np
from matplotlib import pyplot as plt

from import_ngsim import read_ngsim_csv
from kalman import KalmanFilterWithConstraints
from structs import BehaviorStrategy

def create_longitudinal_strategy() -> BehaviorStrategy:
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

    strategy_acceleration = BehaviorStrategy(
        name='LongitudinalMovement',
        group_names=(
            'cruise', 'acceleration', 'braking'
        ),
        ak=ak,
        bk=bk,
        D=D
    )

    return strategy_acceleration

def create_lateral_strategy() -> BehaviorStrategy:
    ak = np.matrix([
        [-2],
        [-0.25],
        [0.25]
    ])

    bk = np.matrix([
        [-0.25],
        [0.25],
        [2]
    ])

    D = np.matrix([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1]
    ])

    strategy_acceleration = BehaviorStrategy(
        name='LateralMovement',
        group_names=(
            'lcl', 'lk', 'lcr'
        ),
        ak=ak,
        bk=bk,
        D=D
    )

    return strategy_acceleration


if __name__ == '__main__':
    vehicles = read_ngsim_csv()
    vehicle_measurements = vehicles[5]

    kfc = KalmanFilterWithConstraints()

    kfc.behavior_strategy = create_lateral_strategy()

    kalman_out = kfc.apply_2d_ca_cv_filter(vehicle_measurements=vehicle_measurements)
    rts_out = kfc.apply_rts_filter(ko=kalman_out)
    probs = kfc.apply_constraint_filter(rts_output=rts_out)

    a = kalman_out.x_plus_arr[3, :]
    a_rts = rts_out.x_smooth[3, :]

    # plt.plot(data.t, data.d, label='raw')
    plt.plot(vehicle_measurements.s, a, label='Kalman')
    plt.plot(vehicle_measurements.s, a_rts, label='RTS')
    plt.legend()
    plt.title('a')
    plt.figure(2)


    plt.plot(vehicle_measurements.s, probs[0], label='lcr')
    plt.plot(vehicle_measurements.s, probs[1], label='lk')
    plt.plot(vehicle_measurements.s, probs[2], label='lcl')
    plt.legend()
    # plt.title(f'probs for {str(i)}')
    plt.show()

