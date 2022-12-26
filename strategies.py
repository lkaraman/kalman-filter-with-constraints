import numpy as np

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

    strategy_lane_change = BehaviorStrategy(
        name='LateralMovement',
        group_names=(
            'lcl', 'lk', 'lcr'
        ),
        ak=ak,
        bk=bk,
        D=D
    )

    return strategy_lane_change