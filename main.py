from matplotlib import pyplot as plt

from import_ngsim import read_ngsim_csv
from kalman import KalmanFilterWithConstraints
from strategies import create_longitudinal_strategy

if __name__ == '__main__':
    vehicles = read_ngsim_csv()
    vehicle_measurements = vehicles[1]

    kfc = KalmanFilterWithConstraints()

    kfc.behavior_strategy = create_longitudinal_strategy()

    kalman_out = kfc.apply_2d_ca_cv_filter(vehicle_measurements=vehicle_measurements)
    rts_out = kfc.apply_rts_filter(ko=kalman_out)
    probs = kfc.apply_constraint_filter(rts_output=rts_out)

    velocity_forward = kalman_out.x_plus_arr[1, :]
    velocity_rts = rts_out.x_smooth[1, :]

    plt.figure(1)
    plt.plot(vehicle_measurements.s, velocity_forward, label='Kalman')
    plt.plot(vehicle_measurements.s, velocity_rts, label='RTS')
    plt.legend()
    plt.title('Velocity profile')

    plt.figure(2)
    plt.plot(vehicle_measurements.s, probs[0], label='cruise')
    plt.plot(vehicle_measurements.s, probs[1], label='acc')
    plt.plot(vehicle_measurements.s, probs[2], label='brk')
    plt.legend()

    plt.show()
