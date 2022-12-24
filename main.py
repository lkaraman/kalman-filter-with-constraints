from matplotlib import pyplot as plt

from import_ngsim import read_ngsim_csv
from kalman import apply_2d_ca_cv_filter, apply_rts, strategy_acceleration

if __name__ == '__main__':
    vehicles = read_ngsim_csv()

    for i, data in enumerate(vehicles):
        kalman_output = apply_2d_ca_cv_filter(data)
        rts_o = apply_rts(kalman_output)

        a = kalman_output.x_plus_arr[2, :]
        a_rts = rts_o.x_smooth[2, :]

        # plt.plot(data.t, data.d, label='raw')
        plt.plot(data.t, a, label='Kalman')
        plt.plot(data.t, a_rts, label='RTS')
        plt.legend()
        plt.title('a')
        plt.figure(2)

        pdf_lane_keep, acceleration, deacceleration = strategy_acceleration(rts_output=rts_o)

        plt.plot(data.t, pdf_lane_keep, label='cruise')
        plt.plot(data.t, acceleration, label='acceleration')
        plt.plot(data.t, deacceleration, label='deacceleration')
        plt.legend()
        plt.title(f'probs for {str(i)}')
        plt.show()
