import pathlib
from typing import List

import numpy as np
import shapefile as shp
from matplotlib import pyplot as plt
from shapely.geometry import LineString, MultiLineString


import pandas as pd

from kalman import apply_2d_constant_acceleration_filter, apply_rts, strategy_acceleration
from structs import Vehicle

FINAL_VEH_ID = 20


def import_vehicles(p: pathlib.Path) -> List[Vehicle]:


    df = pd.read_csv('data/measurements.csv')
    df = df[df.Location == 'us-101']

    del df['Time_Headway']
    del df['Space_Headway']
    del df['Following']
    del df['Preceding']

    out = []

    for i in range(2, FINAL_VEH_ID):
        df_current = df[df.Vehicle_ID == i]
        df_current = df_current.sort_values(by='Frame_ID')

        unique_frames = df_current.Total_Frames.unique()

        # Split vehicles with same id

        for jj in unique_frames:
            df_temp = df_current[df_current.Total_Frames == jj]
            frames = [int(i) for i in list(df_temp.Frame_ID.values)]
            t = [int(i) for i in list(df_temp.Global_Time.values)]
            t = ((np.asarray(t) - t[0])/1000).tolist()
            v = Vehicle(
                id=-1,
                s=list(df_temp.Local_Y.values / 3.28),
                d=list(df_temp.Local_X.values / 3.28),
                t=t,
                frames=frames,
            )

            out.append(v)

    return out

if __name__ == '__main__':
    scenario = import_vehicles(None)

    for i, data in enumerate(scenario):
        kalman_output = apply_2d_constant_acceleration_filter(data)
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
