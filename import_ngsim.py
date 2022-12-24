from typing import List

import numpy as np
import pandas as pd

from structs import Vehicle

PATH_TO_CSV = 'data/measurements.csv'

FINAL_VEH_ID = 20
FEET_TO_M = 1 / 3.28


def read_ngsim_csv() -> List[Vehicle]:
    """
    Reads csv file which contains measurements for NGSIM dataset and converts it to internal vehicle structure
    :return: list of vehicles
    """

    df = pd.read_csv(PATH_TO_CSV)
    df = df[df.Location == 'us-101']

    del df['Time_Headway']
    del df['Space_Headway']
    del df['Following']
    del df['Preceding']

    vehicles: List[Vehicle] = []

    for i in range(2, FINAL_VEH_ID):
        df_current = df[df.Vehicle_ID == i]
        df_current = df_current.sort_values(by='Frame_ID')

        unique_frames = df_current.Total_Frames.unique()

        # Split vehicles with same id
        for jj in unique_frames:
            df_temp = df_current[df_current.Total_Frames == jj]
            frames = [int(i) for i in list(df_temp.Frame_ID.values)]
            t = [int(i) for i in list(df_temp.Global_Time.values)]
            t = ((np.asarray(t) - t[0]) / 1000).tolist()
            v = Vehicle(
                id=-1,
                s=list(df_temp.Local_Y.values * FEET_TO_M),
                d=list(df_temp.Local_X.values * FEET_TO_M),
                t=t,
                frames=frames,
            )

            vehicles.append(v)

    return vehicles
