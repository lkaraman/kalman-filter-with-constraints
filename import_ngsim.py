from typing import List

import numpy as np
import pandas as pd

from structs import Vehicle
FINAL_VEH_ID = 20

def import_vehicles() -> List[Vehicle]:


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