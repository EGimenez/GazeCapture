# The purpose of this class is to properly record the difference between the real x, y and the predicted ones

from typing import Dict
import pandas as pd
import os
from pathlib import Path

class ITrackerDiff(object):
    def __init__(self, path):
        self.faces: Dict[str, pd.DataFrame] = {}
        self.path = Path(path)

    def append(self, face_id, frame_id, x, y, x_m, y_m, x_d, y_d, d):
        try:
            self.faces[face_id].loc[int(frame_id)] = [face_id, frame_id, x, y, x_m, y_m, x_d, y_d, d]
        except:
            self.faces[face_id] = pd.DataFrame(columns=['face_id', 'frame_id', 'x', 'y', 'x_model', 'y_model', 'x_diff', 'y_diff', 'dist'])
            self.faces[face_id].loc[int(frame_id)] = [face_id, frame_id, x, y, x_m, y_m, x_d, y_d, d]

    def serialize(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        for f in self.faces.keys():
            df = self.faces[f].to_csv(self.path / ( f + '.csv'), index=False)


