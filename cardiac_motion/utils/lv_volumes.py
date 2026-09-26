'''
Left-ventricular cavity volumes from reconstructed meshes, to find each subject's end-systolic
frame (the one with the smallest cavity volume).

The cavity volume of a frame is approximated by the volume of the convex hull of its endocardial
vertices (cardio_mesh.get_lv_wall_labels(...) == "endo"). On a sample of 200 subjects this gave
~127 ml at the first frame and (max - min) / max ~ 0.61, i.e. plausible end-diastolic volumes and
ejection fractions; hulls of all LV vertices measure the epicardial volume instead (cavity + wall).
'''
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

MM3_PER_ML = 1000.0


def endocardial_volumes(endo_vertices: np.ndarray) -> np.ndarray:
    '''
    endo_vertices: (T, V_endo, 3) in mm, one subject's frames -> (T,) convex hull volumes in ml.
    '''
    return np.array([ConvexHull(frame).volume for frame in endo_vertices]) / MM3_PER_ML


def volume_table(subject_ids, volumes: np.ndarray, frames) -> pd.DataFrame:
    '''
    subject_ids: (N,); volumes: (N, T) in ml; frames: the T frame numbers (1-based, as in
    phases_filter). One row per subject with the volume at every frame and
      es_frame: frame of minimum volume (end systole); ed_frame: frame of maximum volume;
      edv_ml, esv_ml: maximum and minimum volume; ef: (edv - esv) / edv.
    '''
    frames = np.asarray(frames)
    table = pd.DataFrame(volumes, columns=[f"vol_frame{f:02d}" for f in frames])
    table.insert(0, "subject_id", list(subject_ids))
    table["es_frame"] = frames[volumes.argmin(axis=1)]
    table["ed_frame"] = frames[volumes.argmax(axis=1)]
    table["edv_ml"] = volumes.max(axis=1)
    table["esv_ml"] = volumes.min(axis=1)
    table["ef"] = (table.edv_ml - table.esv_ml) / table.edv_ml
    return table
