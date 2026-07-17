"""Paper Eq. (10)-(12) pair geometry."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _angle(a,b):
    return float(np.arccos(np.clip(np.dot(a,b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-8),-1,1)))

@dataclass(frozen=True)
class SituationScore:
    relative_speed_mps: float; relative_altitude_m: float; distance_m: float
    ata_rad: float; aa_rad: float

def assess_pair(ep,ev,tp,tv):
    los=np.asarray(tp,dtype=float)-np.asarray(ep,dtype=float)
    ev=np.asarray(ev,dtype=float); tv=np.asarray(tv,dtype=float)
    return SituationScore(float(np.linalg.norm(ev-tv)),float(ep[2]-tp[2]),
        float(np.linalg.norm(los)),_angle(ev,los),_angle(tv,los))
