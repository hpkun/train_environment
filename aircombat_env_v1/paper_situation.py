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

def paper_situation_score(ep,ev,tp,tv):
    """Paper Eq. (11)-(12) weights with project engineering normalizers.

    The 0.35/0.25/0.20/0.20 weights are paper-explicit.  The 14 km binary
    range, 6000 m altitude, and 400 m/s speed mappings are project choices.
    This score is for target selection only and is not part of reward.
    """
    pair=assess_pair(ep,ev,tp,tv)
    e_angle=1.0-(pair.ata_rad+pair.aa_rad)/(2.0*np.pi)
    e_distance=1.0 if pair.distance_m<=14000.0 else 0.0
    e_altitude=pair.relative_altitude_m/6000.0
    e_speed=pair.relative_speed_mps/400.0
    return float(.35*e_angle+.25*e_distance+.20*e_altitude+.20*e_speed)
