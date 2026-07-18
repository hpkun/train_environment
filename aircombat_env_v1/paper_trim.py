"""Local engineering trim state for the formal JSBSim paper environment."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
import json
from pathlib import Path

@dataclass(frozen=True)
class PaperTrimState:
    altitude_m:float=6000.
    true_airspeed_mps:float=250.
    heading_deg:float=0.
    flight_path_angle_deg:float=0.
    pitch_deg:float=0.
    alpha_deg:float=0.
    beta_deg:float=0.
    roll_deg:float=0.
    throttle:float=.4
    aileron:float=0.
    elevator:float=0.
    rudder:float=0.
    source:str="paper_unspecified_engineering"
    validation_metrics:dict=field(default_factory=dict)

    def controls(self):return (self.throttle,self.aileron,self.elevator,self.rudder)
    def to_dict(self):return asdict(self)

def load_trim(path):return PaperTrimState(**json.loads(Path(path).read_text(encoding="utf-8")))
def save_trim(trim,path):Path(path).write_text(json.dumps(trim.to_dict(),indent=2),encoding="utf-8")
