"""Paper-aligned perception support for the simple 3v2 heterogeneous scenario.

The MAV/UAV sensing roles are paper-explicit.  The deterministic ranges,
perfect instantaneous data link, and absence of track memory are
paper_unspecified_engineering choices for this project.
"""
from __future__ import annotations

import numpy as np

UAV_DIRECT_DETECTION_RANGE_M=14000.0
MAV_DETECTION_RANGE_M=28000.0
PERCEPTION_MODES=("paper_fused","uav_only_ablation")


class HeterogeneousPerceptionSystem:
    """Build the fixed per-decision perception/fusion result for simple 3v2."""
    def __init__(self,mode="paper_fused"):
        if mode not in PERCEPTION_MODES:
            raise ValueError(f"hetero_perception_mode must be one of {PERCEPTION_MODES}")
        self.mode=mode

    @staticmethod
    def _within(agent,enemies,range_m):
        if not agent.alive:
            return []
        return sorted(e.agent_id for e in enemies if e.alive and
                      float(np.linalg.norm(e.position-agent.position))<=range_m)

    def build(self,agents):
        by_id={a.agent_id:a for a in agents}
        red=sorted((a for a in agents if a.side=="red"),key=lambda a:a.agent_id)
        blue=sorted((a for a in agents if a.side=="blue"),key=lambda a:a.agent_id)
        alive_red=sorted(a.agent_id for a in red if a.alive)
        alive_blue=sorted(a.agent_id for a in blue if a.alive)
        mav=by_id["red_mav_0"]
        mav_detected=self._within(mav,blue,MAV_DETECTION_RANGE_M)
        support_active=bool(self.mode=="paper_fused" and mav.alive)
        direct={};shared={};visible={};hidden={};relay_by_uav={}
        for agent in sorted(agents,key=lambda a:a.agent_id):
            if agent.side=="blue":
                direct_ids=list(alive_red) if agent.alive else []
                shared_ids=[]
                visible_ids=list(direct_ids)
                hidden_ids=sorted(set(alive_red)-set(visible_ids))
            elif agent.role=="mav":
                direct_ids=list(mav_detected) if agent.alive else []
                shared_ids=[]
                visible_ids=list(direct_ids)
                hidden_ids=sorted(set(alive_blue)-set(visible_ids))
            else:
                direct_ids=self._within(agent,blue,UAV_DIRECT_DETECTION_RANGE_M)
                shared_ids=list(mav_detected) if support_active and agent.alive else []
                visible_ids=sorted(set(direct_ids)|set(shared_ids))
                hidden_ids=sorted(set(alive_blue)-set(visible_ids))
                relay_by_uav[agent.agent_id]=sorted(set(shared_ids)-set(direct_ids))
            direct[agent.agent_id]=direct_ids
            shared[agent.agent_id]=shared_ids
            visible[agent.agent_id]=visible_ids
            hidden[agent.agent_id]=hidden_ids
        relay_count=sum(len(ids) for ids in relay_by_uav.values())
        return {"direct_enemy_ids_by_agent":direct,"shared_enemy_ids_by_agent":shared,
          "visible_enemy_ids_by_agent":visible,"hidden_enemy_ids_by_agent":hidden,
          "mav_detected_enemy_ids":list(mav_detected),"mav_support_active":support_active,
          "relay_only_track_count":int(relay_count),"relay_only_tracks_by_uav":relay_by_uav}
