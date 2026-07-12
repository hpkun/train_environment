from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from uav_env.JSBSim.envs.role_situation_v3 import (
    _compute_pair_quality,
    collect_v3_effective_samples,
    compute_v3_reward,
)


CFG = {
    "task": {"attrition_scale": 10.0, "attack_uav_loss_weight": 1.0, "mav_loss_weight": 0.75,
             "decisive_win_bonus": 10.0, "decisive_loss_penalty": -10.0, "timeout_advantage_bonus": 2.0},
    "situation": {"softmax_temperature": 0.2, "threat_weight": 1.0, "local_weight": 0.6,
                  "team_weight": 0.4, "speed_modulation_min": 0.75, "speed_modulation_max": 1.0,
                  "distance_optimal_low_ratio": 0.35, "distance_optimal_high_ratio": 0.75},
    "mav": {"marginal_information_weight": 0.5, "support_position_weight": 0.3,
            "threat_weight": 0.4, "role_scale": 0.05, "support_min_distance_ratio": 0.5,
            "support_max_distance_ratio": 1.5, "rear_reference_ratio": 0.75},
    "uav": {"situation_scale": 0.05}, "flight": {"mav_scale": 0.01, "uav_scale": 0.01},
}


class Sim:
    def __init__(self, name, pos, speed=250.0, alive=True, warning=False):
        self.name = name; self.pos = np.asarray(pos, dtype=float); self.speed = speed
        self.is_alive = alive; self.warning = warning

    def get_position(self): return self.pos.copy()
    def get_velocity(self): return np.array([self.speed, 0.0, 0.0])
    def check_missile_warning(self): return object() if self.warning else None


class FakeEnv:
    MISSILE_LAUNCH_RANGE_THRESH = 14000.0
    MISSILE_LAUNCH_MIN_RANGE = 500.0

    def __init__(self, red, blue, roles, geometry, tracks=None, max_steps=1000, current_step=1):
        self.red_planes = {s.name: s for s in red}; self.blue_planes = {s.name: s for s in blue}
        self.red_ids = list(self.red_planes); self.blue_ids = list(self.blue_planes); self.agent_roles = roles
        self.geometry = geometry; self.tracks = tracks or {}; self.max_steps = max_steps; self.current_step = current_step
        self.brma_tam_role_situation_v3_config = copy.deepcopy(CFG)
        self._missile_launch_range_m_effective = 14000.0; self._missile_launch_min_range_m_effective = 500.0
        self._v3_alive_before = {rid: sim.is_alive for rid, sim in self.red_planes.items()}
        self._v3_episode_state = {"prev_blue_dead": 0, "prev_attack_dead": 0, "prev_mav_dead": 0,
                                  "prev_blue_loss": 0.0, "prev_attack_loss": 0.0,
                                  "terminal_applied": False, "latest_j_combat": 0.0}

    def _brma_tam_3d_geometry(self, attacker, target):
        ata, aa, distance = self.geometry[(attacker.name, target.name)]
        return {"tam_ata_rad": ata, "tam_aa_rad": aa, "target_distance_m": distance}

    @staticmethod
    def _brma_tam_safe_vec(sim, method): return np.asarray(getattr(sim, method)(), dtype=float)
    def _mav_shared_track_state(self, uid, bid): return self.tracks.get((uid, bid), {"mav_shared_visible": False, "direct_visible": False})


def run(env, flight=None):
    components = {rid: {"r_pitch": 0.0, "r_roll": 0.0, "r_vel": 0.0} for rid in env.red_ids}
    for rid, values in (flight or {}).items(): components[rid].update(values)
    _, components = compute_v3_reward(env, {rid: 0.0 for rid in env.red_ids}, components)
    return components


def one_uav_case(forward, reverse, distance=7000.0, red_speed=250.0, blue_speed=250.0):
    mav=Sim("red_0", [-10000,0,6500]); uav=Sim("red_1", [0,0,6000], red_speed); blue=Sim("blue_0", [7000,0,6000], blue_speed)
    geom={(uav.name,blue.name):(*forward,distance),(blue.name,uav.name):(*reverse,distance),
          (blue.name,mav.name):(math.pi,0.0,17000.0)}
    return FakeEnv([mav,uav],[blue],{"red_0":"mav","red_1":"attack_uav"},geom)


def test_complete_uav_geometry_ordering():
    cases={
        "tail": one_uav_case((0.0,0.0),(math.pi,0.0)),
        "side_rear": one_uav_case((0.3,0.3),(2.8,0.0)),
        "head_on": one_uav_case((0.5,0.5),(0.8,0.5)),
        "far_neutral": one_uav_case((0.5,0.5),(0.5,0.5),30000.0),
        "tailed": one_uav_case((math.pi,0.0),(0.0,0.0)),
    }
    result={}
    for name,env in cases.items():
        c=run(env)["red_1"]; g=env.geometry[("red_1","blue_0")]
        pq=_compute_pair_quality(env.red_planes["red_1"],env.blue_planes["blue_0"],env,500,4900,10500,14000,.75,1)
        result[name]=c["role_situation_v3_uav_situation_raw"]
        print(name,"ATA",g[0],"AA",g[1],"distance",g[2],"speed_mod",pq["modulation"],
              "offense",c["role_situation_v3_uav_local_offense_raw"],"threat",c["role_situation_v3_uav_local_threat_raw"],"S",result[name])
    assert result["tail"] > result["side_rear"] > result["head_on"] > result["far_neutral"] > result["tailed"]
    assert result["tailed"] < 0.0
    assert one_uav_case((math.pi,0.0),(math.pi,0.0),7000.0) and result["tail"] > result["far_neutral"]


def _matrix_case(offense, threat):
    mav=Sim("red_0",[-10000,0,6500]); uavs=[Sim(f"red_{i+1}",[0,i*1000,6000]) for i in range(2)]
    blues=[Sim(f"blue_{j}",[7000,j*1000,6000]) for j in range(2)]; geom={}
    for i,u in enumerate(uavs):
        for j,b in enumerate(blues):
            geom[(u.name,b.name)]=(math.pi*(1-offense[i][j]),0.0,7000.0)
            geom[(b.name,u.name)]=(math.pi*(1-threat[i][j]),0.0,7000.0)
    for b in blues: geom[(b.name,mav.name)]=(math.pi,0.0,17000.0)
    return FakeEnv([mav,*uavs],blues,{"red_0":"mav","red_1":"attack_uav","red_2":"attack_uav"},geom)


def test_team_coverage_and_exposure_ordering():
    distributed=_matrix_case([[1,.1],[.1,1]],[[.1,.1],[.1,.1]])
    concentrated=_matrix_case([[1,.1],[1,.1]],[[.1,.1],[.1,.1]])
    local=_matrix_case([[.6,.6],[.6,.6]],[[.9,.1],[.1,.1]])
    exposed=_matrix_case([[.6,.6],[.6,.6]],[[.9,.9],[.9,.9]])
    vals={}
    for name,env in (("distributed",distributed),("concentrated",concentrated),("local",local),("exposed",exposed)):
        c=run(env); vals[name]=c
        o=[[round(1-env.geometry[(f"red_{i+1}",f"blue_{j}")][0]/math.pi,6) for j in range(2)] for i in range(2)]
        t=[[round(1-env.geometry[(f"blue_{j}",f"red_{i+1}")][0]/math.pi,6) for j in range(2)] for i in range(2)]
        print(name,"O",o,"T",t,"coverage",c["red_1"]["role_situation_v3_team_coverage_raw"],"exposure",c["red_1"]["role_situation_v3_team_exposure_raw"],
              "S",[c[r]["role_situation_v3_uav_situation_raw"] for r in ("red_1","red_2")])
    assert vals["distributed"]["red_1"]["role_situation_v3_team_coverage_raw"] > vals["concentrated"]["red_1"]["role_situation_v3_team_coverage_raw"]
    assert np.mean([vals["exposed"][r]["role_situation_v3_uav_situation_raw"] for r in ("red_1","red_2")]) < np.mean([vals["local"][r]["role_situation_v3_uav_situation_raw"] for r in ("red_1","red_2")])


def mav_case(mav_pos, shared=False, direct=False, warning=False, blue_to_mav=(math.pi,0.0,17000.0), n=2):
    mav=Sim("red_0",mav_pos,warning=warning); uavs=[Sim(f"red_{i+1}",[0,(i%2)*1000,6000]) for i in range(n)]
    blues=[Sim(f"blue_{j}",[10000,(j%2)*1000,6000]) for j in range(n)]; geom={}; tracks={}
    for u in uavs:
        for b in blues:
            geom[(u.name,b.name)]=(0.0,0.0,7000.0); geom[(b.name,u.name)]=(math.pi,0.0,7000.0)
            tracks[(u.name,b.name)]={"mav_shared_visible":shared,"direct_visible":direct}
    for b in blues: geom[(b.name,mav.name)]=blue_to_mav
    return FakeEnv([mav,*uavs],blues,{"red_0":"mav",**{u.name:"attack_uav" for u in uavs}},geom,tracks)


def test_complete_mav_role_ordering_and_marginal_information():
    envs={"safe_shared":mav_case([-10000,0,6500],True,False),"safe_no_shared":mav_case([-10000,0,6500]),
          "far_no_support":mav_case([-30000,0,6500]),"dangerous_forward":mav_case([8000,0,6000],False,False,True,(0.0,0.0,2000.0))}
    vals={}
    for name,env in envs.items():
        c=run(env)["red_0"]; vals[name]=c["role_situation_v3_mav_role_raw"]
        print(name,"marginal",c["role_situation_v3_mav_marginal_information_raw"],"distance_q",c["role_situation_v3_mav_support_distance_raw"],
              "rear_q",c["role_situation_v3_mav_support_rear_raw"],"support",c["role_situation_v3_mav_support_position_raw"],
              "geom_threat",c["role_situation_v3_mav_geometric_threat_raw"],"warning",c["role_situation_v3_mav_missile_warning"],
              "threat",c["role_situation_v3_mav_threat_raw"],"S",vals[name])
    assert vals["safe_shared"] > vals["safe_no_shared"] > vals["far_no_support"] > vals["dangerous_forward"]
    assert vals["dangerous_forward"] < 0.0
    full=run(mav_case([-10000,0,6500],True,False,n=4))["red_0"]["role_situation_v3_mav_marginal_information_raw"]
    one=mav_case([-10000,0,6500],False,False,n=4); one.tracks[("red_1","blue_0")]={"mav_shared_visible":True,"direct_visible":False}
    one_v=run(one)["red_0"]["role_situation_v3_mav_marginal_information_raw"]
    direct=run(mav_case([-10000,0,6500],True,True,n=4))["red_0"]["role_situation_v3_mav_marginal_information_raw"]
    print("marginal_16_of_16",full,"marginal_1_of_16",one_v,"direct_visible",direct)
    assert full > one_v > direct == 0.0


def test_flight_penalties_remain_negative_and_expose_clip_limit():
    env=one_uav_case((0,0),(math.pi,0))
    states={"normal":(0,0,0),"pitch":(-1,0,0),"roll":(0,-1,0),"velocity":(0,0,-1),"all":(-1,-1,-1)}
    vals={}
    for name,(p,r,v) in states.items():
        c=run(env,{"red_1":{"r_pitch":p,"r_roll":r,"r_vel":v}})["red_1"]
        vals[name]=c["role_situation_v3_flight_encoded"]
        print(name,p,r,v,c["role_situation_v3_flight_raw"],c["role_situation_v3_flight_norm"],vals[name])
    assert vals["normal"] == 0.0
    assert vals["pitch"] < 0 and vals["roll"] < 0 and vals["velocity"] < 0
    assert vals["all"] == vals["pitch"]  # frozen clip[-1, 1] prevents strict stacking


def task_case(blue_alive, attack_alive, mav_alive, timeout=False):
    mav=Sim("red_0",[-10000,0,6500],alive=mav_alive); uavs=[Sim("red_1",[0,0,6000],alive=attack_alive),Sim("red_2",[0,1000,6000],alive=attack_alive)]
    blues=[Sim("blue_0",[7000,0,6000],alive=blue_alive),Sim("blue_1",[7000,1000,6000],alive=blue_alive)]
    geom={}
    for u in uavs:
        for b in blues: geom[(u.name,b.name)]=(0,0,7000); geom[(b.name,u.name)]=(math.pi,0,7000)
    for b in blues: geom[(b.name,mav.name)]=(math.pi,0,17000)
    env=FakeEnv([mav,*uavs],blues,{"red_0":"mav","red_1":"attack_uav","red_2":"attack_uav"},geom,current_step=1000 if timeout else 1)
    env._v3_alive_before={rid:True for rid in env.red_ids}
    return env


def test_terminal_and_complete_task_ordering():
    cases={"win_mav_alive":task_case(False,True,True),"win_mav_dead":task_case(False,True,False),
           "partial_kill_no_red_loss":task_case(True,True,True),"no_loss_timeout":task_case(True,True,True,True),
           "red_partial_loss":task_case(True,False,True,True),"red_eliminated":task_case(True,False,False),
           "mutual":task_case(False,False,False)}
    # Partial kill uses one live and one dead blue.
    cases["partial_kill_no_red_loss"].blue_planes["blue_1"].is_alive=False
    values={}
    for name,env in cases.items():
        c=run(env)["red_0"]
        values[name]=c["role_situation_v3_task_attrition"]+c["role_situation_v3_task_terminal"]
        print(name,"dblue",c["role_situation_v3_blue_loss_delta"],"duav",c["role_situation_v3_uav_loss_delta"],
              "dmav",c["role_situation_v3_mav_loss_delta"],"attr",c["role_situation_v3_task_attrition"],
              "terminal",c["role_situation_v3_task_terminal"],"task",values[name],"J",c["role_situation_v3_j_combat"])
    assert values["win_mav_alive"] > values["win_mav_dead"] > values["partial_kill_no_red_loss"] > values["no_loss_timeout"] > values["red_partial_loss"] > values["red_eliminated"]
    assert run(cases["mutual"])["red_0"]["role_situation_v3_task_terminal"] == 0.0
    env=task_case(False,True,True); first=run(env)["red_0"]["role_situation_v3_task_terminal"]; second=run(env)["red_0"]["role_situation_v3_task_terminal"]
    assert first == 10.0 and second == 0.0


def test_real_v3_path_scale_copy_3v2_to_5v4(capsys):
    env3=mav_case([-10000,0,6500],True,False,n=2)
    env5=mav_case([-10000,0,6500],True,False,n=4)
    c3=run(env3); c5=run(env5)
    pairs={
        "local_offense": (c3["red_1"]["role_situation_v3_uav_local_offense_raw"],c5["red_1"]["role_situation_v3_uav_local_offense_raw"]),
        "local_threat": (c3["red_1"]["role_situation_v3_uav_local_threat_raw"],c5["red_1"]["role_situation_v3_uav_local_threat_raw"]),
        "coverage": (c3["red_1"]["role_situation_v3_team_coverage_raw"],c5["red_1"]["role_situation_v3_team_coverage_raw"]),
        "exposure": (c3["red_1"]["role_situation_v3_team_exposure_raw"],c5["red_1"]["role_situation_v3_team_exposure_raw"]),
        "uav_situation": (c3["red_1"]["role_situation_v3_uav_situation_raw"],c5["red_1"]["role_situation_v3_uav_situation_raw"]),
        "mav_information": (c3["red_0"]["role_situation_v3_mav_marginal_information_raw"],c5["red_0"]["role_situation_v3_mav_marginal_information_raw"]),
        "mav_support": (c3["red_0"]["role_situation_v3_mav_support_position_raw"],c5["red_0"]["role_situation_v3_mav_support_position_raw"]),
        "mav_threat": (c3["red_0"]["role_situation_v3_mav_threat_raw"],c5["red_0"]["role_situation_v3_mav_threat_raw"]),
        "mav_role": (c3["red_0"]["role_situation_v3_mav_role_raw"],c5["red_0"]["role_situation_v3_mav_role_raw"]),
    }
    n3,n5=len(env3.red_ids),len(env5.red_ids)
    pairs["mav_team_mean_contribution"]=(c3["red_0"]["role_situation_v3_mav_role_encoded"]/n3,c5["red_0"]["role_situation_v3_mav_role_encoded"]/n5)
    pairs["uav_team_mean_contribution"]=(sum(c3[r]["role_situation_v3_uav_situation_encoded"] for r in env3.red_ids[1:])/n3,
                                          sum(c5[r]["role_situation_v3_uav_situation_encoded"] for r in env5.red_ids[1:])/n5)
    pairs["flight_team_mean"]=(sum(c3[r]["role_situation_v3_flight_encoded"] for r in env3.red_ids)/n3,
                                sum(c5[r]["role_situation_v3_flight_encoded"] for r in env5.red_ids)/n5)
    for name,(v3,v5) in pairs.items():
        error=abs(v3-v5); print(name,"3v2",v3,"5v4",v5,"abs_error",error); assert error <= 1e-6


def test_current_step_death_uses_alive_before_then_excludes_dead_before():
    env=mav_case([-10000,0,6500],True,False,n=2)
    env.red_planes["red_2"].is_alive=False
    env._v3_alive_before={rid:True for rid in env.red_ids}
    components=run(env)
    sums,counts=collect_v3_effective_samples(components,env.agent_roles,np.ones(3),env.red_ids,step=1)
    code=sums["effective_role_situation_v3_total"]/counts["effective_role_situation_v3_total"]
    manual=sum(components[r]["role_situation_v3_total"] for r in env.red_ids)/3.0
    print("death_transition_alive_before", "code", code, "manual", manual, "count", counts["effective_role_situation_v3_total"])
    assert code == pytest.approx(manual)
    active=np.array([1,1,0],dtype=float)
    sums2,counts2=collect_v3_effective_samples(components,env.agent_roles,active,env.red_ids,step=2)
    manual2=(components["red_0"]["role_situation_v3_total"]+components["red_1"]["role_situation_v3_total"])/2.0
    print("next_step_dead_before", "code", sums2["effective_role_situation_v3_total"]/counts2["effective_role_situation_v3_total"], "manual", manual2, "count", counts2["effective_role_situation_v3_total"])
    assert sums2["effective_role_situation_v3_total"]/counts2["effective_role_situation_v3_total"] == pytest.approx(manual2)
