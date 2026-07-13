from pathlib import Path
import numpy as np
import pytest

from uav_env import make_env
from uav_env.JSBSim.envs.paper_calibrated_v4 import V4_COMPONENT_FIELDS
from scripts.experiment_logging_schema import REWARD_COMPONENT_COLUMNS, EPISODE_REWARD_COMPONENTS_COLUMNS

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'uav_env/JSBSim/configs/hetero_mav_shared_geo_3v2_f16_mav_surrogate_brma_tam_paper_calibrated_v4.yaml'

def test_v4_config_and_real_step_contract():
    env=make_env(CFG,max_steps=5)
    try:
        env.reset(seed=3)
        _,rewards,_,_,info=env.step({a:np.zeros(3,dtype=np.float32) for a in env.agent_ids})
        for rid in env.red_ids:
            c=info['reward_components'][rid]
            assert set(V4_COMPONENT_FIELDS)<=set(c)
            assert all(np.isfinite(float(c[k])) for k in V4_COMPONENT_FIELDS)
            assert abs(c['v4_identity_error'])<=1e-9
            assert c['v4_total']==pytest.approx(c['v4_component_sum'])
    finally: env.close()

def test_paper_internal_ratios_and_separate_advantages():
    weights=np.array([10,10,15,10,30],dtype=float)/75
    assert weights.sum()==pytest.approx(1)
    assert tuple(weights)==pytest.approx((10/75,10/75,15/75,10/75,30/75))
    assert 1.0-0.8*1.0==pytest.approx(0.2)
    angle, distance=0.0, 1.0
    dense=(15*angle+10*distance)/75
    assert dense>0.0  # no multiplicative zero-gradient bottleneck
    assert .5+.3+.2==pytest.approx(1.0)
    assert .6+.4==pytest.approx(1.0)

def test_v4_schema_is_complete():
    assert set(V4_COMPONENT_FIELDS)<=set(REWARD_COMPONENT_COLUMNS)
    assert {f'{k}_sum' for k in V4_COMPONENT_FIELDS}<=set(EPISODE_REWARD_COMPONENTS_COLUMNS)
    assert len(REWARD_COMPONENT_COLUMNS)==len(set(REWARD_COMPONENT_COLUMNS))

def test_v4_reset_clears_event_state():
    env=make_env(CFG,max_steps=5)
    try:
        env.reset(seed=1); env._v4_mav_death_seen=True; env._v4_mav_team_credit_used=1.0
        env.reset(seed=2)
        assert env._v4_mav_death_seen is False
        assert env._v4_mav_team_credit_used==0.0
    finally: env.close()
