"""Audit original BRMA-MAPPO training protocol vs hetero_uav paper-aligned.
Read-only. No training."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
PARENT = ROOT.parent

def read_file(path, max_lines=0):
    p = PARENT / path
    if p.exists(): return p.read_text(encoding="utf-8", errors="replace")[:max_lines or None]
    return None

def check_exists(rel_path):
    return (PARENT / rel_path).exists()

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", default="outputs/protocol_audit/original_training_protocol_alignment.json")
    p.add_argument("--output-md", default="outputs/protocol_audit/original_training_protocol_alignment.md")
    args = p.parse_args()

    original = dict(
        files_found=dict(
            train_vanilla_mappo=check_exists("train_vanilla_mappo.py"),
            train_ppo=check_exists("train_ppo.py"),
            train_attention_mappo=check_exists("train_attention_mappo.py"),
            rule_based_agent=check_exists("rule_based_agent.py"),
        ),
        algorithm=dict(
            name="Attention MAPPO with GRU (train_ppo.py) / GRU MLP (train_vanilla_mappo.py)",
            attention="4-head entity attention (train_ppo.py Config)",
            recurrent="GRU RNN hidden=128 (both scripts)",
            centralized_critic="yes (implicit in MAPPO shared critic)",
            parameter_sharing="yes (shared actor across red agents)",
        ),
        scale=dict(
            total_env_steps="10,000,000 (10M)",
            num_envs="8 (vanilla) / 32 (attention)",
            rollout_buffer_size=2000,
            ppo_epochs=10,
            minibatches="4-8",
            entropy_coef=0.05,
            learning_rate_actor=2e-4,
            learning_rate_critic=5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            max_grad_norm=5.0,
            max_episode_length=1400,
        ),
        network=dict(
            mlp_hidden=128,
            rnn_hidden_size=128,
            feature_dim=11,
            num_heads=4,
        ),
        composition=dict(
            start_config="2v2 (vanilla recommended) or 6v6 (full scale)",
            blue_opponent="rule_based_agent.blue_coordinated_actions from step 0",
            curriculum="NOT found — no evidence of curriculum or warmup opponent",
            blue_gcas="False in vanilla training",
            aircraft="F-16 for all agents",
            heterogeneity="NONE — all aircraft are same model (F-16), no MAV/UAV distinction",
        ),
        action_interface=dict(
            pitch_range="[-90, +90] deg (BRMA paper eq)",
            heading="absolute heading ([-180, +180])",
            action_clamp="clipped to [-1,1] by env, but no safe-action wrapper found",
            pitch_trim="NOT found in original — no per-agent action_trim_by_role",
            low_altitude_safety="GCAS exists but disabled for blue in vanilla training",
        ),
        reward=dict(
            weights="r_pitch=0.01, r_roll=0.002, r_vel=0.02, r_alt=0.04, r_bound=0.04, r_adv=0.15 — matches BRMA paper Table 4",
            r_end="30*(N_alive-N_enemy)/N_max per teammate",
            crash_penalty="r_death=-10",
            matches_hetero_brma_legacy="YES — hetero brma_legacy uses same weights and formulas",
        ),
    )

    hetero = dict(
        algorithm=dict(
            name="Shared MLP MAPPO (no attention, no GRU)",
            attention="NONE",
            recurrent="NONE",
            centralized_critic="yes",
            parameter_sharing="yes (shared MLP actor)",
        ),
        scale=dict(
            total_env_steps="50,000",
            num_envs=1,
            rollout_buffer_size=128,
            ppo_epochs=4,
            minibatches=4,
            entropy_coef="~0.01 (default in PPOTrainer)",
            learning_rate="default",
            max_grad_norm=10.0,
            max_episode_length=1000,
        ),
        composition=dict(
            train_config="3v2 (red=1 MAV + 2 attack_UAV, blue=2 attack_UAV)",
            blue_opponent="brma_rule (delegates to parent rule_based_agent.py)",
            mav_trim="pitch=0.10 (ORIGINALLY) / now 0.0 in no_mav_trim config",
            aircraft="F-22 MAV (no missiles), F-16 UAV (2 missiles each)",
            heterogeneity="YES — MAV/UAV role distinction, MAV has no missiles",
        ),
        paper_aligned_50k_result=dict(
            red_win_3v2=0.00,
            blue_elimination_win_rate_3v2=1.00,
            red_win_5v4=0.00,
            blue_elimination_win_rate_5v4=1.00,
            train_red_alive="0.0 from episode 3 onward",
            train_ret="-52.9 at end",
            action_saturation=0.15,
            entropy=0.34,
            best_checkpoint_score=-1.48,
            failure_summary="Red eliminated in every episode from start; no learning signal; policy never improved",
        ),
    )

    gaps = [
        dict(dimension="training_steps", gap="200x", original="10,000,000", hetero="50,000", severity="CRITICAL"),
        dict(dimension="algorithm", gap="attention+GRU vs shared MLP", original="4-head entity attention + GRU MAPPO", hetero="shared MLP MAPPO", severity="MAJOR"),
        dict(dimension="num_envs", gap="8-32x", original="8 (vanilla) / 32 (attention)", hetero="1", severity="MAJOR"),
        dict(dimension="rollout_buffer", gap="15.6x", original="2000", hetero="128", severity="MODERATE"),
        dict(dimension="ppo_epochs", gap="2.5x", original="10", hetero="4", severity="MINOR"),
        dict(dimension="entropy_coef", gap="5x", original="0.05", hetero="~0.01", severity="MINOR"),
        dict(dimension="composition", gap="heterogeneous vs homogeneous", original="2v2/6v6 all F-16", hetero="3v2 with F-22 MAV (unarmed) + F-16 UAVs", severity="MAJOR"),
        dict(dimension="max_episode_length", gap="1.4x", original="1400", hetero="1000", severity="MINOR"),
        dict(dimension="blue_gcas", gap="disabled vs enabled", original="blue GCAS=False (vanilla)", hetero="blue GCAS=True", severity="MODERATE"),
        dict(dimension="aircraft_model", gap="F-22 vs F-16", original="F-16 for all", hetero="F-22 MAV (larger, heavier)", severity="MODERATE"),
        dict(dimension="action_trim", gap="added", original="no per-agent trim", hetero="MAV pitch_trim=0.10 (now 0.0)", severity="MODERATE"),
    ]

    causes = [
        "training_steps (200x gap): 50k is insufficient to learn against a full-capability blue opponent",
        "algorithm: shared MLP without attention/GRU may lack capacity for heterogeneous multi-agent combat",
        "composition: MAV (unarmed F-22) adds a role the MLP must learn to protect, making the task harder than 2v2 F-16-only",
        "blue opponent: brma_rule fires missiles, tracks targets, and eliminates red — unlike rule_nearest which allowed red survival",
    ]

    options = [
        dict(id="A", name="scale_up_steps", description="Run 200k-500k paper-aligned baseline to test whether 50k is simply too short. Parent used 10M.", prerequisite="None, can run now", risk="Time cost; may still fail if algorithm gap is real"),
        dict(id="B", name="align_ppo_hyperparams", description="Increase PPO epochs to 10, entropy to 0.05, rollout to 2000, max_steps to 1400. Match parent protocol as closely as possible without changing network architecture.", prerequisite="Simple config changes", risk="May help but network architecture gap remains"),
        dict(id="C", name="2v2_homogeneous_baseline", description="Run 2v2 all-F-16 (remove MAV) with brma_rule to isolate heterogeneity as a variable. If 2v2 homogeneous works, heterogeneity is the blocker.", prerequisite="Need 2v2 config", risk="Deviates from 3v2 heterogeneous paper goal; serves only as diagnostic"),
    ]

    data = dict(original_project_protocol=original, hetero_uav_paper_aligned_protocol=hetero,
                differences=gaps, likely_failure_causes=causes, minimal_next_options=options)

    md = ["# Original Training Protocol Alignment Audit", "",
          "## Original Project Protocol", f"- Algorithm: {original['algorithm']['name']}",
          f"- Steps: {original['scale']['total_env_steps']}",
          f"- Envs: {original['scale']['num_envs']}",
          f"- Composition: {original['composition']['start_config']}",
          f"- Blue: {original['composition']['blue_opponent']}",
          f"- Curriculum: {original['composition']['curriculum']}",
          f"- Heterogeneity: {original['composition']['heterogeneity']}",
          f"- Reward: matches hetero brma_legacy",
          "", "## Hetero Paper-Aligned Protocol",
          f"- Algorithm: {hetero['algorithm']['name']}",
          f"- Steps: {hetero['scale']['total_env_steps']}",
          f"- Envs: {hetero['scale']['num_envs']}",
          f"- Result: red_win=0.00, blue_elim=1.00",
          "", "## Key Gaps"]
    for g in gaps:
        md.append(f"- **{g['dimension']}**: {g['gap']} ({g['severity']}) — original={g['original']}, hetero={g['hetero']}")
    md.append("")
    md.append("## Likely Failure Causes")
    for c in causes: md.append(f"- {c}")
    md.append("")
    md.append("## Minimal Next Options")
    for o in options: md.append(f"- **{o['id']}**: {o['name']} — {o['description']}")

    for path, content in [(args.output_json, json.dumps(data, indent=2)), (args.output_md, "\n".join(md))]:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
    print(f"output_json: {args.output_json}"); print(f"output_md: {args.output_md}")
    for g in gaps:
        if g["severity"] == "CRITICAL": print(f"CRITICAL GAP: {g['dimension']} ({g['gap']})")

if __name__ == "__main__": main()
