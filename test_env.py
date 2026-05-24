"""
Standalone test script for my_uav_env — 10v10 scale, 3 random-action episodes.

Episode 1 is recorded to  test_episode.acmi  for TacView visualization.
Usage:  python test_env.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from my_uav_env import UavCombatEnv

MAX_STEPS = 200
NUM_EPISODES = 3
ACMI_OUTPUT = "test_episode.acmi"


def run():
    env = UavCombatEnv(max_num_blue=10, max_num_red=10, max_steps=MAX_STEPS)
    print(f"Env created: {env.max_num_blue}v{env.max_num_red}, "
          f"{len(env.agent_ids)} agents, max_steps={MAX_STEPS}")
    print(f"Action space per agent:  {env.action_space['blue_0']}")
    print(f"Obs keys:  {list(env.observation_space['blue_0'].keys())}")
    print(f"  ego_state:    {env.observation_space['blue_0']['ego_state'].shape}")
    print(f"  ally_states:  {env.observation_space['blue_0']['ally_states'].shape}")
    print(f"  enemy_states: {env.observation_space['blue_0']['enemy_states'].shape}")
    print(f"  death_mask:   {env.observation_space['blue_0']['death_mask'].shape}")
    print("=" * 60)

    for ep in range(1, NUM_EPISODES + 1):
        # ---- Enable recording for episode 1 only ----
        if ep == 1:
            env.render()  # activates TacviewRecorder
            print(f"[Episode {ep}] TacView recording ENABLED → will save to '{ACMI_OUTPUT}'")
        else:
            print(f"[Episode {ep}] (no recording)")

        obs, info = env.reset()
        ep_rewards = {aid: 0.0 for aid in env.agent_ids}
        print_interval = 1 if ep == 1 else 20  # verbose only for recorded episode

        for step in range(1, MAX_STEPS + 1):
            actions = {aid: env.action_space[aid].sample() for aid in env.agent_ids}
            obs, rewards, terminated, truncated, info = env.step(actions)

            for aid in env.agent_ids:
                ep_rewards[aid] += rewards[aid]

            # ---- Console debug ----
            if step % print_interval == 0 or any(terminated.values()) or any(truncated.values()):
                blue_obs = obs["blue_0"]
                total_reward = sum(rewards.values())
                alive_count = sum(1 for v in info.values() if v["alive"])

                print(f"[Ep {ep} | Step {step:3d}] "
                      f"enemy_states.shape={blue_obs['enemy_states'].shape}  "
                      f"death_mask={blue_obs['death_mask'].tolist()}  "
                      f"alive={alive_count}/{len(env.agent_ids)}  "
                      f"total_reward={total_reward:+.3f}")

            # ---- Termination handling ----
            if any(terminated.values()) or any(truncated.values()):
                blue_alive = sum(1 for s in env.blue_planes.values() if s.is_alive)
                red_alive = sum(1 for s in env.red_planes.values() if s.is_alive)

                if any(truncated.values()):
                    print(f"  >>> Episode {ep} ended at step {step}: TIMEOUT "
                          f"(blue_alive={blue_alive}, red_alive={red_alive})")
                elif blue_alive == 0:
                    print(f"  >>> Episode {ep} ended at step {step}: RED WINS "
                          f"(blue_alive=0, red_alive={red_alive})")
                elif red_alive == 0:
                    print(f"  >>> Episode {ep} ended at step {step}: BLUE WINS "
                          f"(blue_alive={blue_alive}, red_alive={red_alive})")
                else:
                    dead = [aid for aid, t in terminated.items() if t]
                    print(f"  >>> Episode {ep} ended at step {step}: agents terminated={dead}")

                blue_sum = sum(ep_rewards[aid] for aid in env.blue_ids)
                red_sum = sum(ep_rewards[aid] for aid in env.red_ids)
                print(f"  >>> Reward summary: blue_total={blue_sum:+.1f}, "
                      f"red_total={red_sum:+.1f}")
                break
        else:
            print(f"  >>> Episode {ep}: completed {MAX_STEPS} steps without done.")

        # ---- Save .acmi after episode 1 ----
        if ep == 1:
            n_frames = env.save_acmi(ACMI_OUTPUT)
            print(f"  >>> TacView file saved: {ACMI_OUTPUT}  ({n_frames} frames)")

        print("-" * 60)

    env.close()
    print("All episodes finished.")


if __name__ == "__main__":
    run()
