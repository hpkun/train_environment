# MAPPO Network Architecture Audit

The current network is a MAPPO baseline. It uses a shared actor, centralized
critic, continuous Gaussian policy, and PPO update.

This is reasonable as a BRMA-MAPPO / MAPPO baseline for checking whether the
main environment can train and evaluate end to end.

It is not the final proposed method for the heterogeneous TAM-HAPPO paper. The
current baseline does not implement temporal feature extraction, entity
attention, GRU memory, or HAPPO-style sequential agent updates.

No network structure is changed by this audit. After the baseline and logging
path are stable, the method module can introduce entity attention actor/critic
components separately.
