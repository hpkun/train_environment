# Blue Rule Audit
- `OpponentPolicy(brma_rule)` delegates to parent `rule_based_agent.blue_coordinated_actions`.
- Local wrapper passes blue observations, num_blue/num_red, engaged targets, own positions and own headings.
- The local wrapper does not pass red roles directly, but observations may encode geometry that makes red_0 a target.
- Blue has GCAS in the environment while red does not; blue trajectory stability can therefore be higher.
- Whether blue is too strong must be judged from scripted rollout first-launch/hit and blocked-reason summaries, not assumed statically.