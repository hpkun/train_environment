# TAM Recurrent Categorical HAPPO Design

## Scope

Advance the formal `multidiscrete_categorical` TAM path from one-step hidden replay to time-ordered recurrent HAPPO. Preserve the environment, rewards, missile dynamics, observations, initial states, blue targeting, and categorical action contract.

## Policy initialization

`TAMCategoricalRecurrentHAPPOPolicy` keeps learned categorical sampling. Its MAV and UAV output-layer biases receive Gaussian priors centered at `[39, 20, 20, 20]`, with configurable standard deviation in bins. Output weights retain small random initialization. Metadata records the prior and centers.

## Sequence data contract

The rollout buffer records `episode_start_masks`, `env_step_index`, `agent_alive_masks`, and each environment's initial recurrent hidden state. Formal recurrent training groups transitions by environment and preserves insertion/time order. Before each time step, hidden state is multiplied by the inverse episode-start mask; inactive agents remain masked from losses.

## Policy sequence API

The policy exposes a sequence evaluator that accepts `[T,N,obs]`, role IDs, actions, initial hidden state, episode starts, and active masks. It unfolds the encoder and GRU over time and returns categorical log-probabilities, entropy, logits/probabilities, and final hidden state. Existing one-step APIs remain for rollout and legacy callers.

## Trainer

`TAMCategoricalHAPPOTrainer` owns the formal categorical path. It computes grouped GAE, updates the attention critic, then updates MAV followed by UAV. The current role advantage is multiplied by a detached correction factor. After each role update the factor is multiplied by `exp(new_log_prob - old_log_prob)` for that role's active samples. Shared encoder/GRU parameters are updated in the explicit role order.

## Validation

Unit tests cover neutral priors, sequence advancement/reset, inactive masking, categorical PPO and HAPPO correction. A no-blue-missile validation script compares stochastic and deterministic initialized policies for 10x300-step episodes and writes JSON/Markdown evidence. Runtime gates are the specified 2k and 50k CUDA runs.
