# Fix Reward Shift Design

## Summary

The current RL implementation records rewards at the same moment as states and actions.
In practice, this means the stored reward corresponds to the pre-action state rather than
to the transition caused by the sampled action. This creates an off-by-one alignment bug
between `features/actions/log_probs` and `rewards`.

This design fixes the issue at trajectory collection time rather than inside the loss
functions. Each trajectory index `t` will represent a single transition:

- `s_t`: state before the action
- `a_t`: action sampled from `s_t`
- `r_t`: reward observed after applying `a_t` and advancing the environment

This keeps raw recorded trajectories semantically correct and avoids downstream slicing
repairs in every RL consumer.

## Problem

In the current code path:

1. `ActorCriticAgent.calc_action()` computes the observable.
2. It samples an action.
3. It also computes and stores the reward immediately.
4. The engine later applies the action and advances the system.

As a result, `reward[t]` does not describe the outcome of `action[t]`. Instead, it is
computed from the old state. This is inconsistent with standard RL transition semantics
and makes PPO/value estimation harder to reason about.

## Chosen Approach

Use a two-phase trajectory update:

1. Before stepping:
   - compute and store `features`
   - compute and store `actions`
   - compute and store `log_probs`
2. After stepping:
   - compute and store `rewards`

This follows the direction taken in `mag_r_clone_reward_fix` and is preferred over the
older `reward_shift_stovey.txt` approach, which repairs misalignment only inside the loss
code by slicing arrays.

## Why This Approach

Advantages:

- fixes the bug at the correct abstraction layer
- keeps stored trajectory data truthful and easier to debug
- avoids duplicating shift logic across PPO, policy gradient, GAE, logging, and future
  environment adapters
- provides a clean basis for later Gym-style integration where rewards may come from an
  external environment step

Why not the slicing approach:

- it leaves raw trajectory data semantically wrong
- each loss/value function must remember the same shift convention
- non-loss consumers of trajectory data can still observe the bug
- terminal and bootstrapping edge cases become harder to audit

## Scope

This design intentionally keeps the fix narrow.

Included:

- split action collection from reward collection
- update engine/force-function flow so reward is computed after the step
- keep policy gradient and PPO consuming aligned arrays directly
- preserve the current no-extra-bootstrap terminal handling in GAE
- adapt existing tests and add regression coverage for the new trajectory semantics

Not included:

- redesigning the rollout format to store an extra final observation/value
- changing the mathematical form of GAE beyond using correctly aligned rewards
- broad trainer/checkpointer refactors from the Mag-r branch

## API and Control-Flow Changes

### Agent API

`Agent` gains a `calc_reward(colloids, external_reward=0.0)` method.

For `ActorCriticAgent`:

- `calc_action()` stores only action-side trajectory data
- `calc_reward()` computes extrinsic reward from the post-step state
- intrinsic reward, if present, is added during reward collection
- reward is appended to the trajectory only after stepping

`external_reward` remains part of the method signature because it supports Gym-like
environments where the environment step may directly return a reward. It is not required
for the Espresso path most of the time, but keeping it avoids designing the interface too
narrowly around the current engine.

### Force Function

`ForceFunction` gains a `calc_reward()` pass that forwards reward collection to each
managed agent after the engine has advanced.

### Engine

`EspressoMD.integrate()` continues to:

1. collect actions at slice boundaries
2. apply forces/torques
3. advance the integrator

It is extended to call `force_model.calc_reward(...)` after the integrator has run for
the slice increment so that the stored reward reflects the resulting next state.

## Loss and Value Function Behavior

The loss code should consume aligned rollout arrays directly.

### Policy Gradient

`features[t]`, `actions[t]`, and `rewards[t]` refer to the same transition index.
No reward shifting or array trimming is needed.

### PPO / GAE

For this fix, keep the existing terminal treatment shape:

- for non-final steps: use `r_t + gamma * V(s_{t+1}) - V(s_t)`
- for the final stored step: use `r_T - V(s_T)`

This is a deliberate minimal-change choice. It fixes the off-by-one bug without expanding
the rollout schema to include an additional final bootstrap state.

## Limitations

### Limitation of the chosen fix

The rollout still does not explicitly store `s_{T+1}` as a separate terminal bootstrap
state. That means the final transition cannot use a bootstrap value unless the rollout
format is extended in a later change.

### Limitation of the previous slicing approach

If the project stayed with loss-side shifting only, every present and future trajectory
consumer would need to understand the same indexing convention. That is fragile and easy
to regress.

## Testing Strategy

Adapting tests is part of the implementation, not optional follow-up work. Existing tests
that implicitly depend on pre-step reward collection must be updated to match the new
trajectory semantics.

Add or update tests to verify trajectory semantics explicitly.

Required coverage:

- reward is recorded after the environment step, not before it
- `len(features) == len(actions) == len(log_probs) == len(rewards)` for completed
  rollouts
- PPO and policy gradient consume aligned arrays without manual reward slicing
- a deterministic toy setup can distinguish pre-step reward collection from post-step
  reward collection
- existing RL tests that inspect stored rewards/trajectory contents are updated to assert
  the new alignment convention rather than the old one

Recommended test shape:

- unit test around agent/force-function/engine interaction with a simple deterministic
  task whose reward changes only after a movement step
- regression test that would fail under the old behavior and pass under the new one

## Implementation Notes

The `mag_r_clone_reward_fix` branch contains the right conceptual direction for the reward
alignment bug, but it also includes unrelated changes. The implementation on
`fix_reward_shift` should copy only the reward-timing fix and avoid unrelated trainer,
logging, or checkpointing churn.

## Recommendation

Implement the Mag-r style collection-time fix in `fix_reward_shift`, keep the rollout
schema otherwise unchanged, and defer final-state bootstrapping improvements to a focused
follow-up if needed.
