# Fix Reward Shift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align RL trajectory rewards with the action that produced them by collecting rewards after the engine step, and update tests to enforce the new transition semantics.

**Architecture:** The rollout path will be split into two phases. `ActorCriticAgent.calc_action()` will record only pre-step policy data, while a new `calc_reward()` pass on agents and `ForceFunction` will append post-step rewards after `EspressoMD.integrate()` advances the simulation. Existing losses continue to consume aligned arrays directly, and tests are updated to prove the new ordering.

**Tech Stack:** Python, JAX, pytest/unittest-style tests, EspressoMD integration loop

---

## File Map

- Modify: `swarmrl/agents/agent.py`
  - Add the base `calc_reward()` contract for all agents.
- Modify: `swarmrl/agents/actor_critic.py`
  - Remove reward collection from `calc_action()`.
  - Add `calc_reward()` that computes post-step rewards and appends them to the trajectory.
- Modify: `swarmrl/force_functions/force_fn.py`
  - Add `calc_reward()` to forward reward collection to managed agents.
- Modify: `swarmrl/engine/espresso.py`
  - Call `force_model.calc_reward(...)` after the integrator advances each slice.
- Modify: `CI/unit_tests/force_function/test_force_fn.py`
  - Update the old reward assertion and add a regression test for reward-after-step semantics.
- Modify: `CI/unit_tests/losses/test_proximal_policy_loss.py`
  - Keep PPO expectations aligned with one reward per action index.
- Modify: `CI/unit_tests/value_functions/test_gae.py`
  - Preserve and document the current no-extra-bootstrap final-step behavior.

### Task 1: Add a failing regression test for post-step reward collection

**Files:**
- Modify: `CI/unit_tests/force_function/test_force_fn.py`
- Reference: `swarmrl/force_functions/force_fn.py`
- Reference: `swarmrl/agents/actor_critic.py`

- [ ] **Step 1: Write the failing test**

Add a task whose reward depends on the colloid position so the test can distinguish pre-step from post-step evaluation. Insert the helper near the existing `DummyTask` classes and add a new test method under `TestForceFunction`.

```python
class PositionRewardTask(srl.tasks.Task):
    def __call__(self, data):
        return [item.pos[0] for item in data if item.type == 0]


def test_reward_is_not_recorded_during_calc_action(self):
    for agent in self.interaction.agents.values():
        agent.reset_trajectory()

    colloid = Colloid(
        np.array([3.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        0,
        np.array([0.0, 0.0, 0.0]),
        0,
    )
    observable = srl.observables.PositionObservable(
        box_length=np.array([1000, 1000, 1000])
    )
    network = FlaxModel(
        flax_model=FlaxNet(),
        input_shape=(3,),
        optimizer=optax.sgd(0.001),
        rng_key=6862168,
        exploration_policy=srl.exploration_policies.RandomExploration(
            probability=0.0
        ),
        sampling_strategy=CategoricalDistribution(),
    )
    agent = ActorCriticAgent(
        particle_type=0,
        network=network,
        actions=self.action_space,
        task=PositionRewardTask(),
        observable=observable,
    )
    interaction = ForceFunction(agents={"0": agent})

    interaction.calc_action([colloid])

    assert len(agent.trajectory.features) == 1
    assert len(agent.trajectory.actions) == 1
    assert len(agent.trajectory.log_probs) == 1
    assert agent.trajectory.rewards == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest CI/unit_tests/force_function/test_force_fn.py::TestForceFunction::test_reward_is_not_recorded_during_calc_action -v`

Expected: FAIL because `agent.trajectory.rewards` currently contains one entry appended from `ActorCriticAgent.calc_action()`.

- [ ] **Step 3: Add the second failing assertion for the new reward API**

Extend the same test so that, after the explicit reward pass, the trajectory contains exactly one reward based on the current state.

```python
    interaction.calc_reward([colloid])

    assert len(agent.trajectory.rewards) == 1
    assert agent.trajectory.rewards[0][0] == 3.0
```

- [ ] **Step 4: Run test to verify it still fails for the right reason**

Run: `pytest CI/unit_tests/force_function/test_force_fn.py::TestForceFunction::test_reward_is_not_recorded_during_calc_action -v`

Expected: FAIL with an attribute or behavior error because `ForceFunction.calc_reward()` does not exist yet.

- [ ] **Step 5: Commit**

```bash
git add CI/unit_tests/force_function/test_force_fn.py
git commit -m "test: add reward timing regression coverage"
```

### Task 2: Implement the reward-after-step data flow

**Files:**
- Modify: `swarmrl/agents/agent.py`
- Modify: `swarmrl/agents/actor_critic.py`
- Modify: `swarmrl/force_functions/force_fn.py`
- Modify: `swarmrl/engine/espresso.py`
- Test: `CI/unit_tests/force_function/test_force_fn.py`

- [ ] **Step 1: Add the base agent reward hook**

Add the new abstract method to `Agent`.

```python
    def calc_reward(
        self, colloids: typing.List[Colloid], external_reward: float = 0.0
    ) -> None:
        """
        Compute the reward for the agent based on the current state.
        """
        raise NotImplementedError("Implemented in Child class.")
```

- [ ] **Step 2: Run the new force-function regression test**

Run: `pytest CI/unit_tests/force_function/test_force_fn.py::TestForceFunction::test_reward_is_not_recorded_during_calc_action -v`

Expected: FAIL because `ForceFunction` still has no `calc_reward()` implementation.

- [ ] **Step 3: Move reward collection out of `ActorCriticAgent.calc_action()`**

Update `swarmrl/agents/actor_critic.py` so `calc_action()` only stores action-side data, and add `calc_reward()`.

```python
    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        state_description = self.observable.compute_observable(colloids)
        action_indices, log_probs = self.network.compute_action(
            observables=state_description
        )
        chosen_actions = np.take(list(self.actions.values()), action_indices, axis=-1)

        if self.train:
            self.trajectory.features.append(state_description)
            self.trajectory.actions.append(action_indices)
            self.trajectory.log_probs.append(log_probs)
            self.trajectory.killed = self.task.kill_switch

        return chosen_actions

    def calc_reward(
        self, colloids: typing.List[Colloid], external_reward: float = 0.0
    ) -> typing.List[float]:
        rewards = self.task(colloids)

        if self.intrinsic_reward:
            rewards += self.intrinsic_reward.compute_reward(
                episode_data=self.trajectory
            )

        rewards += external_reward
        if self.train:
            self.trajectory.rewards.append(rewards)
        self.kill_switch = self.task.kill_switch
        return rewards
```

- [ ] **Step 4: Add reward forwarding to `ForceFunction`**

Add the new method in `swarmrl/force_functions/force_fn.py`.

```python
    def calc_reward(
        self, colloids: typing.List[Colloid], external_reward: float = 0.0
    ) -> None:
        for agent in self.agents:
            self.agents[agent].calc_reward(
                colloids=colloids, external_reward=external_reward
            )
```

- [ ] **Step 5: Call reward collection after the engine step**

In `swarmrl/engine/espresso.py`, extend the integration loop immediately after `self.system.integrator.run(...)`.

```python
            self.system.integrator.run(
                steps_to_next, reuse_forces=True, recalc_forces=False
            )
            if force_model is not None:
                force_model.calc_reward(self.colloids)
            self.step_idx += steps_to_next
```

- [ ] **Step 6: Run the targeted regression test**

Run: `pytest CI/unit_tests/force_function/test_force_fn.py::TestForceFunction::test_reward_is_not_recorded_during_calc_action -v`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add swarmrl/agents/agent.py swarmrl/agents/actor_critic.py swarmrl/force_functions/force_fn.py swarmrl/engine/espresso.py CI/unit_tests/force_function/test_force_fn.py
git commit -m "fix: collect rl rewards after engine steps"
```

### Task 3: Update existing force-function expectations to the new semantics

**Files:**
- Modify: `CI/unit_tests/force_function/test_force_fn.py`
- Reference: `swarmrl/force_functions/force_fn.py`

- [ ] **Step 1: Update the existing reward assertion test**

Change `test_species_and_order_handling` so it explicitly performs the reward pass before inspecting recorded rewards.

```python
        actions = self.multi_interaction.calc_action(
            [colloid_1, colloid_2, colloid_3],
        )
        self.multi_interaction.calc_reward([colloid_1, colloid_2, colloid_3])

        loaded_data_0 = self.multi_interaction.agents["0"].trajectory
        loaded_data_2 = self.multi_interaction.agents["2"].trajectory

        loaded_data_0 = loaded_data_0.rewards[0][0]
        loaded_data_2 = loaded_data_2.rewards[0][0]
        assert loaded_data_2 == 5.0
        assert loaded_data_0 == 1.0
```

- [ ] **Step 2: Add an alignment assertion to the same test**

```python
        assert len(self.multi_interaction.agents["0"].trajectory.features) == 1
        assert len(self.multi_interaction.agents["0"].trajectory.actions) == 1
        assert len(self.multi_interaction.agents["0"].trajectory.log_probs) == 1
        assert len(self.multi_interaction.agents["0"].trajectory.rewards) == 1
```

- [ ] **Step 3: Run the force-function test module**

Run: `pytest CI/unit_tests/force_function/test_force_fn.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add CI/unit_tests/force_function/test_force_fn.py
git commit -m "test: align force function reward expectations"
```

### Task 4: Verify loss and value-function behavior still matches the rollout contract

**Files:**
- Modify: `CI/unit_tests/losses/test_proximal_policy_loss.py`
- Modify: `CI/unit_tests/value_functions/test_gae.py`
- Reference: `swarmrl/losses/proximal_policy_loss.py`
- Reference: `swarmrl/value_functions/generalized_advantage_estimate.py`

- [ ] **Step 1: Add a comment-level regression in the PPO loss test**

Document that the test uses one reward per action index and does not depend on any reward shifting. Insert this comment above the reward fixtures in `test_compute_actor_loss`.

```python
        # One reward is consumed per action/state index; no reward shifting is applied.
        rewards = np.ones((n_time_steps, n_particles))
```

- [ ] **Step 2: Strengthen the GAE test to lock in current terminal handling**

Add an assertion comment and keep the expected final-step behavior explicit.

```python
        # The final step is intentionally unbootstrapped in the current rollout format.
        expected_advantages = np.array([4, 2, 0, -2, -4])
```

- [ ] **Step 3: Run the focused loss/value tests**

Run: `pytest CI/unit_tests/losses/test_proximal_policy_loss.py CI/unit_tests/value_functions/test_gae.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add CI/unit_tests/losses/test_proximal_policy_loss.py CI/unit_tests/value_functions/test_gae.py
git commit -m "test: document aligned reward assumptions"
```

### Task 5: Run the end-to-end verification sweep

**Files:**
- No code changes
- Verify: `CI/unit_tests/force_function/test_force_fn.py`
- Verify: `CI/unit_tests/losses/test_proximal_policy_loss.py`
- Verify: `CI/unit_tests/value_functions/test_gae.py`

- [ ] **Step 1: Run the targeted regression suite**

Run:

```bash
pytest CI/unit_tests/force_function/test_force_fn.py CI/unit_tests/losses/test_proximal_policy_loss.py CI/unit_tests/value_functions/test_gae.py -v
```

Expected: PASS for all selected tests.

- [ ] **Step 2: Run one trainer-facing integration test if dependencies allow**

Run:

```bash
pytest CI/espresso_tests/integration_tests/test_rl_trainers.py -k episodic -v
```

Expected: PASS, or a clear environment/dependency failure unrelated to reward alignment.

- [ ] **Step 3: Record any blocked verification inline**

If Espresso integration tests cannot run locally, note the exact failing command and reason in the implementation summary before final handoff.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: verify reward shift rollout fix"
```
