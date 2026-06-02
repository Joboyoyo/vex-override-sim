"""Smoke tests for the ToggleDuel RL env and the NN policy plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_env_reset_and_step_shapes():
    """reset() returns a 14-d float32 obs; step() returns (obs, reward, done, info)."""
    from envs import ToggleDuelEnv
    env = ToggleDuelEnv(episode_steps=20, seed=0)
    obs = env.reset()
    assert obs.shape == (env.OBS_DIM,)
    assert obs.dtype.name == "float32"
    obs, reward, done, info = env.step(0)
    assert obs.shape == (env.OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    # New env tracks aggregate ownership across all 4 toggles, not a single
    # toggle's resting_state field.
    assert "ours_count" in info
    assert "n_toggles" in info


def test_env_completes_full_episode():
    """Running the configured number of steps ends with done=True exactly once."""
    from envs import ToggleDuelEnv
    env = ToggleDuelEnv(episode_steps=30, seed=1)
    env.reset()
    dones = []
    for _ in range(30):
        _, _, done, _ = env.step(0)
        dones.append(done)
    assert dones[-1] is True
    assert sum(1 for d in dones if d) == 1


def test_policy_forward_pass_and_action_sampling():
    """MLPPolicy.act() returns a valid discrete action index for any obs."""
    import torch
    from envs import ToggleDuelEnv
    from ai.nn_policy import MLPPolicy
    env = ToggleDuelEnv(episode_steps=10, seed=2)
    obs = env.reset()
    net = MLPPolicy(obs_dim=env.OBS_DIM, action_dim=env.ACTION_DIM, hidden=16)
    a = net.act(obs)
    assert isinstance(a, int)
    assert 0 <= a < env.ACTION_DIM
    # Greedy and stochastic both return valid actions
    a2 = net.act(obs, greedy=True)
    assert 0 <= a2 < env.ACTION_DIM


def test_policy_save_load_roundtrip(tmp_path):
    """Saving and loading the policy weights preserves identical outputs."""
    import torch
    from envs import ToggleDuelEnv
    from ai.nn_policy import MLPPolicy
    env = ToggleDuelEnv(episode_steps=10, seed=3)
    obs = env.reset()
    obs_t = torch.as_tensor(obs).unsqueeze(0)

    net1 = MLPPolicy(obs_dim=env.OBS_DIM, action_dim=env.ACTION_DIM, hidden=16)
    with torch.no_grad():
        logits1, value1 = net1(obs_t)

    p = tmp_path / "policy.pt"
    net1.save(p)
    net2 = MLPPolicy.load(p, obs_dim=env.OBS_DIM, action_dim=env.ACTION_DIM,
                           hidden=16)
    with torch.no_grad():
        logits2, value2 = net2(obs_t)
    assert torch.allclose(logits1, logits2)
    assert torch.allclose(value1, value2)


def test_nn_bot_drives_robot():
    """NNBot.update() actually moves the robot through the physics pipeline
    (driven by an untrained network — verifies the integration, not skill)."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App
    from ai.nn_policy import MLPPolicy
    from ai.nn_bot import NNBot

    app = App()
    app._enter_duel_mode()
    # Replace R0's bot with an NN bot powered by an UNTRAINED net
    from envs import ToggleDuelEnv as _E
    net = MLPPolicy(obs_dim=_E.OBS_DIM, action_dim=_E.ACTION_DIM, hidden=16)
    r0_idx = next(i for i, b in enumerate(app.bots) if b.robot_id == 0)
    app.bots[r0_idx] = NNBot(robot_id=0, world=app.world, policy=net,
                              collides_at=app._make_collides_for(0))
    r0 = app.world.robots[0]
    start_x, start_y = r0.x, r0.y

    dt = 0.05
    for _ in range(60):    # 3 seconds
        for b in app.bots:
            b.update(dt)
        app._resolve_robot_overlaps()
        app._update_toggle_contact()

    # The robot's position should have changed (untrained net still produces
    # SOME action distribution which won't be perfectly stationary).
    import math
    moved = math.hypot(r0.x - start_x, r0.y - start_y)
    assert moved > 0.1, f"NN bot didn't move: started ({start_x},{start_y}), ended ({r0.x},{r0.y})"

    pygame.quit()
