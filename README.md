# Override Sim

2D simulator for the VEX V5RC 2026–2027 game **Override**. Built in stages:

1. **Scoring engine** — pure Python, no physics. (`core/`)
2. **Renderer** — pygame top-down view.
3. **Physics** — pymunk 2D rigid bodies.
4. **Driver mode** — gamepad control vs scripted opponent.
5. **RL env** — Gymnasium + PettingZoo wrapper.
6. **Training** — CleanRL PPO, eventually self-play.

## Getting started

### 1. Prerequisites

You need **Python 3.11 or newer** and `git`. Install Python from
<https://www.python.org/downloads/> if you don't have it (tick "Add Python
to PATH" during install on Windows). Check it works:

```bash
python --version    # should print 3.11+ (3.12 or 3.13 ideal)
git --version
```

### 2. Clone and install

```bash
git clone https://github.com/Joboyoyo/vex-override-sim.git
cd vex-override-sim

# (optional but recommended) make a virtual environment so deps don't
# clash with your system Python:
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# install everything we need
pip install -r requirements.txt
```

The dependencies are pygame (window/rendering), numpy + torch (neural-network
bot), pytest (tests), and PyYAML (rule data).

### 3. Verify it works

```bash
pytest tests/
```

You should see something like `64 passed`. If a test fails, that's the
fastest signal that the install isn't right — usually a missing dep.

### 4. Run the simulator

```bash
python -m render.pygame_view
```

A window opens with the field. In-game controls (also shown in the on-screen
Help panel):

| Key | What it does |
|---|---|
| `WASD` / arrows | Drive your red robot (R0) |
| `E` / R2 button | Pick up / drop / place into a goal / spawn from a loader |
| `X` / X button | Deploy / retract the front wing (24" push zone) |
| `T` / A button | Cycle the toggle behind you (must back into it) |
| `Y` | Set the toggle behind you to your alliance color |
| `B` | Enable / disable AI bots (your partner + the opponents) |
| `F` | Enter / exit the AI-vs-AI toggle duel demo |
| `N` | (In duel) swap red bot ↔ neural-network policy |
| `SPACE` | Start the 1:45 driver-period clock |
| `D` / `R` | Load 113-pt demo / reset field |
| `F11` / `ESC` | Fullscreen / quit |

Plug in an Xbox/PS-style controller if you have one — the game will detect
it on launch and you can drive with the left stick.

### 5. Try the neural-network bot

```bash
python -m render.pygame_view
# then in-game: press F (duel mode), then N (swap red bot to the NN)
```

If you want to retrain the network yourself (~10 sec via behavior cloning):

```bash
python scripts/train_bc.py --epochs 60
# overwrites ai/toggle_duel_policy.pt with the new weights
```

## Layout

```
override-sim/
├── core/
│   ├── state.py        # World, Robot, Pin, Cup, Goal, Toggle dataclasses
│   ├── rules.yaml      # All numeric rules — edit when manual updates
│   └── scoring.py      # Pure score(world) -> ScoreResult
├── tests/
│   └── test_scoring.py # Rulebook examples + strategy floor test
├── README.md
└── requirements.txt
```

## Design rules

- `core/scoring.py` is the **single source of truth** for points. It's pure,
  has no physics/render dependencies, and is imported by the HUD, the strategy
  Monte Carlo, and the RL reward function.
- All numeric rules live in `core/rules.yaml`. When the manual revises (v0.1 →
  v0.2 → …), edit the YAML, never the Python.
- Coordinates: field centered at (0,0), x and y in feet, range [-6, +6].

## Neural-network bot (toggle duel)

A minimal RL setup for the AI-vs-AI toggle duel mode:

```
envs/toggle_duel_env.py   # headless 14-d obs, 9-action discrete env
ai/nn_policy.py           # 2-layer MLP (64-wide) policy + value head
ai/nn_bot.py              # adapter so the network can drive a robot
                          #   in the live duel like a ScriptedBot does
scripts/train_toggle_duel.py
                          # REINFORCE-with-baseline trainer in pure PyTorch
```

Train and play:

```bash
# 1a) FAST: behavior-cloning from ScriptedBot rollouts (~10 seconds CPU).
#     Records (obs, action) pairs from the rule-based bot driving R0 in
#     both nav-only and opponent-present scenarios, then supervised-trains
#     the policy via cross-entropy.
python scripts/train_bc.py --epochs 60

# 1b) OR: REINFORCE-with-baseline from scratch (~4 min for 300 episodes).
#     Slower, doesn't converge as cleanly in this little budget, but pure RL
#     with no expert demos. Uses a curriculum: phase 1 vs frozen opponent
#     (learn navigation), phase 2 vs full ScriptedBot.
python scripts/train_toggle_duel.py --episodes 300 --steps 300

# Either writes:  ai/toggle_duel_policy.pt   (~5,800 params, 23 KB)

# 2) watch it play
python -m render.pygame_view
#   F  → enter duel mode
#   N  → swap the red bot from ScriptedBot to the trained NN policy
#        (press N again to revert)
```

The BC trainer is the easier path: it just imitates the existing rule-based
bot, so the resulting policy at least navigates to the toggle and tries to
engage. The REINFORCE trainer is a starting point for actual RL — extend
it to PPO + self-play for a policy that can outplay the rule-based bot
(task #9 on the roadmap).

## Strategy reference

The current working strategy (see chat history for derivation):

- **Auto:** 4 yellow pins in one quadrant + set that toggle → ~40 pts. Auto is
  protected so the toggle is safe.
- **Auto bonus:** +12 pts to the auto leader.
- **Driver:** 10 alliance match-load pins → +50 pts locked (can't be descored).
- **Endgame:** Deny opponent toggles (touch = unset = their yellows worth 0).
- **Floor:** ~102 pts plus denial swing.

The `tests/test_scoring.py::test_user_strategy_floor_matches_expected_102`
test pins this number down.
