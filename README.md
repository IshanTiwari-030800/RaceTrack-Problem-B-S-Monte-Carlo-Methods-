# Racetrack — Off-Policy Monte Carlo Control

An implementation of the **Racetrack** problem from Sutton & Barto's
*Reinforcement Learning: An Introduction* (Chapter 5), solved with **off-policy
Monte Carlo control** using weighted importance sampling.

The agent (a car) starts on the start line and must reach the finish line in as
few steps as possible. Velocity components are non-negative and bounded
(`0..4`), so the car can only accelerate up and to the right. Running off the
track resets the car to a random start cell and the episode continues until the
finish line is crossed.

## Project layout

| File | Responsibility |
|------|----------------|
| `map_creator.py` | Builds the two Sutton & Barto tracks (`a`, `b`). |
| `actions.py` | The 9 acceleration actions: `{-1,0,1} x {-1,0,1}`. |
| `environment.py` | Reward function (`-1` per step, `0` at finish, off-track = reset). |
| `agent.py` | Car physics (velocity update, trajectory) and the `Q` / `C` tables. |
| `utils.py` | `create_episode` — generates one episode under an ε-soft behaviour policy. |
| `training.py` | The MC control training loop + CLI. |
| `persistence.py` | Save / load trained weights (`.npz`). |
| `play.py` | Replay a trained agent greedily and compare agents side by side. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# Quick run on track A, no wandb account needed:
python training.py --track a --epsilon 0.1 --steps 50000 \
    --wandb-mode disabled --seed 0 --save weights/trackA_eps01.npz
```

Key flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--track` | `a` | Which map (`a` or `b`). |
| `--epsilon` | `0.6` | Behaviour-policy exploration rate. |
| `--steps` | `10000` | Number of training episodes. |
| `--seed` | `None` | Seed for reproducibility. |
| `--save` | auto | Where to write weights (`weights/<run>.npz` by default). |
| `--wandb-mode` | `online` | `online`, `offline`, or `disabled`. |
| `--name` | `None` | wandb run name. |

Training logs `episode_return` and `episode_length` to Weights & Biases (project
`racetrack-mc`). Lower episode length / higher (less negative) return = better.

## Replaying & comparing agents

After training, watch an agent drive greedily:

```bash
python play.py --weights weights/trackA_eps01.npz --show
```

Compare several agents **side by side** (each on its own copy of the track, with
its outcome and total reward in the title):

```bash
python play.py \
    --weights weights/trackA_eps01.npz weights/trackA_eps06.npz \
    --labels "epsilon=0.1" "epsilon=0.6" \
    --save-gif comparison.gif --show
```

`play.py` rolls each agent out under its **greedy** policy, records the path, and
animates a marker driving along it. The track to use is read from each saved
agent's metadata (override with `--track`).

## Reproducibility

Pass `--seed` to `training.py` for a deterministic run, and `--save` to persist
the resulting `Q`/`C` tables together with the run metadata (track, epsilon,
steps). The saved `.npz` is everything needed to replay the agent later.

By default `.gitignore` excludes `weights/*.npz` (they can be large). If you want
trained weights committed for exact reproducibility, remove that line from
`.gitignore`.
