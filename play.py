"""
Load one or more trained agents and watch them drive the racetrack greedily.

Single agent:
    python play.py --weights weights/my_run.npz

Compare several agents side by side:
    python play.py --weights weights/a.npz weights/b.npz --labels "eps=0.1" "eps=0.6"

Saves an animated GIF (and optionally shows it live with --show).

This module is pure visualization: it only *reads* an agent's Q table via the
existing agent physics (update_velocity / calculate_trajectory) and the existing
environment reward. It does not modify the learning logic.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from agent import CarAgent
from environment import Environment
from map_creator import build_track
from actions import ACTIONS
from persistence import load_weights


def load_agent(path):
    """Build a CarAgent and load its trained Q table from disk."""
    Q, C, meta = load_weights(path)
    agent = CarAgent(name=meta.get("run_name", "agent") or "agent", velocity=[0, 0])
    agent.Q = Q
    if C is not None:
        agent.C = C
    return agent, meta


def greedy_rollout(agent, environment, start_state, max_steps=1000):
    """
    Roll the agent out under its *greedy* (argmax) policy and record the cells
    visited. Stops on finish, crash, or max_steps.

    Returns
    -------
    positions : list[tuple[int, int]]   cells occupied over time (for animation)
    status    : str                     "finish" | "crash" | "timeout"
    total_reward : int                  -1 per step taken
    """
    positions = [(start_state[0], start_state[1])]
    current_state = start_state
    total_reward = 0

    for _ in range(max_steps):
        row, col, vx, vy = current_state
        action = ACTIONS[int(np.argmax(agent.Q[row, col, vx, vy, :]))]

        new_velocity = agent.update_velocity(current_state, action)
        trajectory = agent.calculate_trajectory((row, col, vx, vy), new_velocity)
        reward = environment.calculate_reward(trajectory)
        total_reward -= 1

        if reward == -2:
            positions.extend((int(p[0]), int(p[1])) for p in trajectory)
            return positions, "crash", total_reward

        if reward == 0:
            positions.extend((int(p[0]), int(p[1])) for p in trajectory)
            return positions, "finish", total_reward

        next_row, next_col = trajectory[-1]
        current_state = (next_row, next_col, new_velocity[0], new_velocity[1])
        positions.append((int(next_row), int(next_col)))

    return positions, "timeout", total_reward


def animate(rollouts, track_maps, labels, save_gif=None, show=False, interval=120):
    """
    rollouts   : list of (positions, status, total_reward)
    track_maps : list of np.ndarray, one track per rollout (same order)
    Renders each rollout on its own track, side by side, with a marker moving
    along the recorded path.
    """
    n = len(rollouts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    markers = []
    trails = []
    max_len = max(len(r[0]) for r in rollouts)

    for ax, (positions, status, total_reward), track_map, label in zip(axes, rollouts, track_maps, labels):
        ax.imshow(track_map, cmap="GnBu", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}\n{status}  |  reward {total_reward}", fontsize=10)

        (trail,) = ax.plot([], [], "-", color="tab:orange", linewidth=1.5, alpha=0.7)
        (marker,) = ax.plot([], [], "o", color="red", markersize=10)
        trails.append(trail)
        markers.append(marker)

    def update(frame):
        artists = []
        for (positions, _, _), marker, trail in zip(rollouts, markers, trails):
            idx = min(frame, len(positions) - 1)
            rows = [p[0] for p in positions[: idx + 1]]
            cols = [p[1] for p in positions[: idx + 1]]
            trail.set_data(cols, rows)
            marker.set_data([positions[idx][1]], [positions[idx][0]])
            artists.extend([trail, marker])
        return artists

    anim = FuncAnimation(fig, update, frames=max_len, interval=interval, blit=True)
    plt.tight_layout()

    if save_gif:
        anim.save(save_gif, writer=PillowWriter(fps=max(1, int(1000 / interval))))
        print(f"Saved animation to {save_gif}")

    if show:
        plt.show()
    plt.close(fig)
    return anim


def main():
    parser = argparse.ArgumentParser(description="Replay / compare trained racetrack agents")
    parser.add_argument("--weights", nargs="+", required=True, help="one or more .npz weight files")
    parser.add_argument("--labels", nargs="+", default=None, help="labels for each agent (matches --weights order)")
    parser.add_argument("--track", type=str, default=None, choices=["a", "b"],
                        help="override track (default: read from each agent's metadata)")
    parser.add_argument("--start-index", type=int, default=0, help="which start cell to launch all agents from")
    parser.add_argument("--max-steps", type=int, default=1000, help="max greedy steps before giving up")
    parser.add_argument("--save-gif", type=str, default="comparison.gif", help="output GIF path")
    parser.add_argument("--show", action="store_true", help="display the animation window")
    parser.add_argument("--seed", type=int, default=None, help="random seed (start cell tie-breaks etc.)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    agents_meta = [load_agent(p) for p in args.weights]
    labels = args.labels or [m.get("run_name") or p for (_, m), p in zip(agents_meta, args.weights)]

    rollouts = []
    track_maps = []
    for (agent, meta), label in zip(agents_meta, labels):
        track = args.track or meta.get("track") or "a"
        track_map = build_track(track)
        environment = Environment(map=track_map)

        start_positions = np.argwhere(track_map == 0.8)
        idx = min(args.start_index, len(start_positions) - 1)
        r, c = start_positions[idx]
        start_state = (int(r), int(c), 0, 0)

        positions, status, total_reward = greedy_rollout(
            agent, environment, start_state, max_steps=args.max_steps
        )
        print(f"{label}: track={track}, {status}, reward={total_reward}, steps={len(positions)}")
        rollouts.append((positions, status, total_reward))
        track_maps.append(track_map)  # each agent is drawn on its own track

    animate(rollouts, track_maps, labels, save_gif=args.save_gif, show=args.show)


if __name__ == "__main__":
    main()
