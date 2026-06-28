import math
import numpy as np
import wandb
from agent import CarAgent
from environment import Environment
from tqdm.auto import tqdm
from utils import create_episode
from map_creator import build_track
from actions import ACTIONS
from persistence import save_agent

DEFAULT_TOTAL_STEPS = 10000

def train(
    driver: CarAgent,
    environment: Environment,
    epsilon: float,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    run_name: str = None,
    wandb_mode: str = "online",
    save_path: str = None,
    track: str = None,
):

    wandb.init(
        project="racetrack-mc",
        name=run_name,
        mode=wandb_mode,
        config={"total_steps": total_steps, "epsilon": epsilon, "track": track},
    )

    # Initialize the state value functions
    driver.init_action_value_functions(environment)
    start_positions = np.argwhere(environment.map == 0.8)

    for step in tqdm(
        range(total_steps),
        desc="Training agent....."
    ):

        # Generate one episode using an epsilon soft policy
        row, col = start_positions[np.random.randint(len(start_positions))]
        start_state = (int(row), int(col), 0, 0)
        rewards, states, actions = create_episode(driver.Q, epsilon, driver, environment, start_state)

        wandb.log({
            "episode_return": sum(rewards),
            "episode_length": len(states),
        }, step=step)

        G, W = 0, 1 # W will be our importance sampling rato

        # Update the state value functions visited during this episode
        for St, At, R_t_1 in zip(states[::-1], actions[::-1], rewards[::-1]): # Traverse from T-1

            row, col, vx, vy = St
            a_idx = ACTIONS.index(At) # action tuple -> integer index into action axis
            G = G + R_t_1

            # Update C and Q
            driver.C[row, col, vx, vy, a_idx] += W
            driver.Q[row, col, vx, vy, a_idx] += (W/driver.C[row, col, vx, vy, a_idx]) * (G - driver.Q[row, col, vx, vy, a_idx])

            # Update W
            action_values = driver.Q[row, col, vx, vy, :]
            optimal_action = np.argmax(action_values) # index of action under our optimal policy, pi

            if a_idx != optimal_action: # If action taken at that time not optimal, break (cause pi(St, At) = 0 for non optimal At)
                break

            W *= 1.0 / ((1 - epsilon) + (epsilon / len(ACTIONS))) # Update the value of importance sampling ratio

    wandb.finish()

    if save_path is not None:
        save_agent(
            driver,
            save_path,
            meta={
                "track": track,
                "epsilon": epsilon,
                "total_steps": total_steps,
                "run_name": run_name or "",
            },
        )
        print(f"Saved trained agent weights to {save_path}")

    return driver

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Train racetrack MC agent")
    parser.add_argument("--name", type=str, default=None, help="wandb run name")
    parser.add_argument("--epsilon", type=float, default=0.6, help="behaviour policy epsilon")
    parser.add_argument("--track", type=str, default="a", choices=["a", "b"], help="track map")
    parser.add_argument("--steps", type=int, default=DEFAULT_TOTAL_STEPS, help="number of training episodes")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--save", type=str, default=None, help="path to save trained weights (.npz)")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb logging mode",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    save_path = args.save
    if save_path is None:
        run_tag = args.name or f"track{args.track}_eps{args.epsilon}_steps{args.steps}"
        save_path = f"weights/{run_tag}.npz"

    driver = CarAgent(name="Agent_1", velocity=[0,0])
    race_map = build_track(args.track)
    environment = Environment(map=race_map)

    train(
        driver,
        environment,
        args.epsilon,
        total_steps=args.steps,
        run_name=args.name,
        wandb_mode=args.wandb_mode,
        save_path=save_path,
        track=args.track,
    )