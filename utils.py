from actions import ACTIONS
import numpy as np

def create_episode(
    action_value_function,
    epsilon,
    agent, 
    environment,
    start_state,
):

    rewards, states, actions = [], [], []
    current_state = start_state

    while True:

        row, col, vx, vy = current_state
        action_values = agent.Q[row, col, vx, vy, :]

        if np.random.random() < epsilon:
            action = ACTIONS[np.random.randint(len(ACTIONS))]
        else:
            action = ACTIONS[np.argmax(action_values)]

        new_velocity = agent.update_velocity(current_state, action)
        trajectory = agent.calculate_trajectory((row, col, vx, vy), new_velocity)
        reward = environment.calculate_reward(trajectory)

        states.append(current_state)
        actions.append(action)

        if reward == -2:
            rewards.append(-1)
            
            # Restart the run, the episode must run till it reaches terminal state, not crashes
            starts = np.argwhere(environment.map == 0.8)
            r, c = starts[np.random.randint(len(starts))]
            current_state = (int(r), int(c), 0, 0)
            states.append(current_state)
            continue

        elif reward == 0:
            rewards.append(reward)
            return rewards, states, actions

        rewards.append(-1)  # per-step reward for a normal move

        next_row, next_col = trajectory[-1]
        current_state = (next_row, next_col, new_velocity[0], new_velocity[1])