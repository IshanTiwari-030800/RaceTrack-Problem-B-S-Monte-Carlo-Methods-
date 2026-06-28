from environment import Environment
from typing import Tuple
import numpy as np
from actions import ACTIONS

MAX_VELOCITY = 5  

class CarAgent:

    """
    Agent that interacts and learns from the environment. 
    """

    def __init__(
        self,
        name: str,
        velocity: Tuple[int, int]
    ):

        self.name = name
        self.velocity = [0, 0]

    def next_state(
        self,
        state: Tuple[int, int, int, int],
        new_velocity: Tuple[int, int],
    ):
        row, col, vx, vy = state
        return (row - new_velocity[0], col + new_velocity[1])

    def update_velocity(
        self,
        state: Tuple[int, int, int, int],
        acceleration: Tuple[int, int]
    ):
        row, col, vx, vy = state
        updated_velocity = (vx + acceleration[0], vy + acceleration[1])

        if not all(0 <= x <= MAX_VELOCITY - 1 for x in updated_velocity):
            return (vx, vy)

        if updated_velocity == (0, 0):
            return (vx, vy)

        return updated_velocity

    def calculate_trajectory(
        self,
        state: Tuple[int, int, int, int],
        new_velocity: Tuple[int, int],
    ):

        """
        Calculates the trajectory as the line segment equation
        """

        row, col, vx, vy = state
        x0, y0 = row, col
        x1, y1 = self.next_state(state, new_velocity)

        points = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        err = dx - dy

        while True:

            points.append((x0, y0))

            if (x0, y0) == (x1, y1):
                break

            e2 = 2 * err

            if e2 > -dy:
                err -= dy
                x0 += sx

            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def init_action_value_functions(
        self, 
        environment: Environment
    ):

        rows, cols = environment.map.shape
        # Pessimistic init so unvisited actions (which keep this value) don't beat
        # explored actions in argmax. Weighted-IS overwrites a pair's Q on first visit.
        self.Q = np.random.normal(-500, 1, size=(rows, cols, MAX_VELOCITY, MAX_VELOCITY, len(ACTIONS)))
        self.C = np.zeros((rows, cols, MAX_VELOCITY, MAX_VELOCITY, len(ACTIONS)))