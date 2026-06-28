import numpy as np

class Environment:

    def __init__(
        self,
        map: list[list[float]]
    ):

        self.map = map
    
    def calculate_reward(
        self,
        trajectory: list[tuple[int]]
    ):

        rows, cols = self.map.shape

        for point in trajectory:

            row, col = point
            if row < 0 or row >= rows or col < 0 or col >= cols:
                return -2  # off map

            if self.map[point] == 0: # Car skid out
                return -2

            if self.map[point] == 0.4: # Finishing line crossed
                return 0

        return -1