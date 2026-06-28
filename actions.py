"""
Contains the set of possible actions that can be taken for any given state St

A(s) = {(-1, 0), (-1, 1), (-1, -1)......}

NOTE: Can add more velocity updates to make this set more diverse
"""
from itertools import product

ACTIONS = list(product([-1, 0, 1], repeat=2))