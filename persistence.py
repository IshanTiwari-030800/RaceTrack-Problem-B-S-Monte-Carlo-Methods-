"""
Save / load trained agent weights (the Q and C tables) plus run metadata.

Weights are stored as a single compressed .npz file so a trained agent can be
reproduced and replayed at any time.
"""
import os
import numpy as np


def save_agent(agent, path, meta=None):
    """
    Persist an agent's action-value tables to ``path`` (a .npz file).

    Parameters
    ----------
    agent : CarAgent
        Must have ``Q`` and ``C`` attributes (i.e. init_action_value_functions
        has been called and training has run).
    path : str
        Destination file. A ``.npz`` extension is added if missing.
    meta : dict, optional
        Arbitrary run metadata (track, epsilon, total_steps, ...). Stored
        alongside the arrays so a replay knows which track to load.
    """
    meta = meta or {}
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    np.savez_compressed(
        path,
        Q=agent.Q,
        C=agent.C,
        meta=np.array(meta, dtype=object),
    )


def load_weights(path):
    """
    Load weights saved by :func:`save_agent`.

    Returns
    -------
    Q : np.ndarray
    C : np.ndarray or None
    meta : dict
    """
    if not path.endswith(".npz"):
        path = path + ".npz"

    data = np.load(path, allow_pickle=True)
    Q = data["Q"]
    C = data["C"] if "C" in data.files else None
    meta = data["meta"].item() if "meta" in data.files else {}
    return Q, C, meta
