import numpy as np
import matplotlib.pyplot as plt
import argparse

# MACROS
START = 0.8
FINISH = 0.4

def _build_track_a():

    track = np.ones((32, 17), dtype=float)

    track[14:, 0] = 0
    track[22:, 1] = 0
    track[-3:, 2] = 0

    track[:4, 0] = 0
    track[:3, 1] = 0
    track[0, 2] = 0

    track[6:, -8:] = 0
    track[6, 9] = 1

    track[:6, -1] = FINISH
    track[-1, 3:9] = START

    return track

def _build_track_b():

    track = np.ones((30, 32), dtype=float)

    for i in range(14):
        track[:(-3 - i), i] = 0

    track[3:7, 11] = 1
    track[2:8, 12] = 1
    track[1:9, 13] = 1

    track[0, 14:16] = 0

    track[-17:, -9:] = 0
    track[12, -8:] = 0
    track[11, -6:] = 0
    track[10, -5:] = 0
    track[9, -2:] = 0

    track[-1] = np.where(track[-1] == 0, 0, START)
    track[:, -1] = np.where(track[:, -1] == 0, 0, FINISH)

    return track

def build_track(track_type="a"):

    """
    Build one of Sutton & Barto's Racetrack maps.

    Parameters
    ----------
    track_type : str
        Either "a" or "b".

    Returns
    -------
    np.ndarray
        2D racetrack map.

        Values:
            0.0 -> Wall
            1.0 -> Drivable track
            0.8 -> Starting cells
            0.4 -> Finish cells
    """

    track_type = track_type.lower()

    if track_type == "a":
        return _build_track_a()

    if track_type == "b":
        return _build_track_b()

    raise ValueError(f"Unknown track type '{track_type}'. Use 'a' or 'b'.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Racetrack map builder")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Visualize the available racetracks."
    )

    args = parser.parse_args()

    if args.demo:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for ax, track_name in zip(axes, ["a", "b"]):

            track = build_track(track_name)

            ax.imshow(track, cmap="GnBu", origin="upper")
            ax.set_title(f"Track {track_name.upper()}")
            ax.set_xticks([])
            ax.set_yticks([])

            print(f"Track {track_name.upper()}")
            print(f"Shape : {track.shape}")
            print(f"Start cells : {(track == 0.8).sum()}")
            print(f"Finish cells: {(track == 0.4).sum()}")
            print(f"Track cells : {(track == 1).sum()}")
            print(f"Wall cells  : {(track == 0).sum()}")
            print("-" * 40)

        plt.tight_layout()
        plt.show()