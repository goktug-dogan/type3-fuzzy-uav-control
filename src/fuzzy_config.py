from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FuzzyConfig:
    """
    Reproduction-specific fuzzy configuration.

    IMPORTANT:
    The reference paper defines the mathematical Type-3 FLS
    structure but does not report these numerical design
    parameters.

    Therefore, these values are implementation choices
    and must be reported as such.
    """

    # Normalized MF centers:
    # Negative, Zero, Positive
    centers = (-1.0, 0.0, 1.0)

    # Symmetric spread in normalized input space
    sigma: float = 1.0

    # Two Type-3 z-slice pairs
    slice_pairs = (
        (0.50, 0.75),
        (0.75, 1.00),
    )


# Scaling values used before fuzzy inference.
#
# Each channel uses two physical states.
# Values are normalized and clipped to [-1, 1].
CHANNEL_INPUT_SCALES = {
    "altitude": np.array([3.0, 1.0]),
    "roll": np.array([np.pi / 2.0, 1.0]),
    "pitch": np.array([np.pi / 2.0, 1.0]),
    "yaw": np.array([np.pi, 1.0]),
}