from dataclasses import dataclass, field
import numpy as np

from src.reference_trajectory import ConstantReference
from src.tracking import (
    ALPHA,
    K,
    tracking_errors,
    sliding_surfaces,
)


@dataclass
class AdaptiveParameters:
    """
    Controller and adaptation parameters reported
    in the simulation section of the reference paper.
    """

    alpha_surface: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 2.0, 2.0, 2.0])
    )

    k_surface: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 5.0, 5.0])
    )

    alpha_adapt: np.ndarray = field(
        default_factory=lambda: np.full(4, 0.05)
    )

    beta_adapt: np.ndarray = field(
        default_factory=lambda: np.full(4, 0.05)
    )

    vartheta: float = 3.0 / 7.0


@dataclass
class AdaptiveEstimates:
    """
    Initial adaptive estimates used in the paper.

    zeta_hat:
        estimates zeta_1 ... zeta_9

    h_hat:
        estimates h_1 ... h_3

    m_hat:
        estimate of UAV mass
    """

    zeta_hat: np.ndarray = field(
        default_factory=lambda: np.full(9, 0.10)
    )

    h_hat: np.ndarray = field(
        default_factory=lambda: np.array([0.10, 0.20, 0.30])
    )

    m_hat: float = 0.65


def compute_kappa(
    e_velocity,
    s,
    reference_acceleration,
    g=9.81,
    params=None,
):
    """
    Compute kappa_1 ... kappa_4 according to Eq. (35).
    """

    if params is None:
        params = AdaptiveParameters()

    e_velocity = np.asarray(e_velocity, dtype=float)
    s = np.asarray(s, dtype=float)
    reference_acceleration = np.asarray(
        reference_acceleration,
        dtype=float,
    )

    if (
        e_velocity.shape != (4,)
        or s.shape != (4,)
        or reference_acceleration.shape != (4,)
    ):
        raise ValueError("All vectors must contain exactly 4 values.")

    kappa = (
        -params.alpha_surface * e_velocity
        -params.k_surface * s
        + reference_acceleration
    )

    # Eq. (35): kappa_4 also contains gravity
    kappa[3] += g

    return kappa


if __name__ == "__main__":
    reference = ConstantReference()
    params = AdaptiveParameters()
    estimates = AdaptiveEstimates()

    initial_state = np.full(8, 0.10)
    integral_s = np.zeros(4)

    e_position, e_velocity = tracking_errors(
        initial_state,
        reference,
    )

    s = sliding_surfaces(
        e_position,
        e_velocity,
        integral_s,
    )

    kappa = compute_kappa(
        e_velocity,
        s,
        reference.acceleration(),
        g=9.81,
        params=params,
    )

    print("Sliding surfaces:")
    print(s)

    print("\nKappa values:")
    print(kappa)

    print("\nInitial zeta estimates:")
    print(estimates.zeta_hat)

    print("\nInitial h estimates:")
    print(estimates.h_hat)

    print("\nInitial mass estimate:")
    print(estimates.m_hat)

    expected_kappa = np.array(
        [
            17.743951,
            7.2719755,
            14.00796325,
            38.11,
        ]
    )

    assert np.allclose(
        kappa,
        expected_kappa,
        atol=1e-7,
    )

    print("\nAdaptive controller initialization test PASSED.")