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


def signed_power(value, exponent):
    """
    Real-valued signed power.

    Needed because sliding surfaces can be negative while
    the paper uses fractional powers such as s^(3/7).
    """
    value = np.asarray(value, dtype=float)

    return np.sign(value) * np.abs(value) ** exponent


def adaptation_derivatives(
    state,
    s,
    kappa,
    omega_bar=0.0,
    params=None,
):
    """
    Compute adaptation-law derivatives based on Eqs. (36)-(39).

    Returns:
        zeta_hat_dot : derivatives of zeta_1 ... zeta_9
        h_hat_dot    : derivatives of h_1 ... h_3
        m_hat_dot    : derivative of estimated mass
    """

    if params is None:
        params = AdaptiveParameters()

    state = np.asarray(state, dtype=float)
    s = np.asarray(s, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    if state.shape != (8,):
        raise ValueError("State must contain exactly 8 values.")

    if s.shape != (4,) or kappa.shape != (4,):
        raise ValueError("s and kappa must contain exactly 4 values.")

    (
        phi,
        phi_dot,
        theta,
        theta_dot,
        psi,
        psi_dot,
        z,
        z_dot,
    ) = state

    s_fractional = signed_power(
        s,
        params.vartheta,
    )

    # The signs of the control effectiveness terms.
    #
    # ds1/du2 -> b1 > 0
    # ds2/du3 -> b2 > 0
    # ds3/du4 -> b3 > 0
    #
    # For altitude:
    # ds4/du1 -> cos(phi)*cos(theta)/m
    sign_roll = 1.0
    sign_pitch = 1.0
    sign_yaw = 1.0

    altitude_effect = np.cos(phi) * np.cos(theta)
    sign_altitude = np.sign(altitude_effect)

    zeta_dot = np.zeros(9)

    # Eq. (36) - roll channel
    zeta_dot[0] = (
        params.alpha_adapt[0]
        * sign_roll
        * theta_dot
        * psi_dot
        * s_fractional[0]
    )

    zeta_dot[1] = (
        params.alpha_adapt[0]
        * sign_roll
        * phi_dot**2
        * s_fractional[0]
    )

    zeta_dot[2] = (
        params.alpha_adapt[0]
        * sign_roll
        * omega_bar
        * theta_dot
        * s_fractional[0]
    )

    # Eq. (37) - pitch channel
    zeta_dot[3] = (
        params.alpha_adapt[1]
        * sign_pitch
        * phi_dot
        * psi_dot
        * s_fractional[1]
    )

    zeta_dot[4] = (
        params.alpha_adapt[1]
        * sign_pitch
        * theta_dot**2
        * s_fractional[1]
    )

    zeta_dot[5] = (
        params.alpha_adapt[1]
        * sign_pitch
        * omega_bar
        * phi_dot
        * s_fractional[1]
    )

    # Eq. (38) - yaw channel
    zeta_dot[6] = (
        params.alpha_adapt[2]
        * sign_yaw
        * phi_dot
        * theta_dot
        * s_fractional[2]
    )

    zeta_dot[7] = (
        params.alpha_adapt[2]
        * sign_yaw
        * psi_dot**2
        * s_fractional[2]
    )

    # Eq. (39) - altitude channel
    zeta_dot[8] = (
        params.alpha_adapt[3]
        * sign_altitude
        * z_dot
        * np.cos(phi)
        * np.cos(theta)
        * s_fractional[3]
    )

    h_dot = np.zeros(3)

    h_dot[0] = (
        -params.beta_adapt[0]
        * sign_roll
        * kappa[0]
        * s[0]
    )

    h_dot[1] = (
        -params.beta_adapt[1]
        * sign_pitch
        * kappa[1]
        * s[1]
    )

    h_dot[2] = (
        -params.beta_adapt[2]
        * sign_yaw
        * kappa[2]
        * s[2]
    )

    m_hat_dot = (
        -params.beta_adapt[3]
        * sign_altitude
        * kappa[3]
        * np.cos(phi)
        * np.cos(theta)
        * s[3]
    )

    return zeta_dot, h_dot, m_hat_dot


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

    zeta_dot, h_dot, m_hat_dot = adaptation_derivatives(
        initial_state,
        s,
        kappa,
        omega_bar=0.0,
        params=params,
    )

    print("\nZeta adaptation derivatives:")
    print(zeta_dot)

    print("\nH adaptation derivatives:")
    print(h_dot)

    print("\nMass adaptation derivative:")
    print(m_hat_dot)

    expected_zeta_dot = np.array(
        [
            -0.00064238,
            -0.00064238,
            0.0,
            -0.00044129,
            -0.00044129,
            0.0,
            -0.00078226,
            -0.00078226,
            -0.01043677,
        ]
    )

    expected_h_dot = np.array(
        [
            1.59198294,
            0.27168011,
            1.99024627,
        ]
    )

    expected_m_hat_dot = 10.75309806

    assert np.allclose(
        zeta_dot,
        expected_zeta_dot,
        atol=1e-7,
    )

    assert np.allclose(
        h_dot,
        expected_h_dot,
        atol=1e-7,
    )

    assert np.isclose(
        m_hat_dot,
        expected_m_hat_dot,
        atol=1e-7,
    )

    print("\nAdaptation-law test PASSED.")