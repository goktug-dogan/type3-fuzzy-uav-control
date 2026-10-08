from dataclasses import dataclass, field
import numpy as np


@dataclass
class StabilizationConfig:
    """
    Implementation-specific stabilization settings.

    These settings are NOT reported in the reference paper.
    They belong to the proposed stabilized extension.
    """

    # Prevent division by a value too close to zero in:
    # 1 / (cos(phi) * cos(theta))
    min_cos_product: float = 0.20

    # Experimental control bounds.
    #
    # u1 is thrust-like and must remain non-negative.
    # Other channels are symmetric.
    control_lower: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, -10.0, -10.0, -10.0]
        )
    )

    control_upper: np.ndarray = field(
        default_factory=lambda: np.array(
            [50.0, 10.0, 10.0, 10.0]
        )
    )

    # Broad projection bounds for adaptive estimates.
    zeta_lower: float = -2.0
    zeta_upper: float = 2.0

    h_lower: float = 0.001
    h_upper: float = 2.0

    # Positive mass-estimate interval.
    m_lower: float = 0.20
    m_upper: float = 2.00


def regularize_cos_product(
    phi,
    theta,
    min_abs_value,
):
    """
    Prevent the altitude-controller denominator from
    approaching zero.

    The sign is preserved.
    """

    value = (
        np.cos(phi)
        * np.cos(theta)
    )

    if min_abs_value <= 0.0:
        raise ValueError(
            "min_abs_value must be positive."
        )

    if abs(value) >= min_abs_value:
        return value

    # Exactly zero has no sign, so use positive floor.
    if value == 0.0:
        return min_abs_value

    return (
        np.sign(value)
        * min_abs_value
    )


def compute_stabilized_control_inputs(
    state,
    kappa,
    estimates,
    fuzzy_compensation=None,
    omega_bar=0.0,
    config=None,
):
    """
    Stabilized version of Eq. (33).

    The mathematical controller structure remains based
    on the paper, but denominator regularization and
    input bounds are implementation-specific additions.
    """

    if config is None:
        config = StabilizationConfig()

    state = np.asarray(
        state,
        dtype=float,
    )

    kappa = np.asarray(
        kappa,
        dtype=float,
    )

    if state.shape != (8,):
        raise ValueError(
            "State must contain exactly 8 values."
        )

    if kappa.shape != (4,):
        raise ValueError(
            "Kappa must contain exactly 4 values."
        )

    if fuzzy_compensation is None:
        fuzzy_compensation = np.zeros(4)

    fuzzy_compensation = np.asarray(
        fuzzy_compensation,
        dtype=float,
    )

    if fuzzy_compensation.shape != (4,):
        raise ValueError(
            "Fuzzy compensation must contain 4 values."
        )

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

    zeta = estimates.zeta_hat
    h = estimates.h_hat
    m_hat = estimates.m_hat

    f1, f2, f3, f4 = (
        fuzzy_compensation
    )

    # Roll
    u2 = (
        -f2
        - zeta[0]
        * theta_dot
        * psi_dot
        - zeta[1]
        * phi_dot**2
        - zeta[2]
        * omega_bar
        * theta_dot
        + h[0]
        * kappa[0]
    )

    # Pitch
    u3 = (
        -f3
        - zeta[3]
        * phi_dot
        * psi_dot
        - zeta[4]
        * theta_dot**2
        - zeta[5]
        * omega_bar
        * phi_dot
        + h[1]
        * kappa[1]
    )

    # Yaw
    u4 = (
        -f4
        - zeta[6]
        * phi_dot
        * theta_dot
        - zeta[7]
        * psi_dot**2
        + h[2]
        * kappa[2]
    )

    safe_cos_product = (
        regularize_cos_product(
            phi=phi,
            theta=theta,
            min_abs_value=(
                config.min_cos_product
            ),
        )
    )

    # Altitude
    u1 = (
        -f1
        + (
            -zeta[8] * z_dot
            + m_hat * kappa[3]
        )
        / safe_cos_product
    )

    raw_control = np.array(
        [
            u1,
            u2,
            u3,
            u4,
        ],
        dtype=float,
    )

    bounded_control = np.clip(
        raw_control,
        config.control_lower,
        config.control_upper,
    )

    return (
        bounded_control,
        raw_control,
        safe_cos_product,
    )


def project_adaptive_estimates(
    estimates,
    config=None,
):
    """
    Projection operator for adaptive estimates.

    This does not change the adaptation laws themselves.
    It prevents estimates from leaving predefined
    admissible intervals.
    """

    if config is None:
        config = StabilizationConfig()

    estimates.zeta_hat = np.clip(
        estimates.zeta_hat,
        config.zeta_lower,
        config.zeta_upper,
    )

    estimates.h_hat = np.clip(
        estimates.h_hat,
        config.h_lower,
        config.h_upper,
    )

    estimates.m_hat = float(
        np.clip(
            estimates.m_hat,
            config.m_lower,
            config.m_upper,
        )
    )


if __name__ == "__main__":

    from src.adaptive_controller import (
        AdaptiveEstimates,
    )

    estimates = AdaptiveEstimates()

    state = np.full(
        8,
        0.10,
    )

    kappa = np.array(
        [
            17.74395102,
            7.27197551,
            14.00796327,
            38.11,
        ]
    )

    u, raw_u, safe_cos = (
        compute_stabilized_control_inputs(
            state=state,
            kappa=kappa,
            estimates=estimates,
            fuzzy_compensation=np.zeros(4),
        )
    )

    print("Raw controls:")
    print(raw_u)

    print("\nBounded controls:")
    print(u)

    print("\nSafe cosine product:")
    print(safe_cos)

    # At the initial condition, stabilization
    # should not modify the controller output.
    assert np.allclose(
        u,
        raw_u,
        atol=1e-10,
    )

    # Explicit singularity test
    singular_state = state.copy()
    singular_state[0] = (
        np.pi / 2.0
    )

    _, _, safe_singular_cos = (
        compute_stabilized_control_inputs(
            state=singular_state,
            kappa=kappa,
            estimates=estimates,
            fuzzy_compensation=np.zeros(4),
        )
    )

    print(
        "\nRegularized cosine product "
        "near singularity:"
    )
    print(safe_singular_cos)

    assert np.isclose(
        abs(safe_singular_cos),
        0.20,
        atol=1e-12,
    )

    print(
        "\nStabilized controller "
        "test PASSED."
    )