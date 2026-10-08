import numpy as np

from src.adaptive_controller import AdaptiveEstimates


def compute_control_inputs(
    state,
    kappa,
    estimates,
    fuzzy_compensation=None,
    omega_bar=0.0,
):
    """
    Compute control inputs u1, u2, u3, u4
    based on Eq. (33) of the reference paper.

    fuzzy_compensation:
        [f1, f2, f3, f4]

    Returns:
        [u1, u2, u3, u4]
    """

    state = np.asarray(state, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    if state.shape != (8,):
        raise ValueError("State must contain exactly 8 values.")

    if kappa.shape != (4,):
        raise ValueError("Kappa must contain exactly 4 values.")

    if fuzzy_compensation is None:
        fuzzy_compensation = np.zeros(4)

    fuzzy_compensation = np.asarray(
        fuzzy_compensation,
        dtype=float,
    )

    if fuzzy_compensation.shape != (4,):
        raise ValueError(
            "Fuzzy compensation must contain exactly 4 values."
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

    f1, f2, f3, f4 = fuzzy_compensation

    # Roll control - Eq. (33)
    u2 = (
        -f2
        - zeta[0] * theta_dot * psi_dot
        - zeta[1] * phi_dot**2
        - zeta[2] * omega_bar * theta_dot
        + h[0] * kappa[0]
    )

    # Pitch control - Eq. (33)
    u3 = (
        -f3
        - zeta[3] * phi_dot * psi_dot
        - zeta[4] * theta_dot**2
        - zeta[5] * omega_bar * phi_dot
        + h[1] * kappa[1]
    )

    # Yaw control - Eq. (33)
    u4 = (
        -f4
        - zeta[6] * phi_dot * theta_dot
        - zeta[7] * psi_dot**2
        + h[2] * kappa[2]
    )

    cos_product = np.cos(phi) * np.cos(theta)

    if abs(cos_product) < 1e-8:
        raise ValueError(
            "Controller is near a singular configuration: "
            "cos(phi) * cos(theta) is too close to zero."
        )

    # Altitude / thrust control - Eq. (33)
    u1 = (
        -f1
        + (
            -zeta[8] * z_dot
            + m_hat * kappa[3]
        )
        / cos_product
    )

    return np.array([u1, u2, u3, u4])


if __name__ == "__main__":
    estimates = AdaptiveEstimates()

    initial_state = np.full(8, 0.10)

    kappa = np.array(
        [
            17.74395102,
            7.27197551,
            14.00796327,
            38.11,
        ]
    )

    # Type-3 FLS has not been connected yet.
    fuzzy_compensation = np.zeros(4)

    controls = compute_control_inputs(
        state=initial_state,
        kappa=kappa,
        estimates=estimates,
        fuzzy_compensation=fuzzy_compensation,
        omega_bar=0.0,
    )

    print("Control inputs:")
    print(controls)

    expected_controls = np.array(
        [
            25.01077517,
            1.77239510,
            1.45239510,
            4.20038898,
        ]
    )

    assert np.allclose(
        controls,
        expected_controls,
        atol=1e-7,
    )

    print("\nController test PASSED.")