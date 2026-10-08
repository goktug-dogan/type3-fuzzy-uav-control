import numpy as np

from src.reference_trajectory import ConstantReference


# Controller design parameters reported in the paper
ALPHA = np.array([2.0, 2.0, 2.0, 2.0])
K = np.array([10.0, 10.0, 5.0, 5.0])


def outputs_from_state(state):
    """
    Extract controlled outputs from the 8-state UAV model.

    state =
    [phi, phi_dot,
     theta, theta_dot,
     psi, psi_dot,
     z, z_dot]

    outputs =
    [phi, theta, psi, z]
    """

    state = np.asarray(state, dtype=float)

    if state.shape != (8,):
        raise ValueError("State must contain exactly 8 values.")

    return state[[0, 2, 4, 6]]


def output_velocities_from_state(state):
    """
    Extract output velocities:

    [phi_dot, theta_dot, psi_dot, z_dot]
    """

    state = np.asarray(state, dtype=float)

    if state.shape != (8,):
        raise ValueError("State must contain exactly 8 values.")

    return state[[1, 3, 5, 7]]


def tracking_errors(state, reference, t=0.0):
    """
    Compute tracking errors according to Eq. (30).

    e_position = y - y_d
    e_velocity = y_dot - y_dot_d
    """

    y = outputs_from_state(state)
    y_dot = output_velocities_from_state(state)

    y_d = reference.position(t)
    y_dot_d = reference.velocity(t)

    e_position = y - y_d
    e_velocity = y_dot - y_dot_d

    return e_position, e_velocity


def sliding_surfaces(
    e_position,
    e_velocity,
    integral_s,
    alpha=ALPHA,
    k=K,
):
    """
    Compute sliding surfaces based on Eq. (31).

    s = e_dot + alpha * e + k * integral(s dt)

    integral_s will later become an additional
    dynamic state in the numerical simulation.
    """

    e_position = np.asarray(e_position, dtype=float)
    e_velocity = np.asarray(e_velocity, dtype=float)
    integral_s = np.asarray(integral_s, dtype=float)

    if (
        e_position.shape != (4,)
        or e_velocity.shape != (4,)
        or integral_s.shape != (4,)
    ):
        raise ValueError("All inputs must contain exactly 4 values.")

    return e_velocity + alpha * e_position + k * integral_s


if __name__ == "__main__":
    reference = ConstantReference()

    # Same UAV initial condition used in the paper
    initial_state = np.full(8, 0.10)

    # At t = 0, the sliding-surface integrals start from zero
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

    print("Reference outputs:")
    print(reference.position())

    print("\nPosition errors:")
    print(e_position)

    print("\nVelocity errors:")
    print(e_velocity)

    print("\nInitial sliding surfaces:")
    print(s)

    expected_s = np.array(
        [
            -1.79439510,
            -0.74719755,
            -2.84159265,
            -5.70000000,
        ]
    )

    assert np.allclose(s, expected_s, atol=1e-7)

    print("\nTracking and sliding-surface test PASSED.")