import numpy as np

from src.tracking import (
    ALPHA,
    K,
)


def proposed_sliding_surfaces(
    e_position,
    e_velocity,
    integral_error,
    alpha=ALPHA,
    k=K,
):
    """
    Proposed tracking-aware integral sliding surface.

    Baseline paper:
        s = e_dot + alpha*e + k*integral(s)

    Proposed extension:
        s = e_dot + alpha*e + k*integral(e)

    This modification is NOT part of the reference paper.
    """

    e_position = np.asarray(
        e_position,
        dtype=float,
    )

    e_velocity = np.asarray(
        e_velocity,
        dtype=float,
    )

    integral_error = np.asarray(
        integral_error,
        dtype=float,
    )

    if (
        e_position.shape != (4,)
        or e_velocity.shape != (4,)
        or integral_error.shape != (4,)
    ):
        raise ValueError(
            "Tracking vectors must contain 4 values."
        )

    return (
        e_velocity
        + alpha * e_position
        + k * integral_error
    )


def compute_proposed_kappa(
    e_position,
    e_velocity,
    reference_acceleration,
    g=9.81,
    alpha=ALPHA,
    k=K,
):
    """
    Kappa terms consistent with the proposed
    tracking-aware sliding surface.

    Because:

        s = e_dot + alpha*e + k*integral(e)

    then:

        s_dot =
            e_ddot
            + alpha*e_dot
            + k*e

    Therefore the stabilizing terms use -k*e
    rather than the paper's -k*s.
    """

    e_position = np.asarray(
        e_position,
        dtype=float,
    )

    e_velocity = np.asarray(
        e_velocity,
        dtype=float,
    )

    reference_acceleration = np.asarray(
        reference_acceleration,
        dtype=float,
    )

    if (
        e_position.shape != (4,)
        or e_velocity.shape != (4,)
        or reference_acceleration.shape != (4,)
    ):
        raise ValueError(
            "All vectors must contain exactly 4 values."
        )

    kappa = (
        -alpha * e_velocity
        -k * e_position
        + reference_acceleration
    )

    # Altitude channel
    kappa[3] += g

    return kappa


if __name__ == "__main__":

    e_position = np.array(
        [
            -0.94719755,
            -0.42359878,
            -1.47079633,
            -2.9,
        ]
    )

    e_velocity = np.array(
        [
            0.1,
            0.1,
            0.1,
            0.1,
        ]
    )

    integral_error = np.zeros(4)

    s = proposed_sliding_surfaces(
        e_position,
        e_velocity,
        integral_error,
    )

    kappa = compute_proposed_kappa(
        e_position=e_position,
        e_velocity=e_velocity,
        reference_acceleration=np.zeros(4),
        g=9.81,
    )

    print(
        "Proposed sliding surfaces:"
    )
    print(s)

    print(
        "\nProposed kappa:"
    )
    print(kappa)

    expected_s = np.array(
        [
            -1.79439510,
            -0.74719756,
            -2.84159266,
            -5.7,
        ]
    )

    expected_kappa = np.array(
        [
            9.2719755,
            4.0359878,
            7.1539816,
            24.11,
        ]
    )

    assert np.allclose(
        s,
        expected_s,
        atol=1e-7,
    )

    assert np.allclose(
        kappa,
        expected_kappa,
        atol=1e-7,
    )

    print(
        "\nProposed tracking "
        "surface test PASSED."
    )