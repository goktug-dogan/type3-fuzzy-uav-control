import numpy as np

from src.uav_model import UAVModel
from src.reference_trajectory import ConstantReference

from src.tracking import (
    tracking_errors,
    sliding_surfaces,
)

from src.adaptive_controller import (
    AdaptiveParameters,
    AdaptiveEstimates,
    compute_kappa,
    adaptation_derivatives,
)

from src.controller import (
    compute_control_inputs,
)

from src.fuzzy_compensator import (
    Type3FuzzyCompensator,
)


def run_simulation(
    t_final=0.1,
    dt=0.001,
):
    """
    Short closed-loop numerical simulation.

    This first version uses explicit Euler integration.

    It is intended as an integration smoke test,
    not yet as the final reproduction experiment.
    """

    if t_final <= 0.0:
        raise ValueError(
            "t_final must be positive."
        )

    if dt <= 0.0:
        raise ValueError(
            "dt must be positive."
        )

    model = UAVModel()
    reference = ConstantReference()

    adaptive_params = (
        AdaptiveParameters()
    )

    estimates = (
        AdaptiveEstimates()
    )

    fuzzy = (
        Type3FuzzyCompensator()
    )

    # Reference paper:
    # chi_i(0) = 0.10
    state = np.full(
        8,
        0.10,
        dtype=float,
    )

    # Integral terms appearing
    # in Eq. (31).
    integral_s = np.zeros(
        4,
        dtype=float,
    )

    times = []
    states = []
    controls = []
    sliding_history = []
    fuzzy_history = []
    mass_estimate_history = []

    n_steps = int(
        np.ceil(t_final / dt)
    )

    for step in range(n_steps + 1):

        t = step * dt

        e_position, e_velocity = (
            tracking_errors(
                state,
                reference,
                t,
            )
        )

        s = sliding_surfaces(
            e_position,
            e_velocity,
            integral_s,
            alpha=(
                adaptive_params.alpha_surface
            ),
            k=(
                adaptive_params.k_surface
            ),
        )

        kappa = compute_kappa(
            e_velocity=e_velocity,
            s=s,
            reference_acceleration=(
                reference.acceleration(t)
            ),
            g=model.p.g,
            params=adaptive_params,
        )

        fuzzy_outputs, xi_vectors = (
            fuzzy.evaluate(state)
        )

        u = compute_control_inputs(
            state=state,
            kappa=kappa,
            estimates=estimates,
            fuzzy_compensation=(
                fuzzy_outputs
            ),
            omega_bar=0.0,
        )

        # Save current state before
        # numerical integration.
        times.append(t)
        states.append(state.copy())
        controls.append(u.copy())
        sliding_history.append(
            s.copy()
        )
        fuzzy_history.append(
            fuzzy_outputs.copy()
        )
        mass_estimate_history.append(
            estimates.m_hat
        )

        if step == n_steps:
            break

        # UAV dynamics
        state_dot = (
            model.state_derivative(
                state,
                u1=u[0],
                u2=u[1],
                u3=u[2],
                u4=u[3],
                omega_bar=0.0,
            )
        )

        # Adaptive model parameter
        # derivatives
        (
            zeta_dot,
            h_dot,
            m_hat_dot,
        ) = adaptation_derivatives(
            state=state,
            s=s,
            kappa=kappa,
            omega_bar=0.0,
            params=adaptive_params,
        )

        # Type-3 consequent
        # adaptation
        weight_dot = (
            fuzzy.weight_derivatives(
                state=state,
                sliding_surfaces=s,
                xi_vectors=xi_vectors,
            )
        )

        # ------------------------
        # Explicit Euler update
        # ------------------------

        state = (
            state
            + dt * state_dot
        )

        integral_s = (
            integral_s
            + dt * s
        )

        estimates.zeta_hat = (
            estimates.zeta_hat
            + dt * zeta_dot
        )

        estimates.h_hat = (
            estimates.h_hat
            + dt * h_dot
        )

        estimates.m_hat = (
            estimates.m_hat
            + dt * m_hat_dot
        )

        fuzzy.update_weights(
            derivatives=weight_dot,
            dt=dt,
        )

        # Numerical safety check
        if not np.all(
            np.isfinite(state)
        ):
            raise RuntimeError(
                f"Non-finite UAV state "
                f"detected at t={t:.6f}"
            )

        if not np.all(
            np.isfinite(
                estimates.zeta_hat
            )
        ):
            raise RuntimeError(
                "Non-finite adaptive "
                "parameter detected."
            )

        if not np.isfinite(
            estimates.m_hat
        ):
            raise RuntimeError(
                "Non-finite mass estimate "
                "detected."
            )

    return {
        "time": np.asarray(times),
        "state": np.asarray(states),
        "control": np.asarray(controls),
        "sliding": np.asarray(
            sliding_history
        ),
        "fuzzy": np.asarray(
            fuzzy_history
        ),
        "mass_estimate": np.asarray(
            mass_estimate_history
        ),
    }


if __name__ == "__main__":

    results = run_simulation(
        t_final=0.1,
        dt=0.001,
    )

    final_state = (
        results["state"][-1]
    )

    final_sliding = (
        results["sliding"][-1]
    )

    final_fuzzy = (
        results["fuzzy"][-1]
    )

    final_mass_estimate = (
        results[
            "mass_estimate"
        ][-1]
    )

    max_control = np.max(
        np.abs(
            results["control"]
        ),
        axis=0,
    )

    print(
        "Number of simulation samples:"
    )
    print(
        len(results["time"])
    )

    print(
        "\nInitial UAV state:"
    )
    print(
        results["state"][0]
    )

    print(
        "\nFinal UAV state:"
    )
    print(final_state)

    print(
        "\nFinal sliding surfaces:"
    )
    print(final_sliding)

    print(
        "\nFinal fuzzy outputs:"
    )
    print(final_fuzzy)

    print(
        "\nFinal mass estimate:"
    )
    print(final_mass_estimate)

    print(
        "\nMaximum absolute "
        "control inputs:"
    )
    print(max_control)

    assert len(
        results["time"]
    ) == 101

    assert np.all(
        np.isfinite(
            results["state"]
        )
    )

    assert np.all(
        np.isfinite(
            results["control"]
        )
    )

    assert np.all(
        np.isfinite(
            results["fuzzy"]
        )
    )

    print(
        "\nClosed-loop smoke "
        "test PASSED."
    )