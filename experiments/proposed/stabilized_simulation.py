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

from src.fuzzy_compensator import (
    Type3FuzzyCompensator,
)

from src.stabilized_controller import (
    StabilizationConfig,
    compute_stabilized_control_inputs,
    project_adaptive_estimates,
)


def run_stabilized_simulation(
    t_final=1.0,
    dt=0.001,
):
    model = UAVModel()
    reference = ConstantReference()

    adaptive_params = AdaptiveParameters()
    estimates = AdaptiveEstimates()

    fuzzy = Type3FuzzyCompensator()

    config = StabilizationConfig()

    state = np.full(
        8,
        0.10,
        dtype=float,
    )

    integral_s = np.zeros(
        4,
        dtype=float,
    )

    time_history = []
    state_history = []
    control_history = []
    raw_control_history = []
    sliding_history = []
    fuzzy_history = []
    mass_history = []
    cos_history = []

    n_steps = int(
        np.ceil(t_final / dt)
    )

    for step in range(
        n_steps + 1
    ):
        t = step * dt

        (
            e_position,
            e_velocity,
        ) = tracking_errors(
            state,
            reference,
            t,
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

        (
            fuzzy_outputs,
            xi_vectors,
        ) = fuzzy.evaluate(
            state
        )

        (
            u,
            raw_u,
            safe_cos,
        ) = (
            compute_stabilized_control_inputs(
                state=state,
                kappa=kappa,
                estimates=estimates,
                fuzzy_compensation=(
                    fuzzy_outputs
                ),
                omega_bar=0.0,
                config=config,
            )
        )

        time_history.append(t)
        state_history.append(
            state.copy()
        )
        control_history.append(
            u.copy()
        )
        raw_control_history.append(
            raw_u.copy()
        )
        sliding_history.append(
            s.copy()
        )
        fuzzy_history.append(
            fuzzy_outputs.copy()
        )
        mass_history.append(
            estimates.m_hat
        )
        cos_history.append(
            safe_cos
        )

        if step == n_steps:
            break

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

        weight_dot = (
            fuzzy.weight_derivatives(
                state=state,
                sliding_surfaces=s,
                xi_vectors=xi_vectors,
            )
        )

        # Euler state integration
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

        # Proposed extension:
        # projection after adaptation.
        project_adaptive_estimates(
            estimates,
            config,
        )

        fuzzy.update_weights(
            derivatives=weight_dot,
            dt=dt,
        )

        if not np.all(
            np.isfinite(state)
        ):
            raise RuntimeError(
                f"Non-finite state "
                f"at t={t:.6f}"
            )

    return {
        "time": np.asarray(
            time_history
        ),
        "state": np.asarray(
            state_history
        ),
        "control": np.asarray(
            control_history
        ),
        "raw_control": np.asarray(
            raw_control_history
        ),
        "sliding": np.asarray(
            sliding_history
        ),
        "fuzzy": np.asarray(
            fuzzy_history
        ),
        "mass_estimate": np.asarray(
            mass_history
        ),
        "safe_cos_product": np.asarray(
            cos_history
        ),
    }


if __name__ == "__main__":

    results = (
        run_stabilized_simulation(
            t_final=1.0,
            dt=0.001,
        )
    )

    reference = (
        ConstantReference()
    )

    target = (
        reference.position()
    )

    final_state = (
        results["state"][-1]
    )

    final_output = (
        final_state[
            [0, 2, 4, 6]
        ]
    )

    final_error = (
        final_output - target
    )

    print(
        "Final outputs "
        "[phi, theta, psi, z]:"
    )
    print(final_output)

    print(
        "\nFinal tracking errors:"
    )
    print(final_error)

    print(
        "\nFinal sliding surfaces:"
    )
    print(
        results["sliding"][-1]
    )

    print(
        "\nFinal mass estimate:"
    )
    print(
        results[
            "mass_estimate"
        ][-1]
    )

    print(
        "\nMaximum bounded |control|:"
    )
    print(
        np.max(
            np.abs(
                results["control"]
            ),
            axis=0,
        )
    )

    print(
        "\nMaximum raw |control|:"
    )
    print(
        np.max(
            np.abs(
                results[
                    "raw_control"
                ]
            ),
            axis=0,
        )
    )

    print(
        "\nMinimum absolute safe "
        "cosine product:"
    )
    print(
        np.min(
            np.abs(
                results[
                    "safe_cos_product"
                ]
            )
        )
    )

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

    print(
        "\nStabilized simulation "
        "test PASSED."
    )