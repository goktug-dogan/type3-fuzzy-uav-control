import numpy as np

from src.uav_model import UAVModel
from src.reference_trajectory import ConstantReference

from src.tracking import (
    tracking_errors,
)

from src.adaptive_controller import (
    AdaptiveParameters,
    AdaptiveEstimates,
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

from src.proposed_tracking import (
    proposed_sliding_surfaces,
    compute_proposed_kappa,
)


def control_effectiveness_factors(
    t,
    disturbance_level,
):
    """
    Time-varying multiplicative control-effectiveness
    uncertainty.

    disturbance_level = 0.20 means approximately
    +/-20% variation in actuator effectiveness.

    This disturbance model is an implementation-specific
    robustness experiment and is NOT part of the
    reference paper.
    """

    if not 0.0 <= disturbance_level <= 1.0:
        raise ValueError(
            "disturbance_level must be between 0 and 1."
        )

    return np.array(
        [
            1.0
            + disturbance_level
            * np.sin(0.70 * t),

            1.0
            + disturbance_level
            * np.sin(
                1.10 * t + 0.40
            ),

            1.0
            + disturbance_level
            * np.sin(
                0.90 * t + 0.80
            ),

            1.0
            + disturbance_level
            * np.sin(
                1.30 * t + 1.20
            ),
        ],
        dtype=float,
    )


def run_stabilized_simulation(
    t_final=1.0,
    dt=0.001,
    fuzzy_enabled=True,
    disturbance_level=0.0,
):
    """
    Run the proposed stabilized controller.

    Parameters
    ----------
    t_final:
        Simulation duration.

    dt:
        Numerical integration step.

    fuzzy_enabled:
        If False, T3 fuzzy compensation and fuzzy
        consequent adaptation are disabled.

        This is used for the ablation experiment.

    disturbance_level:
        Multiplicative time-varying uncertainty applied
        to actuator/control effectiveness.

        Example:
            0.05 -> 5%
            0.10 -> 10%
            0.20 -> 20%
    """

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

    config = (
        StabilizationConfig()
    )

    # Initial condition reported
    # in the reference study.
    state = np.full(
        8,
        0.10,
        dtype=float,
    )

    # Proposed surface integrates
    # position tracking error.
    integral_error = np.zeros(
        4,
        dtype=float,
    )

    time_history = []
    state_history = []
    control_history = []
    effective_control_history = []
    raw_control_history = []
    sliding_history = []
    fuzzy_history = []
    mass_history = []
    cos_history = []
    effectiveness_history = []

    n_steps = int(
        np.ceil(
            t_final / dt
        )
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

        s = proposed_sliding_surfaces(
            e_position=e_position,
            e_velocity=e_velocity,
            integral_error=integral_error,
            alpha=(
                adaptive_params.alpha_surface
            ),
            k=(
                adaptive_params.k_surface
            ),
        )

        kappa = compute_proposed_kappa(
            e_position=e_position,
            e_velocity=e_velocity,
            reference_acceleration=(
                reference.acceleration(t)
            ),
            g=model.p.g,
            alpha=(
                adaptive_params.alpha_surface
            ),
            k=(
                adaptive_params.k_surface
            ),
        )

        # ----------------------------------
        # Type-3 fuzzy compensation
        # ----------------------------------

        if fuzzy_enabled:

            (
                fuzzy_outputs,
                xi_vectors,
            ) = fuzzy.evaluate(
                state
            )

        else:

            fuzzy_outputs = np.zeros(
                4,
                dtype=float,
            )

            xi_vectors = None

        # ----------------------------------
        # Proposed stabilized controller
        # ----------------------------------

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

        # ----------------------------------
        # Time-varying actuator uncertainty
        # ----------------------------------

        effectiveness = (
            control_effectiveness_factors(
                t=t,
                disturbance_level=(
                    disturbance_level
                ),
            )
        )

        effective_u = (
            u * effectiveness
        )

        # ----------------------------------
        # Store current values
        # ----------------------------------

        time_history.append(t)

        state_history.append(
            state.copy()
        )

        control_history.append(
            u.copy()
        )

        effective_control_history.append(
            effective_u.copy()
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

        effectiveness_history.append(
            effectiveness.copy()
        )

        if step == n_steps:
            break

        # ----------------------------------
        # True plant dynamics
        # ----------------------------------

        state_dot = (
            model.state_derivative(
                state,
                u1=effective_u[0],
                u2=effective_u[1],
                u3=effective_u[2],
                u4=effective_u[3],
                omega_bar=0.0,
            )
        )

        # ----------------------------------
        # Adaptive parameter laws
        # ----------------------------------

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

        # ----------------------------------
        # T3 consequent adaptation
        # ----------------------------------

        if fuzzy_enabled:

            weight_dot = (
                fuzzy.weight_derivatives(
                    state=state,
                    sliding_surfaces=s,
                    xi_vectors=xi_vectors,
                )
            )

        # ----------------------------------
        # Euler integration
        # ----------------------------------

        state = (
            state
            + dt * state_dot
        )

        integral_error = (
            integral_error
            + dt * e_position
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

        project_adaptive_estimates(
            estimates,
            config,
        )

        if fuzzy_enabled:

            fuzzy.update_weights(
                derivatives=weight_dot,
                dt=dt,
            )

        # ----------------------------------
        # Numerical safety
        # ----------------------------------

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

        "effective_control": np.asarray(
            effective_control_history
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

        "control_effectiveness": np.asarray(
            effectiveness_history
        ),
    }


if __name__ == "__main__":

    results = (
        run_stabilized_simulation(
            t_final=10.0,
            dt=0.001,
            fuzzy_enabled=True,
            disturbance_level=0.0,
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
        final_output
        - target
    )

    print(
        "Final outputs "
        "[phi, theta, psi, z]:"
    )
    print(
        final_output
    )

    print(
        "\nFinal tracking errors:"
    )
    print(
        final_error
    )

    print(
        "\nFinal sliding surfaces:"
    )
    print(
        results[
            "sliding"
        ][-1]
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
                results[
                    "control"
                ]
            ),
            axis=0,
        )
    )

    print(
        "\nMaximum effective |control|:"
    )
    print(
        np.max(
            np.abs(
                results[
                    "effective_control"
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

    print(
        "\nStabilized simulation "
        "test PASSED."
    )