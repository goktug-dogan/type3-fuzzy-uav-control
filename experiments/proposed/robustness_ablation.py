import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.proposed.generate_results import (
    compute_metrics,
)

from experiments.proposed.stabilized_simulation import (
    run_stabilized_simulation,
)

from src.reference_trajectory import (
    ConstantReference,
)


RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DISTURBANCE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
]

FUZZY_MODES = [
    True,
    False,
]


def compute_control_effort(
    control,
    time_vector,
):
    """
    Integral of squared commanded control:

        J_u = integral(sum(u_i^2)) dt

    This is used as a comparative
    control-effort metric.
    """

    control = np.asarray(
        control,
        dtype=float,
    )

    time_vector = np.asarray(
        time_vector,
        dtype=float,
    )

    instantaneous_effort = np.sum(
        control ** 2,
        axis=1,
    )

    return float(
        np.trapezoid(
            instantaneous_effort,
            time_vector,
        )
    )


def run_experiment(
    disturbance_level,
    fuzzy_enabled,
):
    """
    Run one 10-second experiment and
    return all comparison metrics.
    """

    start_time = (
        time.perf_counter()
    )

    results = (
        run_stabilized_simulation(
            t_final=10.0,
            dt=0.001,
            fuzzy_enabled=fuzzy_enabled,
            disturbance_level=(
                disturbance_level
            ),
        )
    )

    execution_time = (
        time.perf_counter()
        - start_time
    )

    reference = (
        ConstantReference()
    )

    metrics = compute_metrics(
        states=results["state"],
        reference=reference,
        time=results["time"],
        steady_state_start=5.0,
    )

    control_effort = (
        compute_control_effort(
            control=results["control"],
            time_vector=results["time"],
        )
    )

    max_control = np.max(
        np.abs(
            results["control"]
        ),
        axis=0,
    )

    final_output = (
        results["state"][-1][
            [0, 2, 4, 6]
        ]
    )

    return {
        "disturbance_level": (
            disturbance_level
        ),
        "fuzzy_enabled": (
            fuzzy_enabled
        ),
        "mse": (
            metrics["mse"]
        ),
        "rmse": (
            metrics["rmse"]
        ),
        "steady_mse": (
            metrics["steady_mse"]
        ),
        "steady_rmse": (
            metrics["steady_rmse"]
        ),
        "max_abs_error": (
            metrics["max_abs_error"]
        ),
        "final_error": (
            metrics["final_error"]
        ),
        "final_output": (
            final_output
        ),
        "control_effort": (
            control_effort
        ),
        "max_control": (
            max_control
        ),
        "execution_time": (
            execution_time
        ),
    }


def save_results(
    experiments,
):
    """
    Save all eight experiments into
    one CSV table.
    """

    output_path = (
        TABLES_DIR
        / "robustness_ablation.csv"
    )

    signals = [
        "phi",
        "theta",
        "psi",
        "z",
    ]

    header = [
        "disturbance_percent",
        "fuzzy_enabled",
        "control_effort",
        "execution_time_s",
    ]

    for name in signals:
        header.extend(
            [
                f"{name}_rmse",
                f"{name}_steady_rmse",
                f"{name}_max_abs_error",
                f"{name}_final_error",
            ]
        )

    header.extend(
        [
            "max_u1",
            "max_u2",
            "max_u3",
            "max_u4",
        ]
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)

        for experiment in experiments:

            row = [
                experiment[
                    "disturbance_level"
                ] * 100.0,

                experiment[
                    "fuzzy_enabled"
                ],

                experiment[
                    "control_effort"
                ],

                experiment[
                    "execution_time"
                ],
            ]

            for i in range(4):
                row.extend(
                    [
                        experiment[
                            "rmse"
                        ][i],

                        experiment[
                            "steady_rmse"
                        ][i],

                        experiment[
                            "max_abs_error"
                        ][i],

                        experiment[
                            "final_error"
                        ][i],
                    ]
                )

            row.extend(
                experiment[
                    "max_control"
                ]
            )

            writer.writerow(row)

    return output_path


def print_summary(
    experiments,
):
    """
    Print compact terminal comparison.
    """

    print(
        "\n"
        + "=" * 76
    )

    print(
        "ROBUSTNESS + ABLATION SUMMARY"
    )

    print(
        "=" * 76
    )

    for experiment in experiments:

        fuzzy_name = (
            "T3 ON"
            if experiment[
                "fuzzy_enabled"
            ]
            else "T3 OFF"
        )

        disturbance = (
            experiment[
                "disturbance_level"
            ] * 100.0
        )

        print(
            f"\nDisturbance: "
            f"{disturbance:.0f}%"
            f" | {fuzzy_name}"
        )

        print(
            "RMSE "
            "[phi theta psi z]:"
        )

        print(
            np.round(
                experiment[
                    "rmse"
                ],
                6,
            )
        )

        print(
            "Steady RMSE "
            "[phi theta psi z]:"
        )

        print(
            np.round(
                experiment[
                    "steady_rmse"
                ],
                6,
            )
        )

        print(
            "Final error "
            "[phi theta psi z]:"
        )

        print(
            np.round(
                experiment[
                    "final_error"
                ],
                6,
            )
        )

        print(
            "Control effort:"
        )

        print(
            round(
                experiment[
                    "control_effort"
                ],
                6,
            )
        )

        print(
            "Execution time (s):"
        )

        print(
            round(
                experiment[
                    "execution_time"
                ],
                3,
            )
        )


def plot_rmse_comparison(
    experiments,
):
    """
    Generate one robustness figure
    for each controlled output.
    """

    signal_names = [
        "roll_phi",
        "pitch_theta",
        "yaw_psi",
        "altitude_z",
    ]

    display_names = [
        "Roll RMSE",
        "Pitch RMSE",
        "Yaw RMSE",
        "Altitude RMSE",
    ]

    for signal_index in range(4):

        disturbance_percent = np.array(
            DISTURBANCE_LEVELS
        ) * 100.0

        fuzzy_on_values = []
        fuzzy_off_values = []

        for disturbance in (
            DISTURBANCE_LEVELS
        ):

            for experiment in experiments:

                same_level = np.isclose(
                    experiment[
                        "disturbance_level"
                    ],
                    disturbance,
                )

                if not same_level:
                    continue

                if experiment[
                    "fuzzy_enabled"
                ]:
                    fuzzy_on_values.append(
                        experiment[
                            "rmse"
                        ][signal_index]
                    )
                else:
                    fuzzy_off_values.append(
                        experiment[
                            "rmse"
                        ][signal_index]
                    )

        plt.figure(
            figsize=(8, 4.5)
        )

        plt.plot(
            disturbance_percent,
            fuzzy_on_values,
            marker="o",
            label="T3-FLS enabled",
        )

        plt.plot(
            disturbance_percent,
            fuzzy_off_values,
            marker="s",
            label="T3-FLS disabled",
        )

        plt.xlabel(
            "Control-effectiveness "
            "uncertainty (%)"
        )

        plt.ylabel(
            "RMSE"
        )

        plt.title(
            display_names[
                signal_index
            ]
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / (
                "robustness_"
                + signal_names[
                    signal_index
                ]
                + ".png"
            ),
            dpi=200,
        )

        plt.close()


def plot_control_effort(
    experiments,
):
    """
    Compare commanded control effort
    with and without T3-FLS.
    """

    disturbance_percent = np.array(
        DISTURBANCE_LEVELS
    ) * 100.0

    fuzzy_on_values = []
    fuzzy_off_values = []

    for disturbance in (
        DISTURBANCE_LEVELS
    ):

        for experiment in experiments:

            if not np.isclose(
                experiment[
                    "disturbance_level"
                ],
                disturbance,
            ):
                continue

            if experiment[
                "fuzzy_enabled"
            ]:
                fuzzy_on_values.append(
                    experiment[
                        "control_effort"
                    ]
                )
            else:
                fuzzy_off_values.append(
                    experiment[
                        "control_effort"
                    ]
                )

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.plot(
        disturbance_percent,
        fuzzy_on_values,
        marker="o",
        label="T3-FLS enabled",
    )

    plt.plot(
        disturbance_percent,
        fuzzy_off_values,
        marker="s",
        label="T3-FLS disabled",
    )

    plt.xlabel(
        "Control-effectiveness "
        "uncertainty (%)"
    )

    plt.ylabel(
        "Integral squared control"
    )

    plt.title(
        "Control Effort Under Uncertainty"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "robustness_control_effort.png",
        dpi=200,
    )

    plt.close()


def main():

    experiments = []

    print(
        "Running robustness and "
        "T3-FLS ablation experiments..."
    )

    print(
        "Total experiments: 8"
    )

    experiment_number = 0

    for disturbance_level in (
        DISTURBANCE_LEVELS
    ):

        for fuzzy_enabled in (
            FUZZY_MODES
        ):

            experiment_number += 1

            fuzzy_name = (
                "ON"
                if fuzzy_enabled
                else "OFF"
            )

            print(
                f"\n[{experiment_number}/8] "
                f"uncertainty="
                f"{disturbance_level * 100:.0f}% "
                f"T3={fuzzy_name}"
            )

            result = run_experiment(
                disturbance_level=(
                    disturbance_level
                ),
                fuzzy_enabled=(
                    fuzzy_enabled
                ),
            )

            experiments.append(
                result
            )

            print(
                "Completed."
            )

    output_path = (
        save_results(
            experiments
        )
    )

    plot_rmse_comparison(
        experiments
    )

    plot_control_effort(
        experiments
    )

    print_summary(
        experiments
    )

    print(
        "\nSaved table:"
    )

    print(
        output_path
    )

    print(
        "\nSaved robustness figures "
        "to results/figures/"
    )

    print(
        "\nRobustness + ablation "
        "experiment PASSED."
    )


if __name__ == "__main__":
    main()