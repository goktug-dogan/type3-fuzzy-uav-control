import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.proposed.stabilized_simulation import (
    run_stabilized_simulation,
)
from src.reference_trajectory import ConstantReference


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def compute_metrics(
    states,
    reference,
    time=None,
    steady_state_start=5.0,
):
    """
    Compute full-horizon and steady-state
    tracking metrics.

    Full-horizon metrics use the complete
    simulation interval.

    Steady-state metrics use samples from
    steady_state_start until the end.
    """

    states = np.asarray(
        states,
        dtype=float,
    )

    outputs = states[
        :,
        [0, 2, 4, 6]
    ]

    target = reference.position()

    errors = (
        outputs
        - target.reshape(1, -1)
    )

    # --------------------------------
    # Full-horizon metrics
    # --------------------------------

    mse = np.mean(
        errors ** 2,
        axis=0,
    )

    rmse = np.sqrt(
        mse
    )

    max_abs_error = np.max(
        np.abs(errors),
        axis=0,
    )

    final_error = errors[-1]

    # --------------------------------
    # Steady-state metrics
    # --------------------------------

    steady_mse = None
    steady_rmse = None

    if time is not None:

        time = np.asarray(
            time,
            dtype=float,
        )

        if len(time) != len(states):
            raise ValueError(
                "Time and state arrays must "
                "have the same length."
            )

        mask = (
            time >= steady_state_start
        )

        if np.any(mask):

            steady_errors = (
                errors[mask]
            )

            steady_mse = np.mean(
                steady_errors ** 2,
                axis=0,
            )

            steady_rmse = np.sqrt(
                steady_mse
            )

    return {
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "final_error": final_error,
        "steady_mse": steady_mse,
        "steady_rmse": steady_rmse,
    }


def save_time_series(
    results,
):
    """
    Save the complete proposed-controller
    simulation as CSV.
    """

    path = (
        TABLES_DIR
        / "proposed_time_series.csv"
    )

    state = results["state"]
    control = results["control"]
    sliding = results["sliding"]
    fuzzy = results["fuzzy"]
    time = results["time"]
    mass = results[
        "mass_estimate"
    ]

    header = [
        "time",
        "phi",
        "phi_dot",
        "theta",
        "theta_dot",
        "psi",
        "psi_dot",
        "z",
        "z_dot",
        "u1",
        "u2",
        "u3",
        "u4",
        "s1",
        "s2",
        "s3",
        "s4",
        "f1",
        "f2",
        "f3",
        "f4",
        "mass_estimate",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            header
        )

        for i in range(
            len(time)
        ):

            writer.writerow(
                [
                    time[i],
                    *state[i],
                    *control[i],
                    *sliding[i],
                    *fuzzy[i],
                    mass[i],
                ]
            )

    return path


def save_metrics(
    metrics,
):
    """
    Save proposed-controller metrics
    in CSV and JSON formats.
    """

    signal_names = [
        "phi",
        "theta",
        "psi",
        "z",
    ]

    csv_path = (
        TABLES_DIR
        / "proposed_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "signal",
                "mse",
                "rmse",
                "max_abs_error",
                "final_error",
                "steady_mse",
                "steady_rmse",
            ]
        )

        for i, name in enumerate(
            signal_names
        ):

            writer.writerow(
                [
                    name,
                    metrics[
                        "mse"
                    ][i],
                    metrics[
                        "rmse"
                    ][i],
                    metrics[
                        "max_abs_error"
                    ][i],
                    metrics[
                        "final_error"
                    ][i],
                    metrics[
                        "steady_mse"
                    ][i],
                    metrics[
                        "steady_rmse"
                    ][i],
                ]
            )

    json_path = (
        TABLES_DIR
        / "proposed_metrics.json"
    )

    json_data = {}

    for i, name in enumerate(
        signal_names
    ):

        json_data[name] = {
            "mse": float(
                metrics[
                    "mse"
                ][i]
            ),
            "rmse": float(
                metrics[
                    "rmse"
                ][i]
            ),
            "max_abs_error": float(
                metrics[
                    "max_abs_error"
                ][i]
            ),
            "final_error": float(
                metrics[
                    "final_error"
                ][i]
            ),
            "steady_mse": float(
                metrics[
                    "steady_mse"
                ][i]
            ),
            "steady_rmse": float(
                metrics[
                    "steady_rmse"
                ][i]
            ),
        }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_data,
            file,
            indent=4,
        )

    return (
        csv_path,
        json_path,
    )


def plot_outputs(
    results,
    reference,
):
    """
    Save separate tracking plots for
    roll, pitch, yaw, and altitude.
    """

    time = results["time"]
    states = results["state"]

    outputs = states[
        :,
        [0, 2, 4, 6]
    ]

    targets = (
        reference.position()
    )

    names = [
        "Roll phi",
        "Pitch theta",
        "Yaw psi",
        "Altitude z",
    ]

    filenames = [
        "roll_tracking.png",
        "pitch_tracking.png",
        "yaw_tracking.png",
        "altitude_tracking.png",
    ]

    for i in range(4):

        plt.figure(
            figsize=(8, 4.5)
        )

        plt.plot(
            time,
            outputs[:, i],
            label="Actual",
        )

        plt.axhline(
            targets[i],
            linestyle="--",
            label="Reference",
        )

        plt.xlabel(
            "Time (s)"
        )

        plt.ylabel(
            names[i]
        )

        plt.title(
            f"{names[i]} Tracking"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        path = (
            FIGURES_DIR
            / filenames[i]
        )

        plt.savefig(
            path,
            dpi=200,
        )

        plt.close()


def plot_control_inputs(
    results,
):
    """
    Save all four control inputs
    on one figure.
    """

    time = results["time"]
    control = results["control"]

    plt.figure(
        figsize=(9, 5)
    )

    for i in range(4):

        plt.plot(
            time,
            control[:, i],
            label=f"u{i + 1}",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Control input"
    )

    plt.title(
        "Proposed Controller Inputs"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "control_inputs.png",
        dpi=200,
    )

    plt.close()


def plot_sliding_surfaces(
    results,
):
    """
    Save sliding-surface trajectories.
    """

    time = results["time"]
    sliding = results["sliding"]

    plt.figure(
        figsize=(9, 5)
    )

    for i in range(4):

        plt.plot(
            time,
            sliding[:, i],
            label=f"s{i + 1}",
        )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Sliding surface"
    )

    plt.title(
        "Proposed Sliding Surfaces"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "sliding_surfaces.png",
        dpi=200,
    )

    plt.close()


def plot_mass_estimate(
    results,
):
    """
    Plot adaptive mass estimate against
    the nominal UAV mass.
    """

    time = results["time"]

    mass = results[
        "mass_estimate"
    ]

    plt.figure(
        figsize=(8, 4.5)
    )

    plt.plot(
        time,
        mass,
        label="Estimated mass",
    )

    plt.axhline(
        0.486,
        linestyle="--",
        label="Nominal UAV mass",
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Mass estimate (kg)"
    )

    plt.title(
        "Adaptive Mass Estimate"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "mass_estimate.png",
        dpi=200,
    )

    plt.close()


def main():

    print(
        "Running proposed "
        "10-second experiment..."
    )

    # --------------------------------
    # Run experiment
    # --------------------------------

    results = (
        run_stabilized_simulation(
            t_final=10.0,
            dt=0.001,
        )
    )

    reference = (
        ConstantReference()
    )

    # --------------------------------
    # Metrics
    # --------------------------------

    metrics = compute_metrics(
        states=results["state"],
        reference=reference,
        time=results["time"],
        steady_state_start=5.0,
    )

    # --------------------------------
    # Save numerical results
    # --------------------------------

    time_series_path = (
        save_time_series(
            results
        )
    )

    (
        metrics_csv,
        metrics_json,
    ) = save_metrics(
        metrics
    )

    # --------------------------------
    # Generate figures
    # --------------------------------

    plot_outputs(
        results,
        reference,
    )

    plot_control_inputs(
        results
    )

    plot_sliding_surfaces(
        results
    )

    plot_mass_estimate(
        results
    )

    # --------------------------------
    # Terminal summary
    # --------------------------------

    print(
        "\nMSE:"
    )
    print(
        metrics["mse"]
    )

    print(
        "\nRMSE:"
    )
    print(
        metrics["rmse"]
    )

    print(
        "\nMaximum absolute errors:"
    )
    print(
        metrics[
            "max_abs_error"
        ]
    )

    print(
        "\nFinal errors:"
    )
    print(
        metrics[
            "final_error"
        ]
    )

    print(
        "\nSteady-state MSE "
        "(t >= 5 s):"
    )
    print(
        metrics[
            "steady_mse"
        ]
    )

    print(
        "\nSteady-state RMSE "
        "(t >= 5 s):"
    )
    print(
        metrics[
            "steady_rmse"
        ]
    )

    print(
        "\nSaved:"
    )

    print(
        time_series_path
    )

    print(
        metrics_csv
    )

    print(
        metrics_json
    )

    print(
        "\nGenerated figures:"
    )

    print(
        FIGURES_DIR
        / "roll_tracking.png"
    )

    print(
        FIGURES_DIR
        / "pitch_tracking.png"
    )

    print(
        FIGURES_DIR
        / "yaw_tracking.png"
    )

    print(
        FIGURES_DIR
        / "altitude_tracking.png"
    )

    print(
        FIGURES_DIR
        / "control_inputs.png"
    )

    print(
        FIGURES_DIR
        / "sliding_surfaces.png"
    )

    print(
        FIGURES_DIR
        / "mass_estimate.png"
    )

    print(
        "\nProposed experiment "
        "result generation PASSED."
    )


if __name__ == "__main__":
    main()