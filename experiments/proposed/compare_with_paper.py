import csv
from pathlib import Path

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


OUTPUT_PATH = Path(
    "results/tables/paper_vs_proposed.csv"
)


def main():

    signal_names = [
        "phi",
        "theta",
        "psi",
        "z",
    ]

    # Table 2 values explicitly reported
    # in the reference paper.
    paper_mse = np.array(
        [
            0.0284,
            0.0069,
            0.0612,
            0.2236,
        ]
    )

    print(
        "Running proposed controller..."
    )

    results = (
        run_stabilized_simulation(
            t_final=10.0,
            dt=0.001,
        )
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

    proposed_full_mse = (
        metrics["mse"]
    )

    proposed_steady_mse = (
        metrics["steady_mse"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "signal",
                "paper_reported_mse",
                "proposed_full_10s_mse",
                "proposed_steady_5_10s_mse",
                "steady_vs_full_reduction_percent",
            ]
        )

        for i, name in enumerate(
            signal_names
        ):

            reduction = (
                1.0
                - proposed_steady_mse[i]
                / proposed_full_mse[i]
            ) * 100.0

            writer.writerow(
                [
                    name,
                    paper_mse[i],
                    proposed_full_mse[i],
                    proposed_steady_mse[i],
                    reduction,
                ]
            )

    print(
        "\nPaper reported MSE:"
    )
    print(paper_mse)

    print(
        "\nProposed full 10 s MSE:"
    )
    print(
        proposed_full_mse
    )

    print(
        "\nProposed steady-state MSE "
        "(5-10 s):"
    )
    print(
        proposed_steady_mse
    )

    reduction = (
        1.0
        - proposed_steady_mse
        / proposed_full_mse
    ) * 100.0

    print(
        "\nSteady-state improvement "
        "relative to full-horizon MSE (%):"
    )
    print(reduction)

    print(
        "\nIMPORTANT:"
    )
    print(
        "Paper MSE and proposed steady-state "
        "MSE must not be treated as directly "
        "equivalent evaluation protocols."
    )

    print(
        "\nSaved:"
    )
    print(
        OUTPUT_PATH
    )

    print(
        "\nComparison generation PASSED."
    )


if __name__ == "__main__":
    main()