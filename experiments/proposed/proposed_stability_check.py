import numpy as np

from experiments.proposed.stabilized_simulation import (
    run_stabilized_simulation,
)
from src.reference_trajectory import ConstantReference


def main():
    horizons = [
        1.0,
        2.0,
        5.0,
        10.0,
    ]

    reference = ConstantReference()
    target = reference.position()

    print(
        "Proposed controller stability diagnostic"
    )
    print("=" * 60)

    for horizon in horizons:

        print(
            f"\nSimulation horizon: {horizon:.2f} s"
        )

        try:
            results = run_stabilized_simulation(
                t_final=horizon,
                dt=0.001,
            )

        except Exception as exc:
            print(
                "FAILED:",
                type(exc).__name__,
                str(exc),
            )
            continue

        states = results["state"]
        controls = results["control"]
        raw_controls = results["raw_control"]
        sliding = results["sliding"]
        mass = results["mass_estimate"]
        safe_cos = results["safe_cos_product"]

        final_output = states[-1][
            [0, 2, 4, 6]
        ]

        final_error = (
            final_output - target
        )

        rmse = np.sqrt(
            np.mean(
                final_error ** 2
            )
        )

        print(
            "Final outputs "
            "[phi, theta, psi, z]:"
        )
        print(final_output)

        print(
            "Final tracking errors:"
        )
        print(final_error)

        print(
            "Final output-error RMSE:"
        )
        print(rmse)

        print(
            "Final sliding surfaces:"
        )
        print(
            sliding[-1]
        )

        print(
            "Final mass estimate:"
        )
        print(
            mass[-1]
        )

        print(
            "Maximum bounded |control|:"
        )
        print(
            np.max(
                np.abs(controls),
                axis=0,
            )
        )

        print(
            "Maximum raw |control|:"
        )
        print(
            np.max(
                np.abs(raw_controls),
                axis=0,
            )
        )

        print(
            "Minimum absolute safe cosine product:"
        )
        print(
            np.min(
                np.abs(safe_cos)
            )
        )

        print(
            "Maximum absolute state:"
        )
        print(
            np.max(
                np.abs(states),
                axis=0,
            )
        )

        finite = (
            np.all(np.isfinite(states))
            and np.all(
                np.isfinite(controls)
            )
            and np.all(
                np.isfinite(raw_controls)
            )
            and np.all(
                np.isfinite(sliding)
            )
            and np.all(
                np.isfinite(mass)
            )
        )

        print(
            "All values finite:",
            finite,
        )


if __name__ == "__main__":
    main()