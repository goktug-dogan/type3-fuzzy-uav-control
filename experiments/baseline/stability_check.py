import numpy as np

from src.simulation import run_simulation
from src.reference_trajectory import ConstantReference


def main():
    horizons = [
        0.10,
        0.25,
        0.50,
        1.00,
    ]

    reference = ConstantReference()
    target = reference.position()

    print(
        "Baseline closed-loop stability diagnostic"
    )
    print("=" * 55)

    for horizon in horizons:

        print(
            f"\nSimulation horizon: {horizon:.2f} s"
        )

        try:
            results = run_simulation(
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
        sliding = results["sliding"]
        fuzzy = results["fuzzy"]
        mass = results["mass_estimate"]

        final_state = states[-1]

        final_output = final_state[
            [0, 2, 4, 6]
        ]

        final_error = (
            final_output - target
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
            "Final sliding surfaces:"
        )
        print(sliding[-1])

        print(
            "Final fuzzy outputs:"
        )
        print(fuzzy[-1])

        print(
            "Final mass estimate:"
        )
        print(mass[-1])

        print(
            "Maximum |control|:"
        )
        print(
            np.max(
                np.abs(controls),
                axis=0,
            )
        )

        print(
            "Maximum |state|:"
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
                np.isfinite(sliding)
            )
            and np.all(
                np.isfinite(fuzzy)
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