import argparse
import subprocess
import sys
import time


QUICK_MODULES = [
    "src.uav_model",
    "src.tracking",
    "src.adaptive_controller",
    "src.controller",
    "src.type3_fuzzy",
    "src.fuzzy_system",
    "src.fuzzy_compensator",
    "src.stabilized_controller",
    "src.proposed_tracking",
    "experiments.proposed.stabilized_simulation",
]


FULL_MODULES = QUICK_MODULES + [
    "experiments.baseline.stability_check",
    "experiments.proposed.proposed_stability_check",
    "experiments.proposed.generate_results",
    "experiments.proposed.compare_with_paper",
    "experiments.proposed.robustness_ablation",
]


def run_module(module_name):
    print("\n" + "=" * 70)
    print(f"Running: python -m {module_name}")
    print("=" * 70)

    start = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
        ]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if result.returncode != 0:
        print(
            f"\nFAILED: {module_name}"
        )
        print(
            f"Exit code: {result.returncode}"
        )

        return False

    print(
        f"\nPASSED: {module_name}"
    )

    print(
        f"Elapsed: {elapsed:.2f} s"
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run validation and experiments for "
            "the Type-3 fuzzy UAV control project."
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run all validation, baseline, "
            "proposed, robustness and ablation experiments."
        ),
    )

    args = parser.parse_args()

    modules = (
        FULL_MODULES
        if args.full
        else QUICK_MODULES
    )

    mode_name = (
        "FULL"
        if args.full
        else "QUICK"
    )

    print(
        "=" * 70
    )

    print(
        "TYPE-3 FUZZY UAV CONTROL PROJECT"
    )

    print(
        f"{mode_name} VALIDATION"
    )

    print(
        "=" * 70
    )

    total_start = (
        time.perf_counter()
    )

    passed = 0

    for module in modules:

        success = (
            run_module(
                module
            )
        )

        if not success:
            print(
                "\nProject validation stopped "
                "because a module failed."
            )

            sys.exit(1)

        passed += 1

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PROJECT VALIDATION COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Modules passed: "
        f"{passed}/{len(modules)}"
    )

    print(
        f"Total execution time: "
        f"{total_elapsed:.2f} s"
    )

    if args.full:

        print(
            "\nGenerated experiment outputs "
            "are available under:"
        )

        print(
            "  results/figures/"
        )

        print(
            "  results/tables/"
        )

    print(
        "\nAll selected project checks PASSED."
    )


if __name__ == "__main__":
    main()