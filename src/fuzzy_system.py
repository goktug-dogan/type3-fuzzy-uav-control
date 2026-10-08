from itertools import product

import numpy as np

from src.fuzzy_config import (
    FuzzyConfig,
    CHANNEL_INPUT_SCALES,
)

from src.type3_fuzzy import (
    Type3MembershipFunction,
    rule_firing_strength,
    type3_output,
)


class Type3FuzzyChannel:
    """
    Two-input Type-3 fuzzy channel.

    Each input uses three linguistic membership functions:

        Negative
        Zero
        Positive

    Therefore:

        3 x 3 = 9 fuzzy rules.
    """

    def __init__(
        self,
        channel_name,
        config=None,
    ):
        if config is None:
            config = FuzzyConfig()

        if channel_name not in CHANNEL_INPUT_SCALES:
            raise ValueError(
                f"Unknown fuzzy channel: {channel_name}"
            )

        self.channel_name = channel_name
        self.config = config

        self.input_scales = (
            CHANNEL_INPUT_SCALES[channel_name]
        )

        self.membership_functions = [
            [
                Type3MembershipFunction(
                    center=center,
                    sigma_left=config.sigma,
                    sigma_right=config.sigma,
                )
                for center in config.centers
            ]
            for _ in range(2)
        ]

        # All combinations:
        #
        # Negative-Negative
        # Negative-Zero
        # Negative-Positive
        # ...
        # Positive-Positive
        self.rules = list(
            product(
                range(len(config.centers)),
                repeat=2,
            )
        )

        self.n_rules = len(self.rules)

        # Consequent parameters start from zero.
        self.lower_weights = np.zeros(
            self.n_rules
        )

        self.upper_weights = np.zeros(
            self.n_rules
        )

    def normalize_inputs(self, values):
        """
        Normalize physical input values to [-1, 1].
        """

        values = np.asarray(
            values,
            dtype=float,
        )

        if values.shape != (2,):
            raise ValueError(
                "A fuzzy channel requires exactly 2 inputs."
            )

        normalized = (
            values / self.input_scales
        )

        return np.clip(
            normalized,
            -1.0,
            1.0,
        )

    def _slice_firing_strengths(
        self,
        normalized_inputs,
        slice_pair,
    ):
        """
        Compute firing strengths of all 9 rules
        for one Type-3 slice pair.
        """

        lambda_lower, lambda_upper = (
            slice_pair
        )

        all_rule_firings = []

        for rule in self.rules:

            memberships = []

            for input_index, mf_index in enumerate(
                rule
            ):
                mf = self.membership_functions[
                    input_index
                ][mf_index]

                values = mf.memberships(
                    normalized_inputs[input_index],
                    lambda_lower,
                    lambda_upper,
                )

                memberships.append(values)

            firing = rule_firing_strength(
                np.vstack(memberships)
            )

            all_rule_firings.append(
                firing
            )

        return np.asarray(
            all_rule_firings,
            dtype=float,
        )

    def evaluate(self, values):
        """
        Evaluate one fuzzy channel.

        Returns:
            fuzzy_output
            xi

        xi is a normalized numerical firing-strength
        basis vector that will later be used in the
        adaptive consequent update.
        """

        normalized_inputs = (
            self.normalize_inputs(values)
        )

        slice_firings = []

        for slice_pair in (
            self.config.slice_pairs
        ):
            firing = (
                self._slice_firing_strengths(
                    normalized_inputs,
                    slice_pair,
                )
            )

            slice_firings.append(
                firing
            )

        fuzzy_output = type3_output(
            slice_firing_strengths=(
                slice_firings
            ),
            lower_weights=(
                self.lower_weights
            ),
            upper_weights=(
                self.upper_weights
            ),
            slice_pairs=(
                self.config.slice_pairs
            ),
        )

        # Build one firing-strength vector
        # for adaptive consequent learning.
        #
        # First average the four Type-3 firing
        # components of each rule.
        per_slice_basis = []

        for firing in slice_firings:
            per_rule = np.mean(
                firing,
                axis=1,
            )

            per_slice_basis.append(
                per_rule
            )

        xi = np.mean(
            np.vstack(per_slice_basis),
            axis=0,
        )

        denominator = np.sum(xi)

        if denominator > 1e-12:
            xi = xi / denominator

        return fuzzy_output, xi


def state_to_fuzzy_inputs(state):
    """
    Split the UAV state into the four fuzzy channels.

    State:
        [
            phi,
            phi_dot,
            theta,
            theta_dot,
            psi,
            psi_dot,
            z,
            z_dot
        ]

    Returns inputs in controller order:
        f1 -> altitude
        f2 -> roll
        f3 -> pitch
        f4 -> yaw
    """

    state = np.asarray(
        state,
        dtype=float,
    )

    if state.shape != (8,):
        raise ValueError(
            "State must contain exactly 8 values."
        )

    (
        phi,
        phi_dot,
        theta,
        theta_dot,
        psi,
        psi_dot,
        z,
        z_dot,
    ) = state

    return {
        "altitude": np.array(
            [z, z_dot]
        ),
        "roll": np.array(
            [phi, phi_dot]
        ),
        "pitch": np.array(
            [theta, theta_dot]
        ),
        "yaw": np.array(
            [psi, psi_dot]
        ),
    }


if __name__ == "__main__":

    initial_state = np.full(
        8,
        0.10,
    )

    fuzzy_inputs = (
        state_to_fuzzy_inputs(
            initial_state
        )
    )

    channel = Type3FuzzyChannel(
        "roll"
    )

    output, xi = channel.evaluate(
        fuzzy_inputs["roll"]
    )

    print("Number of rules:")
    print(channel.n_rules)

    print("\nNormalized roll inputs:")
    print(
        channel.normalize_inputs(
            fuzzy_inputs["roll"]
        )
    )

    print("\nInitial fuzzy output:")
    print(output)

    print("\nAdaptive firing vector xi:")
    print(xi)

    print("\nSum of xi:")
    print(np.sum(xi))

    assert channel.n_rules == 9

    # Consequent weights initially zero
    assert np.isclose(
        output,
        0.0,
        atol=1e-12,
    )

    assert xi.shape == (9,)

    assert np.isclose(
        np.sum(xi),
        1.0,
        atol=1e-10,
    )

    # If every consequent weight is 2,
    # the output must become 2.
    channel.lower_weights[:] = 2.0
    channel.upper_weights[:] = 2.0

    output_two, _ = channel.evaluate(
        fuzzy_inputs["roll"]
    )

    print(
        "\nOutput with all weights = 2:"
    )
    print(output_two)

    assert np.isclose(
        output_two,
        2.0,
        atol=1e-10,
    )

    print(
        "\nType-3 fuzzy channel test PASSED."
    )