import numpy as np

from src.fuzzy_system import (
    Type3FuzzyChannel,
    state_to_fuzzy_inputs,
)


class Type3FuzzyCompensator:
    """
    Four-channel Type-3 fuzzy compensator.

    Controller mapping:

        f1 -> altitude
        f2 -> roll
        f3 -> pitch
        f4 -> yaw

    The reference paper gives one adaptive consequent
    weight law (Eq. 40), but does not specify separate
    update laws for lower and upper consequent weights.

    Reproduction choice:
        lower_weights == upper_weights

    Both are updated using the same adaptive law.
    """

    CHANNEL_ORDER = (
        "altitude",
        "roll",
        "pitch",
        "yaw",
    )

    def __init__(self):
        self.channels = {
            name: Type3FuzzyChannel(name)
            for name in self.CHANNEL_ORDER
        }

    def evaluate(self, state):
        """
        Evaluate all four fuzzy channels.

        Returns:
            fuzzy_outputs:
                [f1, f2, f3, f4]

            xi_vectors:
                rule firing/basis vectors for
                consequent adaptation.
        """

        fuzzy_inputs = state_to_fuzzy_inputs(
            state
        )

        fuzzy_outputs = []
        xi_vectors = {}

        for name in self.CHANNEL_ORDER:

            output, xi = (
                self.channels[name].evaluate(
                    fuzzy_inputs[name]
                )
            )

            fuzzy_outputs.append(output)
            xi_vectors[name] = xi

        return (
            np.asarray(
                fuzzy_outputs,
                dtype=float,
            ),
            xi_vectors,
        )

    def weight_derivatives(
        self,
        state,
        sliding_surfaces,
        xi_vectors,
    ):
        """
        Consequent-weight adaptation based on Eq. (40).

        Sliding-surface mapping:

            s1 -> roll
            s2 -> pitch
            s3 -> yaw
            s4 -> altitude
        """

        state = np.asarray(
            state,
            dtype=float,
        )

        sliding_surfaces = np.asarray(
            sliding_surfaces,
            dtype=float,
        )

        if state.shape != (8,):
            raise ValueError(
                "State must contain exactly 8 values."
            )

        if sliding_surfaces.shape != (4,):
            raise ValueError(
                "Sliding surfaces must contain "
                "exactly 4 values."
            )

        phi = state[0]
        theta = state[2]

        # Control effectiveness signs:
        #
        # ds1/du2 = b1 > 0
        # ds2/du3 = b2 > 0
        # ds3/du4 = b3 > 0
        #
        # ds4/du1 =
        # cos(phi)*cos(theta)/m
        altitude_effect = (
            np.cos(phi)
            * np.cos(theta)
        )

        channel_signs = {
            "roll": 1.0,
            "pitch": 1.0,
            "yaw": 1.0,
            "altitude": np.sign(
                altitude_effect
            ),
        }

        channel_sliding = {
            "roll": sliding_surfaces[0],
            "pitch": sliding_surfaces[1],
            "yaw": sliding_surfaces[2],
            "altitude": sliding_surfaces[3],
        }

        derivatives = {}

        for name in self.CHANNEL_ORDER:

            xi = np.asarray(
                xi_vectors[name],
                dtype=float,
            )

            derivatives[name] = (
                channel_signs[name]
                * channel_sliding[name]
                * xi
            )

        return derivatives

    def update_weights(
        self,
        derivatives,
        dt,
    ):
        """
        Euler integration of adaptive consequent weights.

        w(t + dt) = w(t) + dt * w_dot(t)
        """

        if dt <= 0.0:
            raise ValueError(
                "dt must be positive."
            )

        for name in self.CHANNEL_ORDER:

            dw = np.asarray(
                derivatives[name],
                dtype=float,
            )

            channel = self.channels[name]

            if dw.shape != (
                channel.n_rules,
            ):
                raise ValueError(
                    f"Invalid derivative shape "
                    f"for {name}."
                )

            # Reproduction implementation choice:
            # use one shared adaptive consequent
            # vector for lower and upper weights.

            channel.lower_weights += (
                dt * dw
            )

            channel.upper_weights += (
                dt * dw
            )


if __name__ == "__main__":

    compensator = (
        Type3FuzzyCompensator()
    )

    # Initial conditions from the paper
    initial_state = np.full(
        8,
        0.10,
    )

    initial_sliding = np.array(
        [
            -1.79439510,
            -0.74719755,
            -2.84159265,
            -5.70000000,
        ]
    )

    # Before adaptation all consequent
    # weights are zero.
    outputs_before, xi = (
        compensator.evaluate(
            initial_state
        )
    )

    print(
        "Fuzzy outputs before adaptation:"
    )
    print(outputs_before)

    print("\nXi sums:")

    for name in (
        compensator.CHANNEL_ORDER
    ):
        print(
            name,
            np.sum(xi[name]),
        )

    assert np.allclose(
        outputs_before,
        np.zeros(4),
        atol=1e-12,
    )

    for name in (
        compensator.CHANNEL_ORDER
    ):
        assert np.isclose(
            np.sum(xi[name]),
            1.0,
            atol=1e-10,
        )

    derivatives = (
        compensator.weight_derivatives(
            state=initial_state,
            sliding_surfaces=(
                initial_sliding
            ),
            xi_vectors=xi,
        )
    )

    print(
        "\nWeight derivative norms:"
    )

    for name in (
        compensator.CHANNEL_ORDER
    ):
        print(
            name,
            np.linalg.norm(
                derivatives[name]
            ),
        )

    # One very small numerical
    # integration step
    dt = 0.001

    compensator.update_weights(
        derivatives,
        dt,
    )

    outputs_after, _ = (
        compensator.evaluate(
            initial_state
        )
    )

    print(
        "\nFuzzy outputs after one "
        "adaptation step:"
    )
    print(outputs_after)

    # Sliding surfaces are initially
    # negative, therefore the adapted
    # fuzzy outputs should move negative.
    assert np.all(
        outputs_after < 0.0
    )

    # Verify shared lower/upper
    # consequent implementation.
    for name in (
        compensator.CHANNEL_ORDER
    ):
        channel = (
            compensator.channels[name]
        )

        assert np.allclose(
            channel.lower_weights,
            channel.upper_weights,
        )

    print(
        "\nFuzzy compensator adaptation "
        "test PASSED."
    )