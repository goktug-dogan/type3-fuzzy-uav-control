from dataclasses import dataclass
import numpy as np


@dataclass
class Type3MembershipFunction:
    """
    Type-3 membership function structure based on
    Eqs. (19)-(22) of the reference paper.

    center:
        Center of the membership function.

    sigma_left:
        Left spread.

    sigma_right:
        Right spread.

    Note:
    The reference paper provides the mathematical structure
    but does not report the numerical centers, spreads, or
    slice values used in the simulation.
    """

    center: float
    sigma_left: float
    sigma_right: float

    def _base_membership(self, x):
        """
        Piecewise triangular base membership.
        """

        x = float(x)

        if self.sigma_left <= 0 or self.sigma_right <= 0:
            raise ValueError("Sigma values must be positive.")

        if x <= self.center - self.sigma_left:
            return 0.0

        if x <= self.center:
            return 1.0 - (
                abs(x - self.center) / self.sigma_left
            )

        if x <= self.center + self.sigma_right:
            return 1.0 - (
                abs(x - self.center) / self.sigma_right
            )

        return 0.0

    def memberships(
        self,
        x,
        lambda_lower,
        lambda_upper,
    ):
        """
        Evaluate the four Type-3 membership values
        corresponding to Eqs. (19)-(22).

        lambda_lower and lambda_upper represent a
        lower/upper slice pair.

        Returns:
            [
                upper_at_upper_slice,
                upper_at_lower_slice,
                lower_at_upper_slice,
                lower_at_lower_slice
            ]
        """

        if not (
            0.0 < lambda_lower
            <= lambda_upper
            <= 1.0
        ):
            raise ValueError(
                "Slice values must satisfy "
                "0 < lambda_lower <= lambda_upper <= 1."
            )

        base = self._base_membership(x)

        if base <= 0.0:
            return np.zeros(4)

        upper_at_upper_slice = (
            base ** lambda_upper
        )

        upper_at_lower_slice = (
            base ** lambda_lower
        )

        lower_at_upper_slice = (
            base ** (1.0 / lambda_upper)
        )

        lower_at_lower_slice = (
            base ** (1.0 / lambda_lower)
        )

        return np.array(
            [
                upper_at_upper_slice,
                upper_at_lower_slice,
                lower_at_upper_slice,
                lower_at_lower_slice,
            ],
            dtype=float,
        )


def rule_firing_strength(membership_values):
    """
    Product inference corresponding to Eqs. (23)-(26).

    membership_values shape:
        (n_inputs, 4)

    Returns four firing strengths for one rule.
    """

    values = np.asarray(
        membership_values,
        dtype=float,
    )

    if (
        values.ndim != 2
        or values.shape[1] != 4
    ):
        raise ValueError(
            "membership_values must have shape "
            "(n_inputs, 4)."
        )

    return np.prod(values, axis=0)


def slice_consequent_output(
    firing_strengths,
    lower_weights,
    upper_weights,
):
    """
    Compute G values for a single slice,
    following Eqs. (28)-(29).

    firing_strengths:
        shape (n_rules, 4)
    """

    firing_strengths = np.asarray(
        firing_strengths,
        dtype=float,
    )

    lower_weights = np.asarray(
        lower_weights,
        dtype=float,
    )

    upper_weights = np.asarray(
        upper_weights,
        dtype=float,
    )

    if (
        firing_strengths.ndim != 2
        or firing_strengths.shape[1] != 4
    ):
        raise ValueError(
            "firing_strengths must have shape "
            "(n_rules, 4)."
        )

    n_rules = firing_strengths.shape[0]

    if (
        lower_weights.shape != (n_rules,)
        or upper_weights.shape != (n_rules,)
    ):
        raise ValueError(
            "Weight vectors must match "
            "the number of rules."
        )

    eps = 1e-12

    # Upper lambda slice
    numerator_upper = np.sum(
        firing_strengths[:, 0] * upper_weights
        + firing_strengths[:, 2] * lower_weights
    )

    denominator_upper = np.sum(
        firing_strengths[:, 0]
        + firing_strengths[:, 2]
    )

    G_upper = (
        numerator_upper
        / (denominator_upper + eps)
    )

    # Lower lambda slice
    numerator_lower = np.sum(
        firing_strengths[:, 1] * upper_weights
        + firing_strengths[:, 3] * lower_weights
    )

    denominator_lower = np.sum(
        firing_strengths[:, 1]
        + firing_strengths[:, 3]
    )

    G_lower = (
        numerator_lower
        / (denominator_lower + eps)
    )

    return G_lower, G_upper


def type3_output(
    slice_firing_strengths,
    lower_weights,
    upper_weights,
    slice_pairs,
):
    """
    Aggregate all z-slices according to Eq. (27).

    slice_firing_strengths:
        list/array with one (n_rules, 4)
        firing-strength matrix per slice.

    slice_pairs:
        [
            (lambda_lower, lambda_upper),
            ...
        ]
    """

    if (
        len(slice_firing_strengths)
        != len(slice_pairs)
    ):
        raise ValueError(
            "Number of firing-strength sets "
            "must match number of slice pairs."
        )

    numerator = 0.0
    denominator = 0.0

    for firing, (
        lambda_lower,
        lambda_upper,
    ) in zip(
        slice_firing_strengths,
        slice_pairs,
    ):
        G_lower, G_upper = (
            slice_consequent_output(
                firing,
                lower_weights,
                upper_weights,
            )
        )

        numerator += (
            lambda_lower * G_lower
            + lambda_upper * G_upper
        )

        denominator += (
            lambda_lower + lambda_upper
        )

    if denominator <= 0.0:
        raise ValueError(
            "Invalid slice denominator."
        )

    return numerator / denominator


if __name__ == "__main__":
    mf = Type3MembershipFunction(
        center=0.0,
        sigma_left=1.0,
        sigma_right=1.0,
    )

    lambda_lower = 0.5
    lambda_upper = 1.0

    center_membership = mf.memberships(
        0.0,
        lambda_lower,
        lambda_upper,
    )

    half_membership = mf.memberships(
        0.5,
        lambda_lower,
        lambda_upper,
    )

    outside_membership = mf.memberships(
        2.0,
        lambda_lower,
        lambda_upper,
    )

    print("Membership at center:")
    print(center_membership)

    print("\nMembership at x = 0.5:")
    print(half_membership)

    print("\nMembership outside support:")
    print(outside_membership)

    assert np.allclose(
        center_membership,
        np.ones(4),
    )

    assert np.allclose(
        outside_membership,
        np.zeros(4),
    )

    # Example two-input rule
    mf_a = mf.memberships(
        0.25,
        lambda_lower,
        lambda_upper,
    )

    mf_b = mf.memberships(
        -0.25,
        lambda_lower,
        lambda_upper,
    )

    firing = rule_firing_strength(
        np.vstack([mf_a, mf_b])
    )

    print("\nExample rule firing:")
    print(firing)

    assert firing.shape == (4,)
    assert np.all(firing >= 0.0)
    assert np.all(firing <= 1.0)

    # Aggregation sanity check:
    # If both consequent bounds are 2,
    # fuzzy output must also be 2.
    firing_matrix = np.vstack(
        [
            firing,
            firing,
        ]
    )

    lower_weights = np.array(
        [2.0, 2.0]
    )

    upper_weights = np.array(
        [2.0, 2.0]
    )

    output = type3_output(
        slice_firing_strengths=[
            firing_matrix
        ],
        lower_weights=lower_weights,
        upper_weights=upper_weights,
        slice_pairs=[
            (
                lambda_lower,
                lambda_upper,
            )
        ],
    )

    print("\nExample Type-3 output:")
    print(output)

    assert np.isclose(
        output,
        2.0,
        atol=1e-10,
    )

    print(
        "\nType-3 fuzzy inference test PASSED."
    )