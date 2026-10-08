from dataclasses import dataclass
import numpy as np


@dataclass
class Type3MembershipFunction:
    """
    Parametric membership function used to reproduce
    Eqs. (19)-(22) of the reference paper.

    center:
        Center of the membership function.

    sigma_left / sigma_right:
        Left and right spreads.

    lambda_lower / lambda_upper:
        Lower and upper slice parameters.
    """

    center: float
    sigma_left: float
    sigma_right: float
    lambda_lower: float
    lambda_upper: float

    def _base_membership(self, x):
        """
        Piecewise triangular base membership.
        """

        x = float(x)

        if x <= self.center - self.sigma_left:
            return 0.0

        if x <= self.center:
            return 1.0 - abs(x - self.center) / self.sigma_left

        if x <= self.center + self.sigma_right:
            return 1.0 - abs(x - self.center) / self.sigma_right

        return 0.0

    def memberships(self, x):
        """
        Compute four Type-3 membership values corresponding
        to the structures described in Eqs. (19)-(22).

        Returns:
            upper_at_upper_slice
            upper_at_lower_slice
            lower_at_upper_slice
            lower_at_lower_slice
        """

        base = self._base_membership(x)

        # Avoid numerical issues at exactly zero.
        if base <= 0.0:
            return np.zeros(4)

        upper_at_upper_slice = base ** self.lambda_upper
        upper_at_lower_slice = base ** self.lambda_lower

        lower_at_upper_slice = base ** (1.0 / self.lambda_upper)
        lower_at_lower_slice = base ** (1.0 / self.lambda_lower)

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
    Compute fuzzy rule firing strength using product inference,
    following the multiplication structure of Eqs. (23)-(26).

    membership_values:
        Array of membership values from all rule antecedents.
    """

    values = np.asarray(membership_values, dtype=float)

    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(
            "membership_values must have shape (n_inputs, 4)."
        )

    return np.prod(values, axis=0)


def type3_weighted_output(
    firing_strengths,
    lower_weights,
    upper_weights,
):
    """
    Simplified numerical implementation of the weighted
    consequent aggregation described in Eqs. (27)-(29).

    Each rule has lower and upper firing strengths and
    lower/upper consequent weights.
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

    if firing_strengths.ndim != 2:
        raise ValueError(
            "firing_strengths must be a 2D array."
        )

    if firing_strengths.shape[1] != 4:
        raise ValueError(
            "Each rule must contain four firing-strength values."
        )

    n_rules = firing_strengths.shape[0]

    if (
        lower_weights.shape != (n_rules,)
        or upper_weights.shape != (n_rules,)
    ):
        raise ValueError(
            "Weight arrays must match the number of rules."
        )

    # Columns:
    # 0 -> upper membership at upper slice
    # 1 -> upper membership at lower slice
    # 2 -> lower membership at upper slice
    # 3 -> lower membership at lower slice

    upper_slice_denominator = (
        firing_strengths[:, 0]
        + firing_strengths[:, 2]
    )

    lower_slice_denominator = (
        firing_strengths[:, 1]
        + firing_strengths[:, 3]
    )

    eps = 1e-12

    G_upper = np.sum(
        firing_strengths[:, 0] * upper_weights
        + firing_strengths[:, 2] * lower_weights
    ) / (np.sum(upper_slice_denominator) + eps)

    G_lower = np.sum(
        firing_strengths[:, 1] * upper_weights
        + firing_strengths[:, 3] * lower_weights
    ) / (np.sum(lower_slice_denominator) + eps)

    return 0.5 * (G_upper + G_lower)


if __name__ == "__main__":
    mf = Type3MembershipFunction(
        center=0.0,
        sigma_left=1.0,
        sigma_right=1.0,
        lambda_lower=0.8,
        lambda_upper=1.2,
    )

    print("Membership at center:")
    center_membership = mf.memberships(0.0)
    print(center_membership)

    print("\nMembership at x = 0.5:")
    half_membership = mf.memberships(0.5)
    print(half_membership)

    print("\nMembership outside support:")
    outside_membership = mf.memberships(2.0)
    print(outside_membership)

    assert np.allclose(
        center_membership,
        np.ones(4),
    )

    assert np.allclose(
        outside_membership,
        np.zeros(4),
    )

    # Two-input test rule
    mf_a = mf.memberships(0.25)
    mf_b = mf.memberships(-0.25)

    firing = rule_firing_strength(
        np.vstack([mf_a, mf_b])
    )

    print("\nExample rule firing strength:")
    print(firing)

    assert firing.shape == (4,)
    assert np.all(firing >= 0.0)
    assert np.all(firing <= 1.0)

    print("\nType-3 membership test PASSED.")