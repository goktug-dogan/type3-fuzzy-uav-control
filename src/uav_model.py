from dataclasses import dataclass
import numpy as np


@dataclass
class UAVParameters:
    """
    Physical parameters reported in Table 1 of the reference paper.
    """

    m: float = 0.486
    d: float = 0.25

    Ix: float = 3.8277e-3
    Iy: float = 3.8277e-3
    Iz: float = 7.6565e-3

    Kfax: float = 5.5671e-4
    Kfay: float = 5.5671e-4
    Kfaz: float = 6.3541e-4

    Kfdz: float = 6.3541e-4

    Cd: float = 3.2321e-2
    Jr: float = 2.8386e-5

    # The paper denotes g as gravitational acceleration
    # but does not give a numerical value in Table 1.
    # Standard gravitational acceleration is used here.
    g: float = 9.81


class UAVModel:
    """
    8-state UAV model based on Eqs. (12)-(16) of the reference paper.

    State vector:
        x1 = phi        : roll angle
        x2 = phi_dot    : roll angular velocity
        x3 = theta      : pitch angle
        x4 = theta_dot  : pitch angular velocity
        x5 = psi        : yaw angle
        x6 = psi_dot    : yaw angular velocity
        x7 = z          : altitude
        x8 = z_dot      : vertical velocity
    """

    def __init__(self, params=None):
        self.p = params if params is not None else UAVParameters()
        self._calculate_coefficients()

    def _calculate_coefficients(self):
        p = self.p

        self.a1 = (p.Iy - p.Iz) / p.Ix
        self.a2 = -p.Kfax / p.Ix
        self.a3 = -p.Jr / p.Ix

        self.a4 = (p.Iz - p.Ix) / p.Iy
        self.a5 = -p.Kfay / p.Iy
        self.a6 = p.Jr / p.Iy

        self.a7 = (p.Ix - p.Iy) / p.Iz

        # Eq. (12) contains -Kfaz * psi_dot^2 / Iz.
        # Therefore a8 follows directly from Eq. (12).
        self.a8 = -p.Kfaz / p.Iz

        self.a9 = -p.Kfdz / p.m

        self.b1 = p.d / p.Ix
        self.b2 = p.d / p.Iy
        self.b3 = p.Cd / p.Iz

    def state_derivative(
        self,
        state,
        u1=0.0,
        u2=0.0,
        u3=0.0,
        u4=0.0,
        omega_bar=0.0,
    ):
        """
        Compute x_dot for the UAV.

        omega_bar = w1 - w2 + w3 - w4

        Control inputs:
            u1 : total thrust / altitude control
            u2 : roll control
            u3 : pitch control
            u4 : yaw control
        """

        x = np.asarray(state, dtype=float)

        if x.shape != (8,):
            raise ValueError("State must contain exactly 8 values.")

        phi, phi_dot, theta, theta_dot, psi, psi_dot, z, z_dot = x

        phi_ddot = (
            self.a1 * theta_dot * psi_dot
            + self.a2 * phi_dot**2
            + self.a3 * omega_bar * theta_dot
            + self.b1 * u2
        )

        theta_ddot = (
            self.a4 * phi_dot * psi_dot
            + self.a5 * theta_dot**2
            + self.a6 * omega_bar * phi_dot
            + self.b2 * u3
        )

        psi_ddot = (
            self.a7 * phi_dot * theta_dot
            + self.a8 * psi_dot**2
            + self.b3 * u4
        )

        z_ddot = (
            self.a9 * z_dot
            - self.p.g
            + (np.cos(phi) * np.cos(theta) / self.p.m) * u1
        )

        return np.array(
            [
                phi_dot,
                phi_ddot,
                theta_dot,
                theta_ddot,
                psi_dot,
                psi_ddot,
                z_dot,
                z_ddot,
            ]
        )


if __name__ == "__main__":
    model = UAVModel()

    # Paper simulation starts every UAV state at 0.10.
    initial_state = np.full(8, 0.10)

    dx = model.state_derivative(initial_state)

    print("Initial state:")
    print(initial_state)

    print("\nState derivative with zero control:")
    print(dx)

    print("\nModel coefficients:")
    print("a1 =", model.a1)
    print("a2 =", model.a2)
    print("a3 =", model.a3)
    print("a4 =", model.a4)
    print("a5 =", model.a5)
    print("a6 =", model.a6)
    print("a7 =", model.a7)
    print("a8 =", model.a8)
    print("a9 =", model.a9)
    print("b1 =", model.b1)
    print("b2 =", model.b2)
    print("b3 =", model.b3)