from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConstantReference:
    """
    Constant reference values used in the simulation
    section of the reference paper.
    """

    phi_d: float = np.pi / 3
    theta_d: float = np.pi / 6
    psi_d: float = np.pi / 2
    z_d: float = 3.0

    def position(self, t=0.0):
        """
        Desired outputs:
        [phi_d, theta_d, psi_d, z_d]
        """
        return np.array(
            [
                self.phi_d,
                self.theta_d,
                self.psi_d,
                self.z_d,
            ],
            dtype=float,
        )

    def velocity(self, t=0.0):
        """
        References are constant, therefore
        their first derivatives are zero.
        """
        return np.zeros(4)

    def acceleration(self, t=0.0):
        """
        References are constant, therefore
        their second derivatives are zero.
        """
        return np.zeros(4)