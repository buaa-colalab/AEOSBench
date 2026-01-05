__all__ = [
    'GeodeticConversion',
]

import math

from .basilisk.constants import RADIUS_EARTH


class GeodeticConversion:

    @classmethod
    def lla2pcpf(
        cls,
        lla_position: tuple[float, float, float],
        planet_spherical_radius: float = RADIUS_EARTH,
        planet_ellipsoid_radius: float = -1,
    ):
        """Lat/Long/Alt coordinate -> planet-centered planet-fixed coordinate.

        Args:
            lla_position: [rad] Position in PCPF coordinates.
            planet_spherical_radius: [m] Planetary equatorial radius,
            assumed to be constant (i.e., spherical).
            planet_ellipsoid_radius: [m] Planetary polar used for
            elliptical surfaces if provided, otherwise spherical.
        Returns:
            pcpf_position: [m] Position in the planet-centered,
            planet-fixed frame.
        """
        pcpf_position = [0., 0., 0.]
        eccentricity_2 = 0.0
        if planet_ellipsoid_radius >= 0:
            square_ratio = (
                planet_ellipsoid_radius**2 / planet_spherical_radius**2
            )
            eccentricity_2 = 1.0 - square_ratio
        s_phi = math.sin(lla_position[0])
        n_val = planet_spherical_radius / math.sqrt(
            1.0 - eccentricity_2 * s_phi * s_phi,
        )
        pcpf_position[0] = (n_val + lla_position[2]) * (
            math.cos(lla_position[0]) * math.cos(lla_position[1])
        )
        pcpf_position[1] = (n_val + lla_position[2]) * (
            math.cos(lla_position[0]) * math.sin(lla_position[1])
        )
        pcpf_position[2] = (((1.0 - eccentricity_2) * n_val + lla_position[2])
                            * s_phi)
        return pcpf_position
