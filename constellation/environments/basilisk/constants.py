from Basilisk.architecture import astroConstants
from Basilisk.utilities.simIncludeGravBody import BODY_DATA

RADIUS_EARTH: float = astroConstants.REQ_EARTH * 1e3  # in m

eccentricity_2 = 0.0  # TODO

MU_EARTH = BODY_DATA['earth'].mu  # in m^3/s^2

IDENTITY_MATRIX_3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]

UNIT_VECTOR_Z = [0, 0, 1]
