"""State estimators for drones."""

import os

# crazyflow and scipy's Rotation in jitted code need scipy's array API support, which has to be
# enabled before scipy is imported. Setting it here covers all imports of this package.
os.environ["SCIPY_ARRAY_API"] = "1"
