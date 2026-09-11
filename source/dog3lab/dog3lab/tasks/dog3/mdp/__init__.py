"""This sub-module contains the functions that are specific to the D1 environment.

Merges the standard Isaac Lab mdp namespace with the D1-specific terms,
mirroring DDT_Lab's `locomotion/mdp/__init__.py` pattern.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from instinctlab.envs.mdp.terminations.general import terrain_out_of_bounds  # noqa: F401

from .curriculums import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
