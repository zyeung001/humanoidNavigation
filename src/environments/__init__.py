# src/environments/__init__.py
"""Walking environment for humanoid locomotion."""

from .walking_env import WalkingEnv, make_walking_env
from .walking_curriculum import WalkingCurriculumEnv, make_walking_curriculum_env

__all__ = [
    'WalkingEnv',
    'make_walking_env',
    'WalkingCurriculumEnv',
    'make_walking_curriculum_env',
]
