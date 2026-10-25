"""Robustness: one definition of what can happen to the robot, used by training and testing.

  scenarios.py  Scenario definitions: the eval SUITE (fixed severities) and TRAINING (wide)
  wrapper.py    ScenarioWrapper: plays a scenario per episode on a StandingEnv
  policies.py   what can be evaluated: SB3 .zip, exported .npz, or no policy (hold the pose)
  suite.py      evaluate() -> SuiteResult: the score, per-scenario detail, JSON, a text table
"""

from .scenarios import SUITE, SUITE_VERSION, TRAINING, Scenario, suite_scenarios, training_scenarios
from .wrapper import ScenarioWrapper

__all__ = ["SUITE", "SUITE_VERSION", "TRAINING", "Scenario", "ScenarioWrapper", "suite_scenarios",
           "training_scenarios"]
