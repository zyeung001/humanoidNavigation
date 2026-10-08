import os
from pathlib import Path

# Configs and MJCFs name their files relative to the repo root (xml_file, actuator_profiles),
# exactly as the training scripts are run. Run the tests from there too.
os.chdir(Path(__file__).resolve().parents[1])
