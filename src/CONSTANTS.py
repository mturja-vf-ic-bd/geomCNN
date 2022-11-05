# This file defines a set constants related to file paths / hyperparamters for training DL models
import os

_HOME = os.path.expanduser('~')
if "Users" in _HOME:
    PREFIX = "Users"
else:
    PREFIX = "home"

FILE_PATHS = {
    "BASE_DIR": f"/{PREFIX}/mturja/geomCNN",
    "TRAIN_DATA_DIR": f"/{PREFIX}/mturja/IBIS_cortical_features",
    "TEST_DATA_DIR": f"/{PREFIX}/mturja/surface_data_test",
    "FEATURE_DIRS": ["eacsf"],
    "FILE_SUFFIX": ["_flat", "_flat", "_flat"],
    "TIME_POINTS": ["V06", "V12"]
}

# HYPERPARAMERS = {
#     ""
# }
