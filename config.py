import os

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

LOGGER_DIR = os.path.join(MAIN_DIR, "logs/")
RESOURCES = os.path.join(MAIN_DIR, "resources/")
DATA = os.path.join(MAIN_DIR, "data/")
LOG_DIR = os.path.join(MAIN_DIR, "logger", "logs")
PICKLE_DIR = os.path.join(RESOURCES, "pickles")

ML_OUTPUT_DIR = os.path.join(DATA, "lasso_data")
ML_PICKLES_DIR = os.path.join(PICKLE_DIR, "ml_models")
LINEAR_REGRESSION_ML_PICKLES_DIR = os.path.join(PICKLE_DIR, "ml_models", "linear_regression")
PREROCESS_PICKLES_DIR = os.path.join(PICKLE_DIR, "preprocess")

