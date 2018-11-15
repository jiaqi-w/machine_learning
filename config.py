import os

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

RESOURCES = os.path.join(MAIN_DIR, "resources/")
DATA = os.path.join(MAIN_DIR, "data/")
LOG_DIR = os.path.join(MAIN_DIR, "logger", "logs")
PICKLE_DIR = os.path.join(RESOURCES, "pickles")

EVALUATE_DATA_DIR = os.path.join(DATA, "evaluate")

ML_OUTPUT_DIR = os.path.join(DATA, "lasso_data")
ML_PICKLES_DIR = os.path.join(PICKLE_DIR, "ml_models")
LINEAR_REGRESSION_ML_PICKLES_DIR = os.path.join(PICKLE_DIR, "ml_models", "linear_regression")
PREROCESS_PICKLES_DIR = os.path.join(PICKLE_DIR, "preprocess")

WORD_EMBEDDING_DIR = os.path.join(RESOURCES, "word_embedding")
GLOVE_SIXB = os.path.join(WORD_EMBEDDING_DIR, "word_embedding")
