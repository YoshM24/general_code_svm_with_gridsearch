from sklearn.model_selection import StratifiedKFold

# Different classes or target cloumn values in the dataset (As an example, I have given two classes as A and B)
CLASS_TYPES = ['A', 'B']

# Size of train test split. In this example, we take 20% for testing.
TEST_SIZE = 0.20

# General constants for cross-validation
CV_N_SPLITS = 10
RANDOM_STATE_VAL = 42

# Default conditions for stratified kfold cross-validation
KF_CROSS = StratifiedKFold(shuffle=True, n_splits=CV_N_SPLITS, random_state=RANDOM_STATE_VAL)

# Parameters for SVM Gridsearch (You can vary the parameters as you wish)
PARAM_GRID = {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.1, 1, 10, 100],
              'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'kernel': ['rbf', 'poly']}


# Full path of .CSV file containing the dataset
FEATURE_CSV_PATH = # Add the full path to your file

# Directory to save all outputs of this program
RESULT_SAVE_DIR = # Add the path to your directory
