import pandas as pd
import os, time
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

from config_val import CLASS_TYPES, FEATURE_CSV_PATH, TEST_SIZE, KF_CROSS, RANDOM_STATE_VAL, RESULT_SAVE_DIR, PARAM_GRID

categories = CLASS_TYPES    # Classes in the dataset target column

# Configure the directory where the outputs of the model would be saved
def configure_directory_to_save_data():
    data_save_dir = RESULT_SAVE_DIR
    # print(data_save_dir)

    # If the directory does not exist, create a new directory.
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    return data_save_dir


# Function to standardize the data based on the scaler passed.
def data_scaling(train_X, test_X, scaler='standard'):
    if(scaler == 'standard'):
        scaling = StandardScaler()
    elif(scaler == 'minmax'):
        scaling = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler. Please specify if you wish to carry out standard or minmax scaler.")

    try:
        X_train = scaling.fit_transform(train_X)
        X_test = scaling.transform(test_X)
    except Exception as e:
        print(e)

    return X_train, X_test


# Execute SVM on dataset with gridsearch
def run_svm_with_gridsearch():
    # Begin timer to monitor time taken to execute entire function and output all results.
    prog_start = time.perf_counter()

    # Read .csv file and get required columns for the model
    df = pd.read_csv(FEATURE_CSV_PATH)

    # Get a copy of the dataset to avoid any changes to the original file
    df_duplicate = df.copy()

    df_copy = df_duplicate.copy()
    df_copy = df_copy.iloc[:, 2:]

    '''
        /Section to undersample data
    '''
    A_classlen = len(df_copy[df_copy['target'] == 'A'])
    B_classlen = len(df_copy[df_copy['target'] == 'B'])

    # Check if the data is balanced.
    # If it is not balanced, perform random undersampling.
    # Here, we extract a number of records from the majority class to match count of the minority class.
    if A_classlen != B_classlen:
        print("Row count for A class: ", A_classlen)
        print("Row count for B class: ", B_classlen, end="\n\n")

        np.random.seed(RANDOM_STATE_VAL)

        # By default, we consider class B to be the majority class.
        length_of_minority = A_classlen
        majority_class_ids = df_copy[df_copy['target'] == 'B'].index
        minority_class_ids = df_copy[df_copy['target'] == 'A'].index

        # If class A is the majority class, switch the labels for majority and minority class.
        if A_classlen > B_classlen:
            length_of_minority = B_classlen
            majority_class_ids = df_copy[df_copy['target'] == 'A'].index
            minority_class_ids = df_copy[df_copy['target'] == 'B'].index

        # Randomly select ids of records belonging to the majority class.
        # The number of ids selected is equal to the length of the minority class. 
        rand_majority_ids = np.random.choice(majority_class_ids, length_of_minority, replace=False)

        # Now we merge the ids of the minority class records and the randomly selected majority ids.
        under_sample_ids = np.concatenate([minority_class_ids, rand_majority_ids])

        # We then select the records that correspond to the ids in the under_sample_ids variable to form a new dataframe.
        under_sample_df = df_copy.loc[under_sample_ids]

        # This data will be used for our SVM model
        df_copy = under_sample_df
    '''
        ./Section to undersample data
    '''

    ## Get all data except target columns
    df_data = df_copy.iloc[:, 0:-1]
    print(df_data.info(), end="\n\n")

    ## Get the number of feature columns
    original_feature_count = len(df_data.columns)

    # The directory to store the results
    result_save_dir = configure_directory_to_save_data()

    # Specify file paths to save results for the outputs of this function
    gridsearch_result_file = result_save_dir + '/gridsearch_result.csv'
    test_result_file = result_save_dir + '/test_result.csv'
    class_report_file = result_save_dir + '/classification_report.csv'
    confusion_matrix_img_file = result_save_dir + '/confusion_matrix.png'

    # Get the values (classes) under the target column
    df_target = df_copy.iloc[:, -1]

    # Assign numerical values for target column to feed the SVM model
    for index, val in df_target.items():
        df_target[index] = categories.index(val)


    # Specify x and y values for the ML model
    x = df_data
    y = df_target.astype(int)

    # Splitting the data into training and testing sets 
    # Here we stratify based on the target column to ensure the proportion of target column values are the same in both train and test set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_VAL, stratify=y)

    # Perform a robust scaler transform of the dataset
    scaled_x_train, scaled_x_test = data_scaling(x_train, x_test, scaler='minmax')

    # Defining the parameters grid for GridSearchCV
    param_grid = PARAM_GRID

    # Creating a support vector classifier
    svc = svm.SVC(probability=True)

    '''
        Defining and training a SVM model using GridSearchCV with parameters grid
    '''
    print("Creating a model with GridSearchCV....")
    model = GridSearchCV(estimator=svc, param_grid=param_grid, cv=KF_CROSS, n_jobs=-1, return_train_score=True)

    # Training the model using the training data
    print("Training the model...")
    model.fit(scaled_x_train, y_train)

    # Store Gridsearch results to a .csv file
    gridsearch_result_df = pd.DataFrame(model.cv_results_)
    gridsearch_result_df.to_csv(gridsearch_result_file)
    print("The best parameters are %s with a score of %0.2f" % (model.best_params_, model.best_score_))
    print("#" * 25, end="\n\n")

    '''
        ./Defining and training a SVM model using GridSearchCV with parameters grid
    '''

    # Testing the model using the test data
    y_pred = model.predict(scaled_x_test)

    # Create and save the confusion matrix diagram to the specified path
    conf_matrix = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=categories)
    conf_matrix.figure_.savefig(confusion_matrix_img_file)

    '''
        Compare the predicted y values with actual y values
    '''
    test_result_dict = {}           # Dictionary to temporarily store comparisons
    test_result_df = pd.DataFrame() # Dataframe to store all comparisons before saving to a file

    count = 0  # Keep count of the records

    for index_val, pred_val in y_test.items():
        if pred_val != y_pred[count]:
            test_result_dict[count] = {
                "record_index": index_val,
                "expected_value": pred_val,
                "test_output": y_pred[count],
                "is_match": False
            }
        else:
            test_result_dict[count] = {
                "record_index": index_val,
                "expected_value": pred_val,
                "test_output": y_pred[count],
                "is_match": True
            }
        count = count + 1   # Increment the record count

    # Save the results for comparison between predicted y values compared to actual values in test set.
    test_result_df = pd.DataFrame.from_dict(test_result_dict, orient="index").reset_index()

    # Save the values if test_result_df is NOT empty
    if not test_result_df.empty:
        test_result_df.to_csv(test_result_file)

    '''
        ./Compare the predicted y values with actual y values
    '''

    # Calculating the accuracy of the model (based in the test set)
    accuracy = accuracy_score(y_pred, y_test)

    # Print the accuracy of the model (based in the test set)
    print(f"The accuracy of the model is {accuracy * 100}%", end="\n\n")
    print("#" * 25, end="\n\n")

    # Print classification report for the model for the test set.
    class_report = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=categories))

    # Save classification report to specified .csv file path
    report_df = pd.DataFrame(class_report).transpose()

    # Save the report if report_df is NOT empty
    if not report_df.empty:
        report_df.to_csv(class_report_file)


    # End the program counter and print the time taken to run the entire function
    prog_end = time.perf_counter()
    print(f'Finished executing the program in {round(prog_end - prog_start, 2)} second(s)')


if __name__ == '__main__':
    run_svm_with_gridsearch()
