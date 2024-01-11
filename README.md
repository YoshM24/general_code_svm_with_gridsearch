# About the Project

This is a code I have simplified and generalized based on a code I used previously to create a SVM model with GridSearch.

It is designed for a preprocessed dataset. Hence, it will not look at data cleaning.

The program will read the proprocessed dataset (.csv file defined in the config_val.py file) and perform SVM with GridSearchCV to find the best parameters and the best model.

It will further compare the repdicted results with the actual results for the test set.

The code is tailor made for a dataset which is unbalanced and will require undersampling. Here, random undersampling is adopted.

You can switch between StandardScaler or MinMaxScaler for standardization, or add your own scaler under the data_scaling() function in the main.py file.

The data is split for training and testing with a train_test_split that is startified by the target column.

The common values and file paths can be updated from the config_val.py file. You will find functions and the main program in the main.py file.

I have inserted dummy values for this program (Eg: Target classes = A and B). However, anyone who adopts the code can use it to suit their requirements.

As mentioned above, this is a program I simplified based on one of my previous projects.

I hope the code helps you in your quest to develop an SVM model or any ML model from a preprocessed dataset.

Enjoy your machine learning journey!
