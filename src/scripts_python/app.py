import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def print_divider(df):
    # Print current dataset
    print("Current dataset:")
    print(df.head())
    print("-----------------------------------------------------------------------------------")

# Method to clean the data given a Titanic dataset
def clean_data(input_data_location):
    '''Given the path to a csv file, read it and transform the data for logistic regression'''

    # Read the intial csv file
    print("Reading the csv file...")
    df = pd.read_csv(input_data_location)

    # Print current dataset
    print_divider(df)

    # Drop the unncessary columns
    print("Dropping Columns Ticket, Name, and Cabin...")
    df = df.drop(columns = ['Ticket', 'Name', 'Cabin'])

    # Print current dataset
    print_divider(df)

    # Fill the null values
    print("Replacing null values with their appropriate values...")
    df.fillna({'Age': df['Age'].median()}, inplace=True)
    print(f"Age filled with {df['Age'].median()}...")

    df.fillna({'Embarked': 'S'}, inplace=True)
    print(f"Embarked filled with S...")

    df.fillna({'Fare': df['Fare'].median()}, inplace=True)
    print(f"Fare filled with {df['Fare'].median()}...")

    # Print current dataset
    print_divider(df)

    # Get dummy variables
    print("Creating dummy variables for categorical columns...")
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    print("Created dummy variables for Sex and Embarked")

    # Print current dataset
    print_divider(df)

    # Drop the redundant variable
    print("Dropping redundant dummy variables...")
    df = df.drop(columns=['Sex_female', 'Embarked_C'])

    # Print current dataset
    print_divider(df)


    # Create the new Alone column
    print("Creating new Alone column to test whether passenger was traveling alone...")
    df['Alone'] = np.where((df['SibSp']+df['Parch'])>0, 0, 1)

    # Print current dataset
    print_divider(df)

    # Drop the unncessary columns
    print("Dropping redundant columns SibSp and Parch for multicollinearity")
    df = df.drop(columns=['SibSp', 'Parch'])

    # Print current dataset
    print_divider(df)

    # Return the final dataframe
    return df


def run_logistic_regression(train_data_location = "", test_data_location = ""):
    '''Given paths for both the train and test dataset, run a logistic regression model
    on the training data to predict whether passengers in the test data survived or not'''

    # Clean the training data
    print("Starting data cleaning for train data...")
    df_train = clean_data(train_data_location)

    # Sleep to see the difference
    time.sleep(5)

    # Clean the test data
    print("Starting data cleaning for test data...")
    df_test = clean_data(test_data_location)

    # Separate train and test data
    X_train = df_train[['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Alone']]
    y_train = df_train['Survived']

    X_test = df_test[['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Alone']]

    # Create a Logistic Regression Model and fit it with the training data
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict with this model on the test data
    y_pred = model.predict(X_test)

    # Add the prediction to the test dataframe
    df_test['Predicted Survived'] = y_pred

    print(f"Percentage of people predicted to have survived: {float(np.round(df_test['Predicted Survived'].value_counts()[1] / len(df_test) * 100.0, 2))}%")

    return df_test[['PassengerId', 'Predicted Survived']]



# Main class to run the whole program
def main():
    df_final = run_logistic_regression("/data/train.csv", "/data/test.csv")
    print(f"Final Test Data With Prediction:")
    print(df_final)
    # Save to data/gender_submission.csv
    print(f"Saving DataFrame to a Kaggle Submission CSV File...")
    df_gender_submission = df_final[['PassengerId', 'Predicted Survived']].rename(columns={'Predicted Survived': 'Survived'})

    # Save to the gender_submission.csv file
    print(f"gender_submission.csv updated!")
    df_gender_submission.to_csv("/data/gender_submission.csv", index=False)

# Run main()
if __name__ == "__main__":
    main()