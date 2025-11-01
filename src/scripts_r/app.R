# Required Libraries
suppressWarnings({
  library(readr)
  library(dplyr)
  library(magrittr)
})

# Function to print divider
print_divider <- function(df) {
  cat("Current dataset:\n")
  print(head(df))
  cat("-----------------------------------------------------------------------------------\n")
}

# Function to clean Titanic data
clean_data <- function(input_data_location) {
  cat("Reading the csv file...\n")
  df <- read_csv(input_data_location, show_col_types = FALSE)
  print_divider(df)
  
  # Drop unnecessary columns
  cat("Dropping Columns Ticket, Name, and Cabin...\n")
  df <- df %>% select(-Ticket, -Name, -Cabin)
  print_divider(df)
  
  # Fill null values
  cat("Replacing null values with appropriate values...\n")
  df$Age[is.na(df$Age)] <- median(df$Age, na.rm = TRUE)
  cat(paste("Age filled with", median(df$Age, na.rm = TRUE), "...\n"))
  
  df$Embarked[is.na(df$Embarked)] <- 'S'
  cat("Embarked filled with S...\n")
  
  df$Fare[is.na(df$Fare)] <- median(df$Fare, na.rm = TRUE)
  cat(paste("Fare filled with", median(df$Fare, na.rm = TRUE), "...\n"))
  
  print_divider(df)
  
  # Create dummy variables
  cat("Creating dummy variables for categorical columns...\n")
  df <- df %>%
    mutate(
      Sex_male = ifelse(Sex == "male", 1, 0),
      Embarked_Q = ifelse(Embarked == "Q", 1, 0),
      Embarked_S = ifelse(Embarked == "S", 1, 0)
    ) %>%
    select(-Sex, -Embarked)
  
  print_divider(df)
  
  # Create Alone column
  cat("Creating new Alone column to test whether passenger was traveling alone...\n")
  df <- df %>% mutate(Alone = ifelse((SibSp + Parch) > 0, 0, 1))
  
  print_divider(df)
  
  # Drop redundant columns
  cat("Dropping redundant columns SibSp and Parch for multicollinearity...\n")
  df <- df %>% select(-SibSp, -Parch)
  
  print_divider(df)
  
  return(df)
}

# Function to run logistic regression
run_logistic_regression <- function(train_data_location, test_data_location) {
  cat("Starting data cleaning for train data...\n")
  df_train <- clean_data(train_data_location)
  Sys.sleep(5)
  
  cat("Starting data cleaning for test data...\n")
  df_test <- clean_data(test_data_location)
  
  # Separate predictors and target
  X_train <- df_train %>% select(Pclass, Age, Fare, Sex_male, Embarked_Q, Embarked_S, Alone)
  y_train <- df_train$Survived
  
  X_test <- df_test %>% select(Pclass, Age, Fare, Sex_male, Embarked_Q, Embarked_S, Alone)
  
  # Fit logistic regression
  cat("Training Logistic Regression model...\n")
  model <- glm(y_train ~ ., data = X_train, family = binomial)
  
  # Predict
  y_pred_prob <- predict(model, newdata = X_test, type = "response")
  y_pred <- ifelse(y_pred_prob > 0.5, 1, 0)
  
  df_test$Predicted_Survived <- y_pred
  
  pct_survived <- round(mean(df_test$Predicted_Survived) * 100, 2)
  cat(paste("Percentage of people predicted to have survived:", pct_survived, "%\n"))
  
  return(df_test %>% select(PassengerId, Predicted_Survived))
}

# Main function
main <- function() {
  df_final <- run_logistic_regression("/data/train.csv", "/data/test.csv")
  
  cat("Final Test Data With Prediction:\n")
  print(df_final)
  
  cat("Saving DataFrame to a Kaggle Submission CSV File...\n")
  df_gender_submission <- df_final %>%
    select(PassengerId, Survived = Predicted_Survived)
  
  write_csv(df_gender_submission, "/data/gender_submission.csv")
  cat("gender_submission.csv updated!\n")
}

# Run main
main()