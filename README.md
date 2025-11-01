# MLDS400-HW3
This is my MLDS400 HW3 project on linear regression

This project will perform linear regression in Python and R

To run the code in Python, do the following:
1. Download the Titanic Kaggle Dataset from https://www.kaggle.com/competitions/titanic/data
2. Create the /data folder under src (src/data/)
3. Place train.csv and test.csv in the src/data/ folder

```
The folder structure should look like
├── explore_files
├── src
│   ├── data
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── scripts_python
│   ├── scripts_r
├── README.md
├── run_python_docker.sh
└── run_r_docker.sh
```

4. In the main directory of this repository, run "bash run_python_docker.sh" (The same level as where the .sh file is)
5. This should create the docker image, run the container, and provide the output!
6. The output should be an updated gender_submission.csv with the Logistic Regression Model's predictions

To run the code in R, do the following:
1. Download the Titanic Kaggle Dataset from https://www.kaggle.com/competitions/titanic/data
2. Create the /data folder under src (src/data/)
3. Place train.csv and test.csv in the src/data/ folder

```
The folder structure should look like
├── explore_files
├── src
│   ├── data
│   │   ├── train.csv
│   │   ├── test.csv
│   ├── scripts_python
│   ├── scripts_r
├── README.md
├── run_python_docker.sh
└── run_r_docker.sh
```

4. In the main directory of this repository, run "bash run_r_docker.sh" (The same level as where the .sh file is)
5. This should create the docker image, run the container, and provide the output!
6. The output should be an updated gender_submission.csv with the Logistic Regression Model's predictions

Don't worry if gender_submission.csv is in src/data/ ! Any docker runs will overwrite the csv file.

If you already have the image and want to run the image again:
1. cd to src
2. In src/, run this command: "docker run -v "$(pwd)/data:/data" pythonapp" for Python
3. Or "docker run -v "$(pwd)/data:/data" rapp" for R

To see the analysis below in a ipynb environment, check src/scripts_python/explore.ipynb


# Exploratory Analysis and Data Cleaning 


```python
# First import modules that are relevant

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Read the training data!
df_train = pd.read_csv("../data/train.csv")
```


```python
# And check what it looks like
df_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### From the dataset, we know that the variables mean 
survival:	Survival	0 = No, 1 = Yes

pclass:	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

sex:	Sex	

Age:	Age in years	

sibsp:	# of siblings / spouses aboard the Titanic	

parch:	# of parents / children aboard the Titanic	

ticket:	Ticket number	

fare:	Passenger fare	

cabin:	Cabin number	

embarked:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


```python
# We can then check the data types for each column
df_train.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object




```python
# We can drop the PassengerId, Name, and Ticket Number columns for our analysis as they don't have any numerical value in determining whether a passenger survived or not
df_train_pruned = df_train.drop(columns = ['PassengerId', 'Ticket', 'Name'])
```


```python
# Next we can check what columns have null values
df_train_pruned.isnull().sum()
```




    Survived      0
    Pclass        0
    Sex           0
    Age         177
    SibSp         0
    Parch         0
    Fare          0
    Cabin       687
    Embarked      2
    dtype: int64




```python
# Let's look at Age
plt.hist(df_train_pruned['Age'], bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Passenger Age')
plt.show()
```


    
![png](explore_files/explore_8_0.png)
    



```python
# It does look somewhat skewed, let's check the mean and median
print(f"Mean Age: {float(df_train_pruned['Age'].mean())}, Median Age: {float(df_train_pruned['Age'].median())}") 
```

    Mean Age: 29.69911764705882, Median Age: 28.0


Although there is not much difference, we can put 28 as a substitution for null values


```python
df_train_pruned['Age'].fillna(df_train_pruned['Age'].median(), inplace=True)
```

    /var/folders/t_/_ffs2h250g193ttskmc3188r0000gn/T/ipykernel_23392/3354587220.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df_train_pruned['Age'].fillna(df_train_pruned['Age'].median(), inplace=True)



```python
# Let's look at Cabin
df_train_pruned['Cabin'].isnull().sum() / len(df_train_pruned['Cabin']) * 100.0
```




    np.float64(77.10437710437711)



We can see that 77% of the Cabin column is null, so we should not use this column at all


```python
# Drop the Cabin column
df_train_pruned = df_train_pruned.drop(columns=['Cabin'])
```


```python
# Let's look at Embarked
embarked_counts = df_train_pruned['Embarked'].value_counts()
plt.bar(embarked_counts.index, embarked_counts.values)
plt.title("Count of Embarked Values")
plt.ylabel("Count")
plt.show()
```


    
![png](explore_files/explore_15_0.png)
    


It seems like more people have the value S. Therefore, we can use S to fill the null values


```python
df_train_pruned['Embarked'].fillna('S', inplace=True)
```

    /var/folders/t_/_ffs2h250g193ttskmc3188r0000gn/T/ipykernel_23392/219461157.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df_train_pruned['Embarked'].fillna('S', inplace=True)



```python
# Updated data set
df_train_pruned.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can also create dummy variables for Sex and Embarked as they are categorial variables
df_train_dummies = pd.get_dummies(df_train_pruned, columns=['Sex', 'Embarked'])
df_train_dummies.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can actually drop Sex_female and Embarked_C as if the other columns are all False (i.e. Sex_male is False = they are female), then we know it's the control variable
df_train_dummies = df_train_dummies.drop(columns=['Sex_female', 'Embarked_C'])
```


```python
# Now that we have our columns, we can look for any data trends
df_survived = df_train_dummies[df_train_dummies['Survived'] == 1]
df_did_not_survive = df_train_dummies[df_train_dummies['Survived'] == 0]
```


```python
# Plot stacked bar charts for 
survived_gender_counts = df_survived['Sex_male'].value_counts()
did_not_survive_gender_counts = df_did_not_survive['Sex_male'].value_counts()

# Combine into a DataFrame where index = survival status
df_plot = pd.DataFrame({
    'Male': [survived_gender_counts.iloc[1], did_not_survive_gender_counts.iloc[0]],
    'Female': [survived_gender_counts.iloc[0], did_not_survive_gender_counts.iloc[1]]
}, index=['Survived', 'Did Not Survive'])

# Plot stacked bar chart
df_plot.plot(
    kind='bar',
    stacked=True,
    color=['#2196F3', '#E91E63'],  # blue = male, pink = female
    figsize=(6, 4)
)

plt.title('Passenger Survival by Gender')
plt.xlabel('Survival Status')
plt.ylabel('Number of Passengers')
plt.legend(title='Gender')
plt.tight_layout()
plt.show()
```


    
![png](explore_files/explore_22_0.png)
    


We can see that the males had a higher rate of not surviving, which is reasonable as they are more able to help with moving everyone to safety first


```python
# Plot the two distrbutions for Age
plt.figure(figsize=(8, 6))

plt.hist(df_survived['Age'], bins=20, alpha=0.6, color='blue', label='Survived', edgecolor='black')
plt.hist(df_did_not_survive['Age'], bins=20, alpha=0.6, color='red', label='Did Not Survive', edgecolor='black')

plt.title("Distribution Comparisons For Those Survived And Those Who Did Not")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()
```


    
![png](explore_files/explore_24_0.png)
    


We can see from this plot that they have a somewhat equal distribution of being skewed, with the distribution of those that did not survive being slightly older. There is also a huge number of casualties of those in their late 20s, which might be strong male who helped others survive.


```python
# Check for class
survived_class_counts = df_survived['Pclass'].value_counts()
did_not_survive_class_counts = df_did_not_survive['Pclass'].value_counts()

# Plot the two bar plots side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

axes[0].bar(survived_class_counts.index, survived_class_counts.values, color='blue')
axes[0].set_title("Count of Embarked Values For Survived")
axes[0].set_xticks([1, 2, 3])
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")

axes[1].bar(did_not_survive_class_counts.index, did_not_survive_class_counts.values, color='red')
axes[1].set_title("Count of Embarked Values For Not Survived")
axes[1].set_xticks([1, 2, 3])
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")

plt.show()

```


    
![png](explore_files/explore_26_0.png)
    


We can see that for those that did survive, there is not an outright pattern in how class mattered. However, for those who didn't survive, the 3rd class passengers were the most affected.


```python
# We can check for the fare prices
plt.figure(figsize=(8, 6))

# Make the bins the same width
same_bins = np.histogram(np.hstack((df_survived['Fare'],df_did_not_survive['Fare'])), bins=40)[1]

plt.hist(df_survived['Fare'], bins=same_bins, alpha=0.6, color='blue', label='Survived', edgecolor='black')
plt.hist(df_did_not_survive['Fare'], bins=same_bins, alpha=0.6, color='red', label='Did Not Survive', edgecolor='black')

plt.title("Distribution Comparisons For Those Survived And Those Who Did Not")
plt.xlabel("Fare Price")
plt.ylabel("Count")
plt.legend()
plt.show()
```


    
![png](explore_files/explore_28_0.png)
    


In the histogram, there is no clear trend in the fare prices mattering to the survivability of the passenger, but we can see that there are certain higher fare prices (> $50) where there are more survivability

SibSp and Parch might introduce multicollineary as they both are 0 if the traveler is alone. Therefore, we can just create a new variable to replace these two to check whether the traveler was alone


```python
# Create the new Alone column
df_train_dummies['Alone'] = np.where((df_train_dummies['SibSp']+df_train_dummies['Parch'])>0, 0, 1)
# Drop the unncessary columns
df_train_dummies = df_train_dummies.drop(columns=['SibSp', 'Parch'])
```


```python
# We get our final model
df_train_dummies.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Sex_male</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>8.0500</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's make a method that will transform this data for any new data with the same format

def clean_data(input_data_location):
    '''Given the path to a csv file, read it and transform the data for logistic regression'''

    # Read the intial csv file
    df = pd.read_csv(input_data_location)
    # Drop the unncessary columns
    df = df.drop(columns = ['PassengerId', 'Ticket', 'Name', 'Cabin'])
    # Fill the null values
    df.fillna({'Age': df['Age'].median()}, inplace=True)
    df.fillna({'Embarked': 'S'}, inplace=True)

    # Added after test set had null Fare value
    df.fillna({'Fare': df['Fare'].median()}, inplace=True)

    # Standardize gender values to all lower case 
    df['Sex'] = df['Sex'].str.lower()

    # Get dummy variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    # Drop the redundant variable
    df = df.drop(columns=['Sex_female', 'Embarked_C'])
    # Create the new Alone column
    df['Alone'] = np.where((df['SibSp']+df['Parch'])>0, 0, 1)
    # Drop the unncessary columns
    df = df.drop(columns=['SibSp', 'Parch'])

    # Return the final dataframe
    return df
    
```


```python
# Run the clean_data() method
df_train_method = clean_data("../data/train.csv")
df_test_method = clean_data("../data/test.csv")
```


```python
# Let's do a final check on the test file to see if there are missing values
df_test_method.isnull().sum()
```




    PassengerId    0
    Pclass         0
    Age            0
    Fare           0
    Sex_male       0
    Embarked_Q     0
    Embarked_S     0
    Alone          0
    dtype: int64




```python
# There is a missing Fare row, let's fill it with the median as we saw that the data is skewed
df_test_method.fillna({'Fare': df_test_method['Fare'].median()}, inplace=True)
```


```python
# After updating the method, let's run it again
df_train_method = clean_data("../data/train.csv")
df_test_method = clean_data("../data/test.csv")
```

### Now let's run Logistic Regression on the model


```python
# Import the logistic regression library from sklearn
from sklearn.linear_model import LogisticRegression
```


```python
# Divide the data by the training and test set 
X_train = df_train_method[['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Alone']]
y_train = df_train_method['Survived']

X_test = df_test_method[['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'Alone']]
```


```python
# Use the logistic regression model to fit the training data
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```



```python
# And predict the test data!
y_pred = model.predict(X_test)
```


```python
# Add the column of prediction to the test dataset
df_test_method['Predicted Survived'] = y_pred
```


```python
# Calculate the percentage of those who have survived
float(np.round(df_test_method['Predicted Survived'].value_counts()[1] / len(df_test_method) * 100.0, 2))
```




    38.28



38.28% of people in the test data is predicted to have survived...
