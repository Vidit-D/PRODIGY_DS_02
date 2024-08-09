import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training and testing datasets
train_df = pd.read_csv('D:/vidit/prodigy infotech internship/task 2/train.csv')
test_df = pd.read_csv('D:/vidit/prodigy infotech internship/task 2/test.csv')

# Data Cleaning Steps (same as before)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
train_df.drop(columns=['Cabin'], inplace=True)
test_df.drop(columns=['Cabin'], inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
train_df.drop_duplicates(inplace=True)

# EDA: Correlation Heatmap (Fix)
# Select only numerical columns for correlation analysis
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Other EDA Visualizations (unchanged)
sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count')
plt.show()

sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Class')
plt.show()

sns.histplot(train_df['Age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

# Summary of clean data
print("\nCleaned Training Data Info:")
train_df.info()

# Save cleaned data (optional)
train_df.to_csv('D:/vidit/prodigy infotech internship/task 2/cleaned_train.csv', index=False)
test_df.to_csv('D:/vidit/prodigy infotech internship/task 2/cleaned_test.csv', index=False)
