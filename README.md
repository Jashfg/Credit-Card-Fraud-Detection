# Credit-Card-Fraud-Detection
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
credit_card = pd.read_csv("C:\\Users\\VENKATA SAI JASWANTH\\Downloads\\creditcard.csv")
df = pd.DataFrame(credit_card)

# Define features and target variable
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
x = df[features]
y = df["Class"]

# Perform Random Oversampling
oversampler = RandomOverSampler()
x_res, y_res = oversampler.fit_resample(x, y)

# Print class distribution before and after oversampling
print(f'Before oversampling: {Counter(y)}')
print(f'After oversampling: {Counter(y_res)}')

# Split the data into training and testing sets with stratified sampling
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=20, stratify=y_res)

# Create a Decision Tree classifier and fit it to the training data
classifier = DecisionTreeClassifier(random_state=19)
classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(x_test)
print("Predicted labels (y_pred):", y_pred)

# Calculate and print the accuracy score
score = accuracy_score(y_pred, y_test)
print("Accuracy Score:", score)

# Calculate and print the confusion matrix
cm = confusion_matrix(y_pred, y_test)
print("Confusion Matrix:\n", cm)
