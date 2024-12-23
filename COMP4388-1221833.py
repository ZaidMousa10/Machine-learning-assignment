# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Load the dataset
data_path = 'C:/Users/user/Downloads/Customer Churn.csv'
data = pd.read_csv(data_path)

# Set pandas options to display all columns and rows without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# Exploratory Data Analysis (EDA)
print("Summary Statistics:")
print(data.describe())


print("\nClass Label Distribution (Churn):")
print(data['Churn'].value_counts())

data['Churn'].value_counts().plot(kind='bar', title='Distribution of Churn')
plt.xlabel('Churn')
plt.ylabel('Frequency')
plt.show()

# Histograms for Age Group and Charge Amount
sns.histplot(data=data, x='Age Group', hue='Churn', multiple='dodge', shrink=0.8)
plt.title('Churn by Age Group')
plt.show()

sns.histplot(data=data, x='Charge Amount', hue='Churn', multiple='dodge', shrink=0.8)
plt.title('Churn by Charge Amount')
plt.show()

print("Charge Amount Statistics:")
print(data['Charge Amount'].describe())

# Handle missing values explicitly for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')  # Replace missing values with the mean
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Handle missing values for non-numeric columns explicitly
data['Plan'] = data['Plan'].fillna('unknown')  # Replace missing values in 'Plan' with 'unknown'

# Check for missing values again
print("Number of missing values after handling:", data.isna().sum().sum())

# Convert binary attributes to numeric
data['Churn'] = data['Churn'].map({'yes': 1, 'no': 0})  # Convert Churn to binary
data['Plan'] = data['Plan'].map({'pre-paid': 0, 'post-paid': 1, 'unknown': -1})  # Convert Plan to binary or handle 'unknown'
data['Complains'] = data['Complains'].map({'yes': 1, 'no': 0})  # Example for binary encoding
data['Status'] = data['Status'].map({'active': 1, 'not-active': 0})  # Example for binary encoding

# Correlation Analysis
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Selecting Features based on Correlation
selected_features_all = ['ID', 'Call Failure', 'Complains', 'Charge Amount', 'Freq. of use', 'Freq. of SMS',
                         'Distinct Called Numbers', 'Age Group', 'Plan', 'Status', 'Age']
X_all = data[selected_features_all]
y = data['Customer Value']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.3, random_state=42)

# LRM1: Linear Regression using all features
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lrm1 = lr.predict(X_test)

print("\nPerformance of LRM1 (All Features):")
print("MSE:", mean_squared_error(y_test, y_pred_lrm1))
print("R2 Score:", r2_score(y_test, y_pred_lrm1))

# LRM2: Linear Regression using two selected features (Freq. of SMS and Freq. of use)
lr2 = LinearRegression()
lr2.fit(X_train[['Freq. of use', 'Freq. of SMS']], data['Customer Value'][X_train.index])
y_pred_lrm2 = lr2.predict(X_test[['Freq. of use', 'Freq. of SMS']])

print("\nPerformance of LRM2 (Two Features):")
print("MSE:", mean_squared_error(data['Customer Value'][X_test.index], y_pred_lrm2))
print("R2 Score:", r2_score(data['Customer Value'][X_test.index], y_pred_lrm2))

# LRM3: Linear Regression using top correlated features
top_features = ['Charge Amount', 'Freq. of use', 'Status', 'Age Group' , 'Freq. of SMS']  # Example based on correlation
lr3 = LinearRegression()
lr3.fit(X_train[top_features], data['Customer Value'][X_train.index])
y_pred_lrm3 = lr3.predict(X_test[top_features])

print("\nPerformance of LRM3 (Top Correlated Features):")
print("MSE:", mean_squared_error(data['Customer Value'][X_test.index], y_pred_lrm3))
print("R2 Score:", r2_score(data['Customer Value'][X_test.index], y_pred_lrm3))

# Visualization of Regression Performance
regression_results = pd.DataFrame({
    "Model": ["LRM1 (All Features)", "LRM2 (Two Features)", "LRM3 (Top Features)"],
    "MSE": [
        mean_squared_error(data['Customer Value'][X_test.index], y_pred_lrm1) ,
        mean_squared_error(data['Customer Value'][X_test.index], y_pred_lrm2),
        mean_squared_error(data['Customer Value'][X_test.index], y_pred_lrm3)
    ],
    "R2 Score": [
        r2_score(data['Customer Value'][X_test.index], y_pred_lrm1),
        r2_score(data['Customer Value'][X_test.index], y_pred_lrm2),
        r2_score(data['Customer Value'][X_test.index], y_pred_lrm3)
    ]
})


# Classification Tasks
# k-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(X_train, data['Churn'][X_train.index])
y_prob_knn = knn.predict_proba(X_test)[:, 1]

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, data['Churn'][X_train.index])
y_prob_nb = nb.predict_proba(X_test)[:, 1]

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, data['Churn'][X_train.index])
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# Logistic Regression
log_reg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42)
)
log_reg.fit(X_train, data['Churn'][X_train.index])
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

# Accuracy for Logistic Regression
y_pred_lr = log_reg.predict(X_test)
accuracy_lr = accuracy_score(data['Churn'][X_test.index], y_pred_lr)
r2_lr = r2_score(data['Churn'][X_test.index], y_pred_lr)

# Accuracy for Decision Tree
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(data['Churn'][X_test.index], y_pred_dt)
r2_dt = r2_score(data['Churn'][X_test.index], y_pred_dt)

# Accuracy for k-Nearest Neighbors
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(data['Churn'][X_test.index], y_pred_knn)
r2_knn = r2_score(data['Churn'][X_test.index], y_pred_knn)

# Accuracy for Naive Bayes
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(data['Churn'][X_test.index], y_pred_nb)
r2_nb = r2_score(data['Churn'][X_test.index], y_pred_nb)

# Print Results
print("\nClassification Model Performance:")
print("Logistic Regression: Accuracy =", accuracy_lr)
print("Decision Tree: Accuracy =", accuracy_dt)
print("k-Nearest Neighbors: Accuracy =", accuracy_knn)
print("Naive Bayes: Accuracy =", accuracy_nb)

# Plot ROC curves for classification models
plt.figure(figsize=(10, 8))

# Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(data['Churn'][X_test.index], y_prob_lr)
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")

# Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(data['Churn'][X_test.index], y_prob_dt)
plt.plot(fpr_dt, tpr_dt, label="Decision Tree")

# k-Nearest Neighbors
fpr_knn, tpr_knn, _ = roc_curve(data['Churn'][X_test.index], y_prob_knn)
plt.plot(fpr_knn, tpr_knn, label="k-Nearest Neighbors")

# Naive Bayes
fpr_nb, tpr_nb, _ = roc_curve(data['Churn'][X_test.index], y_prob_nb)
plt.plot(fpr_nb, tpr_nb, label="Naive Bayes")

plt.title("ROC Curves for Classification Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(data['Churn'][X_test.index], y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=["No Churn", "Churn"])
disp_lr.plot(cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Confusion Matrix for Decision Tree
cm_dt = confusion_matrix(data['Churn'][X_test.index], y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=["No Churn", "Churn"])
disp_dt.plot(cmap='Greens')
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Confusion Matrix for k-Nearest Neighbors
cm_knn = confusion_matrix(data['Churn'][X_test.index], y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["No Churn", "Churn"])
disp_knn.plot(cmap='Oranges')
plt.title("Confusion Matrix - k-Nearest Neighbors")
plt.show()

# Confusion Matrix for Naive Bayes
cm_nb = confusion_matrix(data['Churn'][X_test.index], y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=["No Churn", "Churn"])
disp_nb.plot(cmap='Purples')
plt.title("Confusion Matrix - Naive Bayes")
plt.show()