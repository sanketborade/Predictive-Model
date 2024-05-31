import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the scored data
data = pd.read_csv('scored_data.csv')

# Convert Anomaly_Label from -1 and 1 to 0 and 1
data['Anomaly_Label'] = data['Anomaly_Label'].replace({-1: 0, 1: 1})

# Separate features and target
X = data.drop(columns=['Anomaly_Label'])
y = data['Anomaly_Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a function to create pipelines
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Define hyperparameters grid for each model
param_grid = {
    'Logistic Regression': {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100], 'classifier__penalty': ['l1', 'l2']},
    'Decision Tree': {'classifier__max_depth': [None, 5, 10, 15, 20], 'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4]},
    'Random Forest': {'classifier__n_estimators': [100, 200, 300], 'classifier__max_depth': [None, 5, 10, 15], 'classifier__min_samples_split': [2, 5, 10], 'classifier__min_samples_leaf': [1, 2, 4]},
    'SVM': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']},
    'KNN': {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance'], 'classifier__metric': ['euclidean', 'manhattan']},
    'Gradient Boosting': {'classifier__n_estimators': [100, 200, 300], 'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7]},
    'XGBoost': {'classifier__n_estimators': [100, 200, 300], 'classifier__learning_rate': [0.1, 0.01, 0.001], 'classifier__max_depth': [3, 5, 7]}
}

# Initialize a dictionary to store the results after hyperparameter tuning
tuned_results = {}

# Loop through models and perform hyperparameter tuning
for model_name, model in models.items():
    pipeline = create_pipeline(model)
    grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    tuned_results[model_name] = {
        'best_model': best_model,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
    print(f"{model_name} Tuned Accuracy: {accuracy}")
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

# Display the accuracy after hyperparameter tuning for all models
for model_name, info in tuned_results.items():
    print(f"\n{model_name} Tuned Accuracy: {info['accuracy']}")
    print(pd.DataFrame(info['classification_report']).transpose())

