import shap
from modules.load_dataframe import load_data

import shap
import matplotlib.pyplot as plt

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def random_forest_classification():
    # Load the dataset
    new_df = load_data()

    X = new_df.drop('target', axis=1)
    y = new_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=3) 

    rf.fit(X_train, y_train)

    # SHAP values for interpretability
    explainer = shap.Explainer(rf)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values[:,:,1], X_test, show=False)

    plt.title("Random Forest SHAP Summary Plot for Feature Importance for Grave's Disease", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Prediction
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # Encode categorical labels 
    label_encoder = LabelEncoder()
    y_test_numeric = label_encoder.fit_transform(y_test)
    y_test_inverted = 1 - y_test_numeric  

    y_prob = rf.predict_proba(X_test)[:, 0]  

    accuracy = accuracy_score(y_test, y_pred)

    with open("Scores.txt", "a") as file:
        file.write(f"Random Forest Classifier Classification Report:\n{classification_report(y_test, y_pred)}\n")

    return y_test_inverted, y_prob, y_test, y_pred,accuracy
