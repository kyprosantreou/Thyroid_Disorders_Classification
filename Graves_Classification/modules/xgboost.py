from modules.load_dataframe import load_data

import shap
from xgboost import XGBClassifier
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def xgboost_classification():
    new_df = load_data()

    label_encoder = LabelEncoder()
    new_df["target"] = label_encoder.fit_transform(new_df["target"])

   
    X = new_df.drop('target', axis=1)
    y = new_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgbc = XGBClassifier()

    xgbc.fit(X_train, y_train)

    # Prediction
    y_pred = xgbc.predict(X_test)

    label_encoder = LabelEncoder()
    y_test_numeric = label_encoder.fit_transform(y_test)
    y_test_inverted = 1 - y_test_numeric  

    y_prob = xgbc.predict_proba(X_test)[:, 0]  

    accuracy = accuracy_score(y_test, y_pred)

    with open("Scores.txt", "a") as file:
        file.write(f"XGBoost Classifier Classification Report:\n{classification_report(y_test, y_pred)}\n")

    return y_test_inverted, y_prob, y_test, y_pred,accuracy

