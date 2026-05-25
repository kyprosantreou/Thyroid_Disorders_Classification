from modules.load_dataframe import load_data 

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler  # Corrected import
from sklearn.model_selection import train_test_split

def gradient_boosting_classification():
    new_df = load_data()
    
    label_encoder = LabelEncoder()
    new_df["target"] = label_encoder.fit_transform(new_df["target"])

    imputer = SimpleImputer(strategy='mean') 
    X = new_df.drop('target', axis=1)
    y = new_df['target']
    
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=30)    
    gb.fit(X_train, y_train)

    # Predictions
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:, 0]  

    # Encode categorical labels
    y_test_numeric = label_encoder.fit_transform(y_test)
    y_test_inverted = 1 - y_test_numeric 


    accuracy = accuracy_score(y_test, y_pred)

    # Save results to a file
    with open("Scores.txt", "a") as file:
        file.write(f"Gradient Boosting Classifier Classification Report:\n{classification_report(y_test, y_pred)}\n")

    return y_test_inverted, y_prob, y_test, y_pred, accuracy
