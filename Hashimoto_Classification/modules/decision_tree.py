from modules.load_dataframe import load_data

from sklearn.calibration import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def decision_tree_classification():
    # Load the dataset
    new_df = load_data()

    X = new_df.drop('target', axis=1)
    y = new_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Initialize the Decision Tree classifier
    model = DecisionTreeClassifier(random_state=42)

    # Train the Decision Tree model using the training data
    model.fit(X_train, y_train)

    # Predict the target values for the test set
    y_pred = model.predict(X_test)

    label_encoder = LabelEncoder()
    y_test_numeric = label_encoder.fit_transform(y_test)
    y_test_inverted = 1 - y_test_numeric 

    y_prob = model.predict_proba(X_test)[:, 0]  

    accuracy = accuracy_score(y_test, y_pred)

    with open("Scores.txt", "a") as file:
        file.write(f"Decision Tree Classifier Classification Report:\n{classification_report(y_test, y_pred)}\n")

    return y_test_inverted, y_prob, y_test, y_pred, accuracy

