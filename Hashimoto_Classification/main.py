import os
from modules import *

def main():
    if os.path.exists("Scores.txt"):
        os.remove("Scores.txt")
    
    # Load Data
    new_df= load_data()
    print(f"Train Dataframe head:\n {new_df.head()}\n")
    print(f"Test Dataframe head:\n {new_df.head()}\n")

    # Print the counts of each class in the target column
    train_class_counts = new_df['target'].value_counts()
    test_class_counts = new_df['target'].value_counts()
    
    print(f"Train Class counts:\n{train_class_counts}\n")
    print(f"Test Class counts:\n{test_class_counts}\n")

    results = {}
    models_results = []  # To store all models' results
    models_names = []  # To store model names

    # Run classifiers and store their scores (assuming they return accuracy)
    xgboost_results = xgboost_classification()
    results['XGBOOST'] = xgboost_results
    models_results.append(xgboost_results)
    models_names.append("XGBoost")

    decision_tree_results = decision_tree_classification()
    results['Decision Tree'] = decision_tree_results
    models_results.append(decision_tree_results)
    models_names.append("Decision Tree")

    random_forest_results = random_forest_classification()
    results['Random Forest'] = random_forest_results
    models_results.append(random_forest_results)
    models_names.append("Random Forest")

    gradient_boosting_results = gradient_boosting_classification()
    results['Gradient Boosting'] = gradient_boosting_results
    models_results.append(gradient_boosting_results)
    models_names.append("Gradient Boosting")

    # Find the best classifier based on accuracy
    best_algorithm = max(results, key=lambda x: results[x][4])
    best_algorithm_results = results[best_algorithm]

    result = f"Best Algorithm: {best_algorithm} with Score: {(best_algorithm_results[4]*100):.2f}%"

    line_length = len(result)

    # Print the dynamically sized border and the message
    print("\n" + "=" * line_length)
    print(result)
    print("=" * line_length)

    # Visualizations
    roc_visualization(best_algorithm_results, best_algorithm)
    confusion_matrix_visualization(best_algorithm_results, best_algorithm)

    # Plot all ROC and accuracies curves for the models
    plot_multiple_roc_curves(models_results, models_names)
    plot_model_accuracies(models_results, models_names)


if __name__ == "__main__":
    main()
