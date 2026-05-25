import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

def roc_visualization(best_algorithm_results, best_algorithm):
    y_test_inverted,y_prob,_,_,_ = best_algorithm_results
    fpr, tpr, threshold = roc_curve(y_test_inverted, y_prob) 

    # Compute ROC curve values
    fpr, tpr , threshold= roc_curve(y_test_inverted, y_prob)

    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label="ROC Curve")

    # Plot random chance line (diagonal)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Labels and Title
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve for {best_algorithm}")
    plt.grid(True)

    # Show the plot
    plt.show()

def confusion_matrix_visualization(best_algorithm_results,best_algorithm):
    _,_,y_test,y_pred,_ = best_algorithm_results
    
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {best_algorithm}')

    plt.show()

def plot_multiple_roc_curves(models_results, models_names):
    plt.figure(figsize=(12, 8))
    
    for model_results, model_name in zip(models_results, models_names):
        y_test_inverted, y_prob, _, _, _ = model_results
        fpr, tpr, _ = roc_curve(y_test_inverted, y_prob)
        
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} ROC Curve")
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Comparison of ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.show()

def plot_model_accuracies(models_results, models_names):
    # Extract accuracies from the model results
    accuracies = np.array([result[4] for result in models_results])  # Accuracy stored at index 4

    # Define a color gradient
    colors = sns.color_palette("viridis", len(models_names))

    # Set Seaborn style
    sns.set_style("whitegrid")

    # Create figure
    plt.figure(figsize=(12, 9))
    bars = plt.bar(models_names, accuracies, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9)

    # Add labels and title
    plt.xlabel("Machine Learning Models", fontsize=14, fontweight='bold', color="#333")
    plt.ylabel("Accuracy Score", fontsize=14, fontweight='bold', color="#333")
    plt.title("Model Accuracy Comparison", fontsize=16, fontweight='bold', color="#222")
    
    plt.ylim(0, 1.1)  # Extend limit slightly for better readability
    plt.xticks(rotation=30, fontsize=12, ha="right")  # Rotate labels slightly

    # Add accuracy values on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f"{acc*100:.2f}%", ha='center', va='bottom', fontsize=12, 
                 fontweight='bold', color='white', bbox=dict(facecolor='black', edgecolor='none', boxstyle='round,pad=0.3'))

    # Add a light grid
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Show the plot
    plt.show()