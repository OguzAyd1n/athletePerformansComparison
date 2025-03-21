import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Optional

class Visualizer:
    @staticmethod
    def plot_correlation(df, save_path: Optional[str] = None) -> None:
        """
        Plot correlation heatmap of features.
        
        Args:
            df: DataFrame containing features
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_distribution(df, column: str, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of a specific column.
        
        Args:
            df: DataFrame containing the data
            column: Column name to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], bins=20, kde=True)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_predictions(y_test: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot actual vs predicted values.
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Values - {model_name}')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_roc_curve(y_test: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curve for binary classification.
        
        Args:
            y_test: Actual values
            y_pred: Predicted probabilities
            save_path: Optional path to save the plot
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_error_distribution(y_test: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot distribution of prediction errors.
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        errors = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=20, kde=True)
        plt.title(f'Error Distribution - {model_name}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(save_path)
        plt.show() 