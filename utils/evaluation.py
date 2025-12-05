"""
Evaluation metrics and visualization utilities for emotion detection models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd

from .constants import EMOTION_CLASSES, INDEX_TO_EMOTION


class EmotionEvaluator:
    """
    Comprehensive evaluation utilities for emotion detection models.
    """
    
    def __init__(self, emotion_classes: List[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            emotion_classes: List of emotion class names
        """
        self.emotion_classes = emotion_classes or EMOTION_CLASSES
        self.num_classes = len(self.emotion_classes)
    
    def compute_metrics(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       y_prob: np.ndarray = None,
                       average: str = 'weighted') -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (class indices)
            y_pred: Predicted labels (class indices)
            y_prob: Predicted probabilities (optional)
            average: Averaging strategy for multi-class metrics
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, emotion in enumerate(self.emotion_classes):
            if i < len(precision_per_class):
                metrics[f'precision_{emotion}'] = precision_per_class[i]
                metrics[f'recall_{emotion}'] = recall_per_class[i]
                metrics[f'f1_{emotion}'] = f1_per_class[i]
        
        # ROC AUC if probabilities are provided
        if y_prob is not None:
            try:
                # Binarize labels for multi-class ROC
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                
                # Compute ROC AUC for each class
                roc_auc_per_class = []
                for i in range(self.num_classes):
                    if i < y_prob.shape[1] and len(np.unique(y_true_bin[:, i])) > 1:
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                        roc_auc = auc(fpr, tpr)
                        roc_auc_per_class.append(roc_auc)
                        metrics[f'roc_auc_{self.emotion_classes[i]}'] = roc_auc
                    else:
                        roc_auc_per_class.append(0.0)
                
                # Macro and micro average ROC AUC
                metrics['roc_auc_macro'] = np.mean(roc_auc_per_class)
                
                # Average precision (AP) scores
                ap_per_class = []
                for i in range(self.num_classes):
                    if i < y_prob.shape[1] and len(np.unique(y_true_bin[:, i])) > 1:
                        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                        ap_per_class.append(ap)
                        metrics[f'ap_{self.emotion_classes[i]}'] = ap
                    else:
                        ap_per_class.append(0.0)
                
                metrics['map'] = np.mean(ap_per_class)  # Mean Average Precision
                
            except Exception as e:
                print(f"Warning: Could not compute ROC/AP metrics: {e}")
        
        return metrics
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            normalize: bool = True,
                            title: str = "Confusion Matrix",
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrix with emotion class labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the confusion matrix
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, 
                   annot=True, 
                   fmt=fmt, 
                   cmap='Blues',
                   xticklabels=self.emotion_classes,
                   yticklabels=self.emotion_classes,
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self, 
                       y_true: np.ndarray, 
                       y_prob: np.ndarray,
                       title: str = "ROC Curves",
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for each emotion class.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        
        for i, (emotion, color) in enumerate(zip(self.emotion_classes, colors)):
            if i < y_prob.shape[1] and len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{emotion} (AUC = {roc_auc:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curves(self, 
                                   y_true: np.ndarray, 
                                   y_prob: np.ndarray,
                                   title: str = "Precision-Recall Curves",
                                   figsize: Tuple[int, int] = (12, 8),
                                   save_path: str = None) -> plt.Figure:
        """
        Plot Precision-Recall curves for each emotion class.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        
        for i, (emotion, color) in enumerate(zip(self.emotion_classes, colors)):
            if i < y_prob.shape[1] and len(np.unique(y_true_bin[:, i])) > 1:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                
                ax.plot(recall, precision, color=color, lw=2,
                       label=f'{emotion} (AP = {ap:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     output_dict: bool = False) -> str or Dict:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Whether to return as dictionary
            
        Returns:
            Classification report as string or dictionary
        """
        return classification_report(
            y_true, y_pred, 
            target_names=self.emotion_classes,
            output_dict=output_dict,
            zero_division=0
        )
    
    def plot_training_history(self, 
                            history: Dict[str, List[float]],
                            metrics: List[str] = None,
                            title: str = "Training History",
                            figsize: Tuple[int, int] = (15, 5),
                            save_path: str = None) -> plt.Figure:
        """
        Plot training history metrics.
        
        Args:
            history: Training history dictionary
            metrics: List of metrics to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            # Default metrics to plot
            available_metrics = list(history.keys())
            metrics = [m for m in ['accuracy', 'loss', 'precision', 'recall'] 
                      if m in available_metrics]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot training metric
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}')
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                epochs = range(1, len(history[val_metric]) + 1)
                ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}')
            
            ax.set_title(f'{metric.capitalize()} History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ModelComparator:
    """
    Utility class for comparing multiple emotion detection models.
    """
    
    def __init__(self, emotion_classes: List[str] = None):
        """
        Initialize the model comparator.
        
        Args:
            emotion_classes: List of emotion class names
        """
        self.emotion_classes = emotion_classes or EMOTION_CLASSES
        self.evaluator = EmotionEvaluator(emotion_classes)
        self.results = {}
    
    def add_model_results(self, 
                         model_name: str,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: np.ndarray = None) -> None:
        """
        Add evaluation results for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
        """
        metrics = self.evaluator.compute_metrics(y_true, y_pred, y_prob)
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def compare_models(self, 
                      metrics: List[str] = None,
                      sort_by: str = 'accuracy') -> pd.DataFrame:
        """
        Compare models across specified metrics.
        
        Args:
            metrics: List of metrics to compare
            sort_by: Metric to sort results by
            
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            raise ValueError("No model results added yet")
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in self.results.items():
            row = {'model': model_name}
            for metric in metrics:
                if metric in results['metrics']:
                    row[metric] = results['metrics'][metric]
                else:
                    row[metric] = None
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=False)
        
        return df
    
    def plot_model_comparison(self, 
                            metrics: List[str] = None,
                            title: str = "Model Comparison",
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: str = None) -> plt.Figure:
        """
        Plot comparison of models across metrics.
        
        Args:
            metrics: List of metrics to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        df = self.compare_models(metrics)
        
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Filter metrics that exist in the dataframe
        available_metrics = [m for m in metrics if m in df.columns]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(df))
        width = 0.8 / len(available_metrics)
        
        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics)/2 + 0.5) * width
            ax.bar(x + offset, df[metric], width, label=metric.capitalize())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df['model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def statistical_significance_test(self, 
                                    model1: str, 
                                    model2: str,
                                    metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform statistical significance test between two models.
        
        Args:
            model1: Name of first model
            model2: Name of second model
            metric: Metric to test
            
        Returns:
            Dictionary with test results
        """
        from scipy import stats
        
        if model1 not in self.results or model2 not in self.results:
            raise ValueError("Both models must be added to results first")
        
        # For now, we'll use a simple paired t-test on predictions
        # In practice, you might want to use more sophisticated tests
        y_true = self.results[model1]['y_true']
        pred1 = self.results[model1]['y_pred']
        pred2 = self.results[model2]['y_pred']
        
        # Calculate per-sample accuracy
        acc1 = (pred1 == y_true).astype(float)
        acc2 = (pred2 == y_true).astype(float)
        
        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(acc1, acc2)
        
        return {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'model1_mean': np.mean(acc1),
            'model2_mean': np.mean(acc2)
        }