import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Logger:
    """
    Handles logging of training progress and metrics to Weights & Biases.
    """
    def __init__(self, config, model_name, project_name="latency-prediction", 
                 enable_logging=True, log_model=False):
        """
        Initialize the logger.
        
        Args:
            config: Dictionary of configuration parameters
            model_name: Name of the model being trained
            project_name: Name of the project in W&B
            enable_logging: Whether to enable logging to W&B
            log_model: Whether to log the model checkpoints
        """
        self.enable_logging = enable_logging
        self.log_model = log_model
        
        if not enable_logging:
            self.run = None
            return
            
        # Generate a unique run name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{model_name}_{timestamp}"
        
        # Initialize W&B
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            reinit=True
        )
    
    def log_hyperparams(self, hyperparams):
        """
        Log hyperparameters to W&B.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        if not self.enable_logging:
            return
            
        wandb.config.update(hyperparams)
    
    def log_metrics(self, metrics, step=None, prefix=""):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step (e.g., epoch)
            prefix: Optional prefix for metric names
        """
        if not self.enable_logging:
            return
            
        # Add prefix to metrics if provided
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics
            
        wandb.log(prefixed_metrics, step=step)
    
    def log_model_checkpoint(self, model, path, name=None):
        """
        Log model checkpoint to W&B.
        
        Args:
            model: Model to log
            path: Path where model is saved
            name: Optional name for the artifact
        """
        if not self.enable_logging or not self.log_model:
            return
            
        if name is None:
            name = f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
    
    def log_confusion_matrix(self, y_true, y_pred, class_names=None, title="Confusion Matrix"):
        """
        Log confusion matrix as an image to W&B.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Title for the plot
        """
        if not self.enable_logging:
            return
            
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        
        # Log to wandb
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def finish(self):
        """
        Finish the logging session.
        """
        if self.enable_logging and self.run is not None:
            wandb.finish() 