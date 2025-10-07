"""
Binary Classification with Fixed Cross-Validation Pipeline - 

This script provides a robust, modular pipeline for evaluating multiple baseline 
classifiers on the Heart Failure Clinical Records dataset using fixed stratified 
k-fold cross-validation splits.

"""

# ============================================================================
# IMPORTS AND CONFIGURATION
# ============================================================================

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, f1_score, recall_score,
    precision_score, balanced_accuracy_score, roc_auc_score,
    precision_recall_curve, auc, brier_score_loss, confusion_matrix,
    classification_report, roc_curve
)

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class Config:
    """Centralized configuration for the pipeline"""
    
    # Data paths
    splits_dir: Path = Path("../data/splits_k5_v1/")
    dataset_csv: Path = Path("../data/splits_k5_v1/heart_failure_clinical_records_dataset_with_row_id.csv")
    results_dir: Path = Path("../results_pipline")
    
    # Evaluation metrics
    metrics_list: List[str] = field(default_factory=lambda: [
        "Brier", "BACC", "MCC", 
        "SE", "PPV", "SP", "NPV"
    ])
    
    # Model parameters
    random_state: int = 42
    n_jobs: int = -1  # Use all available cores
    
    # Visualization settings
    fig_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

class DataLoader:
    """Handles data loading and validation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.manifest = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the dataset"""
        logger.info(f"Loading dataset from {self.config.dataset_csv}")
        
        if not self.config.dataset_csv.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_csv}")
        
        self.df = pd.read_csv(self.config.dataset_csv)
        
        # Validate required columns
        if "row_id" not in self.df.columns:
            raise ValueError("Dataset must include a 'row_id' column")
        
        # Check for missing values
        missing_summary = self.df.isnull().sum()
        if missing_summary.any():
            logger.warning(f"Missing values detected:\n{missing_summary[missing_summary > 0]}")
        
        logger.info(f"Dataset loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def load_manifest(self) -> Dict[str, Any]:
        """Load cross-validation manifest"""
        manifest_path = self.config.splits_dir / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        
        logger.info(f"Manifest loaded: {self.manifest.get('k', 5)}-fold CV, "
                   f"target: {self.manifest.get('target', 'DEATH_EVENT')}")
        return self.manifest
    
    def get_split_ids(self, fold_idx: int) -> Dict[str, pd.Series]:
        """Get train/test IDs for a specific fold"""
        train_path = self.config.splits_dir / f"train_ids_fold{fold_idx}.csv"
        test_path = self.config.splits_dir / f"test_ids_fold{fold_idx}.csv"
        
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Split files not found for fold {fold_idx}")
        
        train_ids = pd.read_csv(train_path)["row_id"]
        test_ids = pd.read_csv(test_path)["row_id"]
        
        # Validate no overlap
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise ValueError(f"Train/test overlap detected in fold {fold_idx}: {overlap}")
        
        return {"train": train_ids, "test": test_ids}

# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator:
    """Comprehensive metrics calculation with confidence intervals"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, 
                       y_prob: np.ndarray, 
                       y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        
        # Clip probabilities to avoid numerical issues
        y_prob = np.clip(y_prob, 1e-12, 1 - 1e-12)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics = {
            "Brier": brier_score_loss(y_true, y_prob),
            "BACC": balanced_accuracy_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "ACC": accuracy_score(y_true, y_pred),
            "SE": recall_score(y_true, y_pred, pos_label=1),  # Sensitivity
            "PPV": precision_score(y_true, y_pred, zero_division=0),  # Precision
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "SP": tn / (tn + fp) if (tn + fp) > 0 else np.nan,  # Specificity
            "NPV": tn / (tn + fn) if (tn + fn) > 0 else np.nan,  # NPV
        }
        
        # ROC-AUC with error handling
        try:
            metrics["AUC_ROC"] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {e}")
            metrics["AUC_ROC"] = np.nan
        
        # PR-AUC
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
        metrics["AUC_PR"] = auc(recall_vals, precision_vals)
        
        return metrics
    
    @staticmethod
    def compute_confidence_intervals(values: List[float], 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence intervals using bootstrap"""
        n_bootstrap = 1000
        bootstrapped_means = []
        
        values = np.array(values)
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrapped_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrapped_means, 100 * alpha / 2)
        upper = np.percentile(bootstrapped_means, 100 * (1 - alpha / 2))
        
        return lower, upper

# ============================================================================
# MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Creating and configuring models"""
    
    @staticmethod
    def get_models(random_state: int = 42, n_jobs: int = -1) -> Dict[str, Any]:
        """Create all baseline models with optimized parameters"""
        
        models = {
            "LR": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, 
                    random_state=random_state,
                    solver='liblinear'   
                ))
            ]),
            
            "SVM_linear": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="linear", 
                    probability=True, 
                    random_state=random_state,
                    class_weight='balanced'   
                ))
            ]),
            
            "SVM_rbf": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel="rbf", 
                    probability=True, 
                    random_state=random_state,
                    gamma='scale'
                ))
            ]),
            
            "kNN_k5": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance',  # 
                    n_jobs=n_jobs
                ))
            ]),
            
            "DT": DecisionTreeClassifier(
                random_state=random_state,
                max_depth=5,  # Prevent overfitting
                min_samples_split=10,
                min_samples_leaf=5
            ),
            
            "RF": RandomForestClassifier(
                n_estimators=100, 
                random_state=random_state,
                max_depth=10,
                min_samples_split=5,
                n_jobs=n_jobs,
                class_weight='balanced'
            ),
            
            "NB": GaussianNB(var_smoothing=1e-9),
            
            "GB": GradientBoostingClassifier(
                random_state=random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8
            ),
            
            "MLP": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    random_state=random_state, 
                    max_iter=1000,
                    hidden_layer_sizes=(100, 50),
                    early_stopping=True,
                    validation_fraction=0.1
                ))
            ])
        }
        
        return models

# ============================================================================
# CROSS-VALIDATION EVALUATOR
# ============================================================================

class CrossValidationEvaluator:
    """Main evaluation pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.metrics_calc = MetricsCalculator()
        self.results = []
        
    def run_evaluation(self) -> pd.DataFrame:
        """Run complete cross-validation evaluation"""
        
        # Load data and manifest
        df = self.data_loader.load_data()
        manifest = self.data_loader.load_manifest()
        
        target_col = manifest.get("target", "DEATH_EVENT")
        k = int(manifest.get("k", 5))
        
        # Prepare features and target
        X_all = df.drop(columns=[target_col])
        y_all = self._ensure_binary_labels(df[target_col])
        
        # Get models
        models = ModelFactory.get_models(
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        
        # Progress tracking
        total_iterations = k * len(models)
        pbar = tqdm(total=total_iterations, desc="Evaluating models")
        
        # Evaluate each model on each fold
        for fold in range(1, k + 1):
            fold_results = self._evaluate_fold(
                df, X_all, y_all, models, fold, pbar
            )
            self.results.extend(fold_results)
        
        pbar.close()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Add timestamp
        results_df['timestamp'] = pd.Timestamp.now()
        
        return results_df
    
    def _evaluate_fold(self, df: pd.DataFrame, 
                       X_all: pd.DataFrame, 
                       y_all: pd.Series,
                       models: Dict, 
                       fold: int, 
                       pbar: tqdm) -> List[Dict]:
        """Evaluate all models on a single fold"""
        
        fold_results = []
        
        # Get split IDs
        ids = self.data_loader.get_split_ids(fold)
        train_mask = df["row_id"].isin(ids["train"])
        test_mask = df["row_id"].isin(ids["test"])
        
        # Split data
        X_train = X_all.loc[train_mask].drop(columns=["row_id"], errors='ignore')
        y_train = y_all.loc[train_mask]
        X_test = X_all.loc[test_mask].drop(columns=["row_id"], errors='ignore')
        y_test = y_all.loc[test_mask]
        
        # Evaluate each model
        for model_name, model in models.items():
            start_time = time.time()
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Get predictions
                y_prob = self._get_probabilities(model, X_test)
                y_pred = (y_prob >= 0.5).astype(int)
                
                # Calculate metrics
                metrics = self.metrics_calc.compute_metrics(
                    y_test.values, y_prob, y_pred
                )
                
                # Store results
                result = {
                    "fold": fold,
                    "model": model_name,
                    "train_time": time.time() - start_time,
                    **metrics
                }
                fold_results.append(result)
                
            except Exception as e:
                logger.error(f"Error in fold {fold}, model {model_name}: {e}")
                # Add NaN results for failed model
                result = {
                    "fold": fold,
                    "model": model_name,
                    "train_time": np.nan,
                    **{m: np.nan for m in self.config.metrics_list}
                }
                fold_results.append(result)
            
            pbar.update(1)
        
        return fold_results
    
    def _get_probabilities(self, model, X_test: pd.DataFrame) -> np.ndarray:
        """Get probability predictions with fallback methods"""
        
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # Sigmoid transformation
            return 1 / (1 + np.exp(-scores))
        else:
            return model.predict(X_test).astype(float)
    
    def _ensure_binary_labels(self, y: pd.Series) -> pd.Series:
        """Ensure labels are binary (0, 1)"""
        unique_vals = sorted(y.unique())
        
        if set(unique_vals) <= {0, 1}:
            return y.astype(int)
        
        # Map to binary
        if len(unique_vals) != 2:
            raise ValueError(f"Target must be binary, found {len(unique_vals)} classes")
        
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        logger.info(f"Mapping labels: {mapping}")
        
        return y.map(mapping).astype(int)

# ============================================================================
# RESULTS ANALYZER
# ============================================================================

class ResultsAnalyzer:
    """Analyze and visualize results"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def create_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics with confidence intervals"""
        
        summary_stats = []
        
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            
            model_summary = {'model': model}
            
            for metric in self.config.metrics_list:
                values = model_data[metric].dropna().values
                
                if len(values) > 0:
                    model_summary[f"{metric}_mean"] = np.mean(values)
                    model_summary[f"{metric}_std"] = np.std(values)
                    model_summary[f"{metric}_median"] = np.median(values)
                    
                    # Confidence intervals
                    ci_lower, ci_upper = MetricsCalculator.compute_confidence_intervals(values)
                    model_summary[f"{metric}_ci_lower"] = ci_lower
                    model_summary[f"{metric}_ci_upper"] = ci_upper
                else:
                    for suffix in ['mean', 'std', 'median', 'ci_lower', 'ci_upper']:
                        model_summary[f"{metric}_{suffix}"] = np.nan
            
            summary_stats.append(model_summary)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Sort by primary metric (e.g., AUC-ROC)
        summary_df = summary_df.sort_values('AUC_ROC_mean', ascending=False)
        
        return summary_df
    
    def plot_model_comparison(self, results_df: pd.DataFrame, 
                             metric: str = "AUC_ROC"):
        """Create box plot comparing models"""
        
        plt.figure(figsize=self.config.fig_size)
        
        # Prepare data
        plot_data = results_df[['model', metric]].dropna()
        
        # Create box plot
        sns.boxplot(data=plot_data, x='model', y=metric)
        plt.xticks(rotation=45)
        plt.title(f'Model Comparison - {metric}')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.tight_layout()
        
        # Save figure
        save_path = self.config.results_dir / "figures" / f"model_comparison_{metric}.png"
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.show()
        
        logger.info(f"Figure saved: {save_path}")
    
    def plot_roc_curves(self, X_test, y_test, models):
        """Plot ROC curves for all models"""
        
        plt.figure(figsize=self.config.fig_size)
        
        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        save_path = self.config.results_dir / "figures" / "roc_curves.png"
        plt.savefig(save_path, dpi=self.config.dpi)
        plt.show()
    
    def perform_statistical_tests(self, results_df: pd.DataFrame, 
                                  metric: str = "AUC_ROC") -> pd.DataFrame:
        """Perform pairwise statistical tests between models"""
        
        models = results_df['model'].unique()
        test_results = []
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                scores1 = results_df[results_df['model'] == model1][metric].values
                scores2 = results_df[results_df['model'] == model2][metric].values
                
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(scores1, scores2)
                
                test_results.append({
                    'model1': model1,
                    'model2': model2,
                    'metric': metric,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        return pd.DataFrame(test_results)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class BinaryClassificationPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.evaluator = CrossValidationEvaluator(self.config)
        self.analyzer = ResultsAnalyzer(self.config)
        
    def run(self):
        """Execute complete pipeline"""
        
        logger.info("=" * 60)
        logger.info("Starting Binary Classification Pipeline")
        logger.info("=" * 60)
        
        # Run evaluation
        logger.info("Running cross-validation evaluation...")
        results_df = self.evaluator.run_evaluation()
        
        # Create summary
        logger.info("Creating summary statistics...")
        summary_df = self.analyzer.create_summary(results_df)
        
        # Save results
        self._save_results(results_df, summary_df)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        for metric in ['AUC_ROC', 'F1', 'MCC']:
            self.analyzer.plot_model_comparison(results_df, metric)
        
        # Statistical tests
        logger.info("Performing statistical tests...")
        stat_tests = self.analyzer.perform_statistical_tests(results_df)
        stat_tests.to_csv(
            self.config.results_dir / "statistical_tests.csv", 
            index=False
        )
        
        # Print summary
        self._print_summary(summary_df)
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
        return results_df, summary_df
    
    def _save_results(self, results_df: pd.DataFrame, summary_df: pd.DataFrame):
        """Save all results to files"""
        
        # CSV files
        results_df.to_csv(
            self.config.results_dir / "cv_results_per_fold.csv", 
            index=False
        )
        summary_df.to_csv(
            self.config.results_dir / "cv_results_summary.csv", 
            index=False
        )
        
        # JSON files for web integration
        results_df.to_json(
            self.config.results_dir / "cv_results_per_fold.json",
            orient="records", 
            indent=2
        )
        summary_df.to_json(
            self.config.results_dir / "cv_results_summary.json",
            orient="records", 
            indent=2
        )
        
        # Excel with multiple sheets
        with pd.ExcelWriter(self.config.results_dir / "cv_results_complete.xlsx") as writer:
            results_df.to_excel(writer, sheet_name='Per Fold Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Parameter': ['Date', 'N_folds', 'N_models', 'Dataset'],
                'Value': [
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    results_df['fold'].nunique(),
                    results_df['model'].nunique(),
                    str(self.config.dataset_csv)
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Results saved to {self.config.results_dir}")
    
    def _print_summary(self, summary_df: pd.DataFrame):
        """Print formatted summary to console"""
        
        print("\n" + "=" * 80)
        print("TOP 5 MODELS BY AUC-ROC")
        print("=" * 80)
        
        top_models = summary_df.nlargest(5, 'AUC_ROC_mean')[
            ['model', 'AUC_ROC_mean', 'AUC_ROC_std', 'F1_mean', 'MCC_mean']
        ]
        
        print(top_models.to_string(index=False))
        print("=" * 80)

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = BinaryClassificationPipeline()
    results, summary = pipeline.run()
    
    print("\nPipeline execution completed successfully!")
    print(f"Results saved in: {pipeline.config.results_dir}")
