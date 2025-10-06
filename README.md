# Binary Classification Pipeline with Fixed Cross-Validation

A robust, reproducible machine learning pipeline for binary classification tasks using stratified k-fold cross-validation with fixed splits.

---

## 📋 Overview

This pipeline provides a complete framework for evaluating multiple baseline classifiers on binary classification datasets. It emphasizes **reproducibility** through fixed cross-validation splits and **comprehensive evaluation metrics**.

### Key Features
- ✅ **Fixed CV Splits**: Ensures reproducibility across experiments  
- ✅ **Multiple Baselines**: 9 pre-configured classifiers (LR, SVM, RF, etc.)  
- ✅ **Comprehensive Metrics**: 11 evaluation metrics including AUC-ROC, F1, MCC  
- ✅ **Statistical Testing**: Pairwise Wilcoxon signed-rank tests  
- ✅ **Rich Visualizations**: Box plots, ROC curves, confidence intervals  
- ✅ **Multiple Export Formats**: CSV, JSON, Excel with metadata  

---

## 🗂️ Project Structure

```
├── notebooks/
│   ├── 01_prepare_splits_and_config.ipynb      # One-time split creation
│   └── 02_run_binary_classification_pipeline.ipynb  # Main evaluation
├── src/
│   └── fixed_cv_binary_classification.py       # Core pipeline code
├── data/
│   └── splits_k5_v1/
│       ├── manifest.json                       # Split metadata
│       ├── train_ids_fold*.csv                 # Training row IDs
│       ├── test_ids_fold*.csv                  # Test row IDs
│       └── heart_failure_..._with_row_id.csv   # Dataset + row_id
└── results_pipeline/
    ├── cv_results_per_fold.csv                 # Detailed results
    ├── cv_results_summary.csv                  # Aggregated statistics
    ├── cv_results_complete.xlsx                # Multi-sheet workbook
    ├── statistical_tests.csv                   # Pairwise comparisons
    └── figures/                                # Visualizations
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy
```

### Step 1: Prepare Fixed Splits (One-Time Setup)
Run `01_prepare_splits_and_config.ipynb` to:
1. Load your dataset  
2. Add stable `row_id` column  
3. Create stratified k-fold splits  
4. Save splits and manifest  

👉 Output: `data/splits_k5_v1/` with `manifest.json` and split files.

### Step 2: Run Evaluation Pipeline
```python
from fixed_cv_binary_classification import BinaryClassificationPipeline, Config

config = Config(
    splits_dir="data/splits_k5_v1/",
    dataset_csv="data/splits_k5_v1/heart_failure_..._with_row_id.csv",
    results_dir="results_pipeline",
    random_state=42
)

pipeline = BinaryClassificationPipeline(config)
results_df, summary_df = pipeline.run()
```

Or use the provided notebook: `02_run_binary_classification_pipeline.ipynb`

---

## 📊 Evaluated Models

| Model      | Description                  | Key Parameters |
|------------|------------------------------|----------------|
| LR         | Logistic Regression          | liblinear solver |
| SVM_linear | Linear SVM                   | class_weight='balanced' |
| SVM_rbf    | RBF kernel SVM               | gamma='scale' |
| kNN_k5     | k-Nearest Neighbors          | k=5, distance weights |
| DT         | Decision Tree                | max_depth=5 |
| RF         | Random Forest                | 100 trees, balanced |
| NB         | Gaussian Naive Bayes         | var_smoothing=1e-9 |
| GB         | Gradient Boosting            | 100 estimators, lr=0.1 |
| MLP        | Neural Network               | (100,50) layers |

---

## 📈 Evaluation Metrics

The pipeline computes **11 metrics**:

- Brier: Brier score (calibration)  
- BACC: Balanced accuracy  
- MCC: Matthews Correlation Coefficient  
- AUC_ROC: Area Under ROC Curve  
- AUC_PR: Area Under Precision-Recall Curve  
- SE: Sensitivity (Recall)  
- PPV: Positive Predictive Value (Precision)  
- F1: F1 Score  
- SP: Specificity  
- NPV: Negative Predictive Value  
- ACC: Accuracy  

---

## 🔧 Configuration Options

```python
@dataclass
class Config:
    splits_dir: Path = Path("../data/splits_k5_v1/")
    dataset_csv: Path = Path("../data/splits_k5_v1/heart_failure_..._with_row_id.csv")
    results_dir: Path = Path("../results_pipeline")
    
    metrics_list: List[str] = ["Brier", "BACC", "MCC", ...]
    random_state: int = 42
    n_jobs: int = -1  # Parallel processing
    
    fig_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
```

---

## 📦 Output Files

### Results
- `cv_results_per_fold.csv`: Fold-by-fold metrics  
- `cv_results_summary.csv`: Mean, std, median, CI  
- `cv_results_complete.xlsx`: Multi-sheet workbook with metadata  
- `statistical_tests.csv`: Wilcoxon test results  

### Visualizations
- `model_comparison_*.png`: Box plots per metric  
- `roc_curves.png`: ROC curves for all models  

### JSON
- `cv_results_per_fold.json`  
- `cv_results_summary.json`  

---

## 🔬 Reproducibility Guarantees
- **Fixed Splits**: Stable row IDs for identical sets  
- **Manifest Validation**: Automatic checks for integrity  
- **Random Seeds**: Consistent `random_state`  
- **Version Control**: Manifest tracks configuration  

---

## 📝 Example Results

```
================================================================================
TOP 5 MODELS BY AUC-ROC
================================================================================
     model      AUC_ROC_mean   AUC_ROC_std   F1_mean   MCC_mean
     RF         0.901355       0.026516      0.744379  0.639031
     GB         0.894490       0.012779      0.725710  0.603015
 SVM_linear     0.876203       0.033730      0.661672  0.542740
     LR         0.875163       0.031707      0.721311  0.607449
  SVM_rbf       0.860661       0.031572      0.678865  0.550067
================================================================================
```

---

## 🛠️ Advanced Usage

### Custom Models
```python
from sklearn.ensemble import RandomForestClassifier

custom_models = {
    "RF_tuned": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        random_state=42
    )
}
```

### Extending Metrics
```python
from sklearn.metrics import log_loss
metrics["LogLoss"] = log_loss(y_true, y_prob)
```

---

## ⚠️ Important Notes
- Dataset **must include `row_id`** column  
- Binary target only (2 classes)  
- Handle missing values beforehand  
- Large datasets may need high RAM  

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| FileNotFoundError: Manifest not found | Run notebook 01 first |
| ValueError: Train/test overlap | Regenerate splits |
| RuntimeWarning: AUC-ROC | Check class imbalance |
| Model crashes | Adjust `n_jobs` |

---

## 📚 Citation

```bibtex
@misc{binary_cv_pipeline_2025,
  title={Binary Classification Pipeline with Fixed Cross-Validation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/project}
}
```

Dataset reference:  
> Chicco, D., & Jurman, G. (2020). *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Medical Informatics and Decision Making, 20(1), 16.

---

## 📄 License
Specify your license (MIT, Apache 2.0, etc.)

---

## 🤝 Contributing
- Fork the repository  
- Create a feature branch  
- Add tests for new functionality  
- Submit a pull request  

---

## 📧 Contact
For questions or issues:  
**your.email@domain.com**

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Python**: 3.8+  
**Dependencies**: scikit-learn ≥ 1.0, pandas ≥ 1.3, numpy ≥ 1.21
