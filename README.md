# 🧪 Binary Classification Pipeline with Fixed Cross-Validation

A robust, reproducible machine learning pipeline for binary classification tasks using stratified k-fold cross-validation with fixed splits.

To enable all partners to work under identical and comparable conditions, a complete experimental environment has been defined. This environment relies on a single dataset, fixed cross-validation splits, a centralized configuration file, and a reference classification pipeline.
The objective is twofold:
- Ensure reproducibility of results, regardless of technical environments or preprocessing applied;
- Provide a common basis for comparison to assess the impact of the studied approaches (watermarking, anonymization, RTAB, etc.).
All these files allow each partner to reproduce the results and fairly compare the methods.

---

## 📋 Overview

This pipeline provides a complete framework for evaluating multiple baseline classifiers on binary classification datasets. It emphasizes **reproducibility** through fixed cross-validation splits and **comprehensive evaluation metrics**.

### Key Features
- ✅ **Fixed CV Splits**: Ensures reproducibility across experiments  
- ✅ **Multiple Baselines**: 9 pre-configured classifiers (LR, SVM, RF, etc.)  
- ✅ **Comprehensive Metrics**: 7 evaluation metrics including Brier Score, BACC, MCC  
- ✅ **Rich Visualizations**: Box plots, ROC curves, confidence intervals  
- ✅ **Multiple Export Formats**: CSV, JSON, Excel with metadata  

---

## 🗂️ Project Structure

```
├── notebooks/
│   ├── 01_prepare_splits_and_config.ipynb      # One-time split creation
│   └── 02_run_binary_classification_pipeline.ipynb  # Main evaluation
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
## 📊 Dataset

**Nom** : Heart Failure Clinical Records Dataset  
**Source** : [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)  
**Taille** : 299 patients (105 femmes, 194 hommes), âge 40–95 ans  
**Contexte** : patients suivis au Faisalabad Institute of Cardiology & Allied Hospital (Pakistan, 2015)  
**Cible** : `DEATH_EVENT` (0 = survie, 1 = décès)  

**Variables principales :**
- Données démographiques : `age`, `sex`, `smoking`  
- Facteurs cliniques : `anaemia`, `diabetes`, `high_blood_pressure`  
- Examens biologiques : `serum_creatinine`, `serum_sodium`, `platelets`, `creatinine_phosphokinase`  
- Mesure cardiaque clé : `ejection_fraction` (%)  
- Durée de suivi : `time` (jours)  

**Classes :**
- 203 survivants (≈ 68 %)  
- 96 décès (≈ 32 %)  
---
## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy
```

## Step 1: Prepare Fixed Splits (One-Time Setup) ✅ *Already Done*  

The preparation of fixed splits has **already been executed**.  
You **do not need to re-run** `01_prepare_splits_and_config.ipynb`.
This notebook was used once to:
1. Load the dataset  
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

In binary classification, relying on a single metric is insufficient to properly evaluate model performance. As highlighted by **Poiron et al. (2025)**, two main dimensions must be considered to ensure a comprehensive evaluation: **calibration** and **discrimination**.  

- **Calibration** refers to the agreement between the predicted probabilities and the actual distribution of the observed phenomenon. A well-calibrated model produces risk estimates that closely match observed event rates.  
- **Discrimination** corresponds to the model’s ability to correctly separate the two classes (e.g., healthy/sick individuals, event/no-event). It is generally measured using the confusion matrix, once a decision threshold has been applied to the predicted probabilities.  

In this framework:  
- Some metrics focus only on the **positive class** (e.g., sensitivity),  
- Others focus on the **negative class** (e.g., specificity),  
- And some are **global metrics** that simultaneously evaluate both classes (e.g., balanced accuracy – BACC).  

Moreover, performance can be measured using:  
- **Intrinsic metrics**, independent of prevalence and context (e.g., sensitivity, specificity, BACC),  
- **Prevalence-dependent metrics**, which reflect model performance in a specific operational setting (e.g., PPV, NPV, MCC).  

Based on these principles, and following the categorization proposed by **Poiron et al. (2025)**, we retained four main categories of metrics in our pipeline:  

1. Calibration  
2. Global discrimination  
3. Positive-class discrimination  
4. Negative-class discrimination  

Discrimination metrics were further subdivided according to whether they depend on prevalence. The table below summarizes these categories and the selected metrics.  

---

## 📋 Summary Table of Metrics

| **Category**              | **Subcategory**                   | **Metric** | **Definition** | **Formula** |
|----------------------------|-----------------------------------|-------------|----------------|-------------|
| **Calibration**            | Quantitative                      | Brier Score | Accuracy of predicted probabilities vs. observed outcomes. | $Brier = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$ |
|                            | Graphical                         | Calibration curves | Visual comparison of predicted vs. observed outcomes. | Graph |
| **Discrimination (positive)** | Prevalence-independent         | Sensitivity (SE) | Rate of positive samples correctly classified. | $SE = \frac{TP}{TP + FN}$ |
|                            | Prevalence-dependent              | PPV (Positive Predictive Value, Precision) | Ratio between correctly classified positive samples and all samples classified as positive. | $PPV = \frac{TP}{TP + FP}$ |
| **Discrimination (negative)** | Prevalence-independent         | Specificity (SP) | Rate of negative samples correctly classified. | $SP = \frac{TN}{TN + FP}$ |
|                            | Prevalence-dependent              | NPV (Negative Predictive Value) | Ratio between correctly classified negative samples and all samples classified as negative. | $NPV = \frac{TN}{TN + FN}$ |
| **Discrimination (global)**   | Prevalence-independent         | Balanced Accuracy (BACC) | Arithmetic average of sensitivity and specificity. | $BACC = \frac{SE + SP}{2}$ |
|                            | Prevalence-dependent              | MCC (Matthews Correlation Coefficient) | Correlation between predictions and ground truth. | $MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ |
 

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

### Visualizations
- `model_comparison_BACC.png`: BACC scores across models
- `model_comparison_MCC.png`: MCC scores across models

### JSON
- `cv_results_per_fold.json`  
- `cv_results_summary.json`  

---

## 📝 Baseline Results

```
================================================================================
TOP 5 MODELS BY BACC (Balanced Accuracy)
================================================================================
  model  Brier_mean  MCC_mean  BACC_mean  SE_mean  PPV_mean  SP_mean  NPV_mean
     RF    0.119412  0.639031   0.809618 0.707895  0.790476 0.911341  0.869802
     GB    0.136438  0.603015   0.797301 0.707895  0.745803 0.886707  0.865909
     LR    0.133686  0.614685   0.794172 0.677368  0.786093 0.910976  0.856903
     DT    0.154568  0.538719   0.769702 0.686842  0.683944 0.852561  0.854569
SVM_rbf    0.141723  0.550067   0.760802 0.625263  0.745873 0.896341  0.834717
================================================================================
```

<img width="1189" height="790" alt="model_comparison_BACC" src="https://github.com/user-attachments/assets/e7835f24-1e1f-4bbf-a9f6-eed35f61e195" />

---

