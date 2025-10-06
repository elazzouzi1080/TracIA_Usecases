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

├── notebooks/
│ ├── 01_prepare_splits_and_config.ipynb # One-time split creation
│ └── 02_run_binary_classification_pipeline.ipynb # Main evaluation
├── src/
│ └── fixed_cv_binary_classification.py # Core pipeline code
├── data/
│ └── splits_k5_v1/
│ ├── manifest.json # Split metadata
│ ├── train_ids_fold*.csv # Training row IDs
│ ├── test_ids_fold*.csv # Test row IDs
│ └── heart_failure_..._with_row_id.csv # Dataset + row_id
└── results_pipeline/
├── cv_results_per_fold.csv # Detailed results
├── cv_results_summary.csv # Aggregated statistics
├── cv_results_complete.xlsx # Multi-sheet workbook
├── statistical_tests.csv # Pairwise comparisons
└── figures/ # Visualizations

yaml
Copier le code

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy
Step 1: Prepare Fixed Splits (One-Time Setup)
Run 01_prepare_splits_and_config.ipynb to:

Load your dataset

Add stable row_id column

Create stratified k-fold splits

Save splits and manifest

👉 Output: data/splits_k5_v1/ with manifest.json and split files.

Step 2: Run Evaluation Pipeline
python
Copier le code
from fixed_cv_binary_classification import BinaryClassificationPipeline, Config

config = Config(
    splits_dir="data/splits_k5_v1/",
    dataset_csv="data/splits_k5_v1/heart_failure_..._with_row_id.csv",
    results_dir="results_pipeline",
    random_state=42
)

pipeline = BinaryClassificationPipeline(config)
results_df, summary_df = pipeline.run()
Or use the provided notebook: 02_run_binary_classification_pipeline.ipynb

📊 Evaluated Models
Model	Description	Key Parameters
LR	Logistic Regression	liblinear solver
SVM_linear	Linear SVM	class_weight='balanced'
SVM_rbf	RBF kernel SVM	gamma='scale'
kNN_k5	k-Nearest Neighbors	k=5, distance weights
DT	Decision Tree	max_depth=5
RF	Random Forest	100 trees, balanced
NB	Gaussian Naive Bayes	var_smoothing=1e-9
GB	Gradient Boosting	100 estimators, lr=0.1
MLP	Neural Network	(100,50) layers

📈 Evaluation Metrics
The pipeline computes 11 metrics:

Brier: Brier score (calibration)

BACC: Balanced accuracy

MCC: Matthews Correlation Coefficient

AUC_ROC: Area Under ROC Curve

AUC_PR: Area Under Precision-Recall Curve

SE: Sensitivity (Recall)

PPV: Positive Predictive Value (Precision)

F1: F1 Score

SP: Specificity

NPV: Negative Predictive Value

ACC: Accuracy

🔧 Configuration Options
python
Copier le code
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
📦 Output Files
Results
cv_results_per_fold.csv: Fold-by-fold metrics

cv_results_summary.csv: Mean, std, median, CI

cv_results_complete.xlsx: Multi-sheet workbook with metadata

statistical_tests.csv: Wilcoxon test results

Visualizations
model_comparison_*.png: Box plots per metric

roc_curves.png: ROC curves for all models

JSON
cv_results_per_fold.json

cv_results_summary.json

🔬 Reproducibility Guarantees
Fixed Splits: Stable row IDs for identical sets

Manifest Validation: Automatic checks for integrity

Random Seeds: Consistent random_state

Version Control: Manifest tracks configuration

📝 Example Results
markdown
Copier le code
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
🛠️ Advanced Usage
Custom Models
python
Copier le code
from sklearn.ensemble import RandomForestClassifier

custom_models = {
    "RF_tuned": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        random_state=42
    )
}
Extending Metrics
python
Copier le code
from sklearn.metrics import log_loss
metrics["LogLoss"] = log_loss(y_true, y_prob)
⚠️ Important Notes
Dataset must include row_id column

Binary target only (2 classes)

Handle missing values beforehand

Large datasets may need high RAM

🐛 Troubleshooting
Issue	Solution
FileNotFoundError: Manifest not found	Run notebook 01 first
ValueError: Train/test overlap	Regenerate splits
RuntimeWarning: AUC-ROC	Check class imbalance
Model crashes	Adjust n_jobs

📚 Citation
bibtex
Copier le code
@misc{binary_cv_pipeline_2025,
  title={Binary Classification Pipeline with Fixed Cross-Validation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/project}
}
Dataset reference:

Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making, 20(1), 16.

📄 License
Specify your license (MIT, Apache 2.0, etc.)

🤝 Contributing
Fork the repository

Create a feature branch

Add tests for new functionality

Submit a pull request

📧 Contact
For questions or issues:
your.email@domain.com

Version: 1.0.0
Last Updated: October 2025
Python: 3.8+
Dependencies: scikit-learn ≥ 1.0, pandas ≥ 1.3, numpy ≥ 1.21

---

## 📊 Étape 2. Dataset utilisé

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

> ⚠️ Dataset petit, monocentrique et déséquilibré, mais idéal pour démonstration et benchmark.

---

## 📒 Étape 3. Notebook 1 — Préparation des splits et configuration
**Fichier** : `01_prepare_splits_and_config.ipynb`  

Ce notebook permet de :
- Charger le dataset brut  
- Ajouter un identifiant unique `row_id`  
- Générer des **splits fixes stratifiés (k=5)** avec graine aléatoire déterministe  
- Sauvegarder les fichiers nécessaires dans `data/splits_k5_v1/` :  
  - `manifest.json` (métadonnées : cible, features, taille des splits, etc.)  
  - `train_ids_fold*.csv` et `test_ids_fold*.csv`  

> Ces splits seront réutilisés pour tous les entraînements afin d’assurer la comparabilité des résultats.

---

## 🖥️ Étape 4. Script — Exécution du pipeline complet
**Fichier** : `fixed_cv_binary_classification.py`  

Ce script exécute automatiquement :
1. Chargement du dataset + splits fixes  
2. Entraînement de plusieurs modèles classiques (LogReg, SVM, RF, MLP…)  
3. Évaluation sur chaque fold avec un ensemble de métriques :  
   - AUC-ROC, AUC-PR, MCC, Brier, ACC, F1, SE, SP, PPV, NPV  
4. Sauvegarde des résultats dans `results_pipline/` :
   - Résultats par fold (`cv_results_per_fold.csv/json`)  
   - Résumé global (`cv_results_summary.csv/json`)  
   - Excel complet (`cv_results_complete.xlsx`)  
   - Tests statistiques (Wilcoxon pairwise sur AUC-ROC)  
   - Figures de comparaison (`figures/model_comparison_*.png`)  

---

## 📒 Étape 5. Notebook 2 — Exécution interactive
**Fichier** : `02_run_binary_classification_pipeline.ipynb`  

Ce notebook reprend les étapes du script mais en version **pas-à-pas** et interactive, permettant de :
- Explorer les splits et vérifier les données  
- Lancer manuellement l’entraînement des modèles  
- Visualiser les résultats et figures dans Jupyter  

---

## 📂 Étape 6. Arborescence du dépôt
```
use_cases/
├─ data/
│  ├─ heart_failure_clinical_records_dataset.csv
│  └─ splits_k5_v1/
│     ├─ manifest.json
│     ├─ train_ids_fold*.csv
│     └─ test_ids_fold*.csv
├─ notebooks_usecases/
│  ├─ fixed_cv_binary_classification.py
│  ├─ 01_prepare_splits_and_config.ipynb
│  ├─ 02_run_binary_classification_pipeline.ipynb
│  └─ requirements.txt
└─ results_pipline/
   ├─ cv_results_per_fold.*
   ├─ cv_results_summary.*
   ├─ cv_results_complete.xlsx
   ├─ statistical_tests.csv
   └─ figures/
```

---

## 🚀 Étape 7. Installation & lancement
```bash
# Cloner le dépôt
git clone https://github.com/elazzouzi1080/TracIA_Usecases.git
cd TracIA_Usecases/use_cases/notebooks_usecases

# Créer un venv (optionnel)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Installer dépendances
pip install -r requirements.txt

# Lancer le pipeline complet
python fixed_cv_binary_classification.py
```
