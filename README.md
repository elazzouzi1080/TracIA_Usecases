 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/README.md
index 0000000000000000000000000000000000000000..0917acfd08bbb3396b486bc4885f7d82dab0ead4 100644
--- a//dev/null
+++ b/README.md
@@ -0,0 +1,85 @@
+# TracIA Use Cases – Pipeline de classification cardiaque
+
+Ce dépôt regroupe un cas d'usage complet autour du **Heart Failure Clinical Records Dataset** pour illustrer la construction d'un pipeline de classification binaire reproductible avec validation croisée fixe. Il comprend les données préparées, les notebooks d'exploration et un script Python orchestrant l'entraînement et l'évaluation de plusieurs modèles de machine learning.
+
+## Contenu principal
+
+- **Données** : le jeu de données original enrichi d'un identifiant de ligne ainsi que des splits stratifiés (train/test) pour 5 folds fixes, prêts à l'emploi pour la validation croisée.【F:use_cases/data/splits_k5_v1/manifest.json†L1-L41】【F:use_cases/data/splits_k5_v1/heart_failure_clinical_records_dataset_with_row_id.csv†L1-L5】
+- **Notebooks & scripts** : deux notebooks Jupyter et un script Python `fixed_cv_binary_classification.py` qui automatisent le pipeline (prétraitement, entraînement, calcul des métriques et génération des figures).【F:use_cases/notebooks_usecases/README.md†L1-L43】【F:use_cases/notebooks_usecases/fixed_cv_binary_classification.py†L1-L120】
+- **Résultats** : exports tabulaires et graphiques (CSV, JSON, Excel) déjà générés à titre d'exemple dans `use_cases/results_pipline/`.【F:use_cases/results_pipline/cv_results_summary.csv†L1-L5】【bf080f†L1-L3】
+
+## Structure du dépôt
+
+```
+TracIA_Usecases/
+├── README.md                 # Ce guide
+└── use_cases/
+    ├── data/                 # Données brutes et splits fixes
+    ├── notebooks_usecases/   # Notebooks, script principal et requirements
+    └── results_pipline/      # Résultats d'exécution et visualisations
+```
+
+## Prérequis
+
+- Python ≥ 3.9 (un environnement virtuel est recommandé)
+- `pip` ou `uv` pour gérer les dépendances
+- (Optionnel) JupyterLab/Notebook si vous souhaitez exécuter les notebooks interactifs
+
+Toutes les dépendances nécessaires au pipeline sont listées dans `use_cases/notebooks_usecases/requirements.txt` (scikit-learn, pandas, matplotlib, seaborn, etc.).【F:use_cases/notebooks_usecases/requirements.txt†L1-L108】
+
+## Installation rapide
+
+```bash
+# 1. Cloner ou décompresser le projet
+cd TracIA_Usecases
+
+# 2. Créer un environnement virtuel (recommandé)
+python -m venv .venv
+source .venv/bin/activate        # Sous Windows : .venv\\Scripts\\activate
+
+# 3. Installer les dépendances
+pip install -r use_cases/notebooks_usecases/requirements.txt
+```
+
+## Exécution du pipeline automatisé
+
+```bash
+cd use_cases/notebooks_usecases
+python fixed_cv_binary_classification.py
+```
+
+Le script :
+- charge les données et les splits déterministes,
+- entraîne plusieurs modèles de référence (Logistic Regression, SVM, Random Forest, MLP, etc.),
+- calcule un ensemble de métriques (AUC-ROC, F1, MCC, Brier Score, Balanced Accuracy…),
+- enregistre tableaux, rapports et figures dans `../results_pipline/` sous forme de CSV, JSON, Excel et PNG.【F:use_cases/notebooks_usecases/fixed_cv_binary_classification.py†L21-L120】【F:use_cases/results_pipline/cv_results_per_fold.csv†L1-L2】
+
+## Utilisation des notebooks
+
+1. Activez votre environnement virtuel et lancez Jupyter Lab ou Notebook.
+2. Ouvrez `01_prepare_splits_and_config.ipynb` pour explorer la génération des splits et la configuration.
+3. Ouvrez `02_run_binary_classification_pipeline.ipynb` pour exécuter le pipeline pas à pas et visualiser les résultats.【F:use_cases/notebooks_usecases/README.md†L1-L43】
+
+Les notebooks reposent sur les mêmes dépendances que le script et produisent les mêmes artefacts dans `use_cases/results_pipline/`.
+
+## Résultats et interprétation
+
+Après exécution, vous trouverez :
+
+- `cv_results_summary.*` : tableau comparatif des performances moyennes par modèle (CSV/JSON/Excel).【F:use_cases/results_pipline/cv_results_summary.csv†L1-L2】
+- `cv_results_per_fold.*` : résultats détaillés par fold pour chaque métrique.【F:use_cases/results_pipline/cv_results_per_fold.csv†L1-L2】
+- `statistical_tests.csv` : résultats des tests statistiques pairés sur les métriques clés.【F:use_cases/results_pipline/statistical_tests.csv†L1-L2】
+- `figures/` : visualisations (courbes ROC, PR, comparaisons de scores, matrices de confusion…).【bf080f†L1-L3】
+
+Ces artefacts peuvent être utilisés pour documenter vos expériences, préparer des rapports ou comparer rapidement de nouveaux modèles en ajoutant vos propres estimateurs dans le script.
+
+## Personnalisation
+
+- **Ajouter un modèle** : étendez la liste des classificateurs dans `fixed_cv_binary_classification.py` en suivant la structure existante (dictionnaire `model_configs`).
+- **Changer les métriques** : modifiez `metrics_list` dans la classe `Config` pour ajouter ou retirer des métriques calculées.【F:use_cases/notebooks_usecases/fixed_cv_binary_classification.py†L52-L86】
+- **Nouvelles données** : placez votre fichier CSV dans `use_cases/data/`, mettez à jour `Config.dataset_csv` et générez de nouveaux splits en adaptant `01_prepare_splits_and_config.ipynb`.
+
+## Support
+
+Ce projet sert de point de départ pour expérimenter les pipelines de classification supervisée dans un contexte médical. N'hésitez pas à forker le dépôt et à adapter les scripts/notebooks à vos propres cas d'usage.
+
 
EOF
)
