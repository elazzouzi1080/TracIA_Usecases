# TracIA Usecases – Pipeline de classification pour l'insuffisance cardiaque

Ce dépôt rassemble un cas d'usage complet de modélisation prédictive appliqué au **Heart Failure Clinical Records Dataset**. Il inclut des données pré-séparées, un pipeline Python reproductible, ainsi que des notebooks exploratoires pour préparer les splits et exécuter la validation croisée sur plusieurs algorithmes de machine learning.

---

## 🎯 Objectifs

- Comparer différentes familles de modèles (Régression Logistique, SVM, Forêts Aléatoires, MLP, etc.).
- Utiliser des **splits de validation croisée fixes** (k=5) afin de garantir la reproductibilité.
- Calculer un large panel de métriques (AUC-ROC, F1, MCC, Brier Score, Sensibilité, Spécificité, etc.) avec intervalles de confiance.
- Générer automatiquement des exports (CSV/JSON/Excel) et des visualisations (boxplots, comparatifs de modèles, courbes ROC).

---

## 🗂️ Structure du dépôt

```
TracIA_Usecases/
├── README.md
└── use_cases/
    ├── data/
    │   ├── heart_failure_clinical_records_dataset.csv
    │   └── splits_k5_v1/          # Manifest + IDs train/test par fold
    ├── notebooks_usecases/
    │   ├── 01_prepare_splits_and_config.ipynb
    │   ├── 02_run_binary_classification_pipeline.ipynb
    │   ├── fixed_cv_binary_classification.py
    │   ├── README.md               # Guide spécifique aux notebooks
    │   └── requirements.txt
    └── results_pipline/
        ├── cv_results_per_fold.csv / .json
        ├── cv_results_summary.csv / .json
        ├── cv_results_complete.xlsx
        ├── statistical_tests.csv
        └── figures/
            ├── model_comparison_AUC_ROC.png
            ├── model_comparison_F1.png
            └── model_comparison_MCC.png
```

---

## ⚙️ Installation rapide

1. **Cloner le dépôt depuis GitHub**
   ```bash
   git clone https://github.com/<organisation>/TracIA_Usecases.git
   cd TracIA_Usecases/use_cases/notebooks_usecases
   ```

   > Remplacez `<organisation>` par le nom du compte ou de l'organisation GitHub qui héberge le dépôt.

2. **Créer un environnement virtuel (optionnel mais recommandé)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Sous Windows : .venv\\Scripts\\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Lancement du pipeline Python

Exécuter le script principal (depuis `use_cases/notebooks_usecases/`) :

```bash
python fixed_cv_binary_classification.py
```

Le pipeline :
- charge le dataset enrichi d'un `row_id`,
- applique les splits fixes définis dans `data/splits_k5_v1/`,
- entraîne chaque modèle sur les 5 folds,
- calcule toutes les métriques et intervalles de confiance,
- enregistre les résultats et figures dans `use_cases/results_pipline/`,
- lance un test statistique (Wilcoxon) pour comparer les modèles deux à deux sur l'AUC-ROC.

---

## 📓 Notebooks disponibles

- **01_prepare_splits_and_config.ipynb** : génération / inspection des splits et de la configuration.
- **02_run_binary_classification_pipeline.ipynb** : exécution pas-à-pas du pipeline, idéal pour expérimenter ou visualiser l'avancement.

Ces notebooks peuvent être ouverts dans JupyterLab/Notebook après activation de l'environnement et installation des dépendances.

---

## 🧾 Données et splits

- `data/heart_failure_clinical_records_dataset.csv` : dataset original (299 patients, 12 variables cliniques + `DEATH_EVENT`).
- `data/splits_k5_v1/manifest.json` : métadonnées des splits (k=5, cible `DEATH_EVENT`, liste des features, taille des ensembles).
- `train_ids_foldX.csv` / `test_ids_foldX.csv` : identifiants de lignes (colonne `row_id`) assignés à chaque fold.

> **Important :** les splits sont **stratifiés** et **déterministes** (`random_state=42`) pour assurer la comparabilité des résultats.

---

## 📊 Résultats générés

| Fichier | Contenu |
| ------- | ------- |
| `cv_results_per_fold.csv/json` | Toutes les métriques pour chaque modèle et chaque fold. |
| `cv_results_summary.csv/json`  | Moyennes, médianes, écarts-types et intervalles de confiance par modèle. |
| `cv_results_complete.xlsx`     | Fichier Excel avec feuilles “Per Fold Results”, “Summary” et “Metadata”. |
| `statistical_tests.csv`        | Résultats des tests de Wilcoxon (significativité pairwise sur l'AUC-ROC). |
| `figures/model_comparison_*.png` | Boxplots comparatifs (AUC-ROC, F1, MCC). |

---

## 🛠️ Personnalisation

- **Paramètres globaux** : la classe `Config` (dans `fixed_cv_binary_classification.py`) centralise les chemins, métriques suivies, taille des figures, etc.
- **Modèles** : la `ModelFactory` regroupe la définition des algorithmes évalués. Ajoutez vos propres modèles dans le dictionnaire `models`.
- **Seuils / Métrologie** : adaptez `MetricsCalculator` pour modifier le seuil de décision ou enrichir les métriques.
- **Splits** : placez vos propres fichiers `train_ids_fold*.csv` / `test_ids_fold*.csv` dans `data/splits_k5_v1/` et mettez à jour `manifest.json`.

---

## 📚 Références

- **Dataset** : [Heart Failure Clinical Records (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Bibliothèques principales** : `scikit-learn`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `tqdm`, `scipy`.

---

## 📄 Licence

La licence n'est pas spécifiée dans le dépôt. Ajoutez un fichier `LICENSE` si nécessaire pour clarifier les droits d'usage.

---

## 🤝 Contribution

1. Forker le dépôt.
2. Cloner votre fork et créer une branche de fonctionnalité :
   ```bash
   git clone https://github.com/<votre-compte>/TracIA_Usecases.git
   cd TracIA_Usecases
   git checkout -b feature/ma-fonctionnalite
   ```
3. Commiter vos changements et ouvrir une Pull Request.

---

## 📬 Contact

Pour toute question ou suggestion, ouvrez une issue GitHub ou contactez l'équipe TracIA.

---

## ✅ Récapitulatif rapide des commandes Git utiles

```bash
# Cloner le dépôt principal
git clone https://github.com/<organisation>/TracIA_Usecases.git

# Mettre à jour votre copie locale
cd TracIA_Usecases
git pull origin main

# Ajouter un dépôt distant pointant vers votre fork
git remote add fork https://github.com/<votre-compte>/TracIA_Usecases.git

# Pousser votre branche de travail vers votre fork
git push fork feature/ma-fonctionnalite
```

Ces commandes permettent de récupérer le code du dépôt GitHub, de le mettre à jour régulièrement et de partager vos modifications via un fork ou une Pull Request.
