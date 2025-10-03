# 🧪 TracIA Usecases — Pipeline ML (Heart Failure Dataset)

Ce dépôt présente un **pipeline reproductible** d’expérimentation machine learning appliqué à un dataset médical de référence : le *Heart Failure Clinical Records Dataset*.  
L’objectif est de fournir un exemple simple, transparent et traçable pour comparer plusieurs modèles de classification binaire.

---

## ⚙️ Étape 1. Contexte général
- **Projet** : démonstration de cas d’usage pour le projet **TracIA** (traçabilité & IA en santé).  
- **But** : tester et comparer différents modèles de machine learning (LogReg, SVM, RF, MLP, etc.) dans un cadre reproductible.  
- **Méthode clé** : utilisation de *splits fixes* (5-fold cross-validation déterministe) afin que tous les partenaires puissent évaluer leurs méthodes sur les **mêmes conditions de validation**.

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
