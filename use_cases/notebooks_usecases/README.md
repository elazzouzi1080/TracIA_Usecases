# Binary Classification Pipeline – Heart Failure Dataset

## Description
Ce projet fournit un pipeline complet de classification binaire avec validation croisée fixe sur le **Heart Failure Clinical Records Dataset**.  
Il permet d’entraîner et d’évaluer plusieurs modèles de machine learning (**Logistic Regression, SVM, Random Forest, MLP, etc.**) à l’aide de splits déterministes.  

Les résultats incluent :
- Des métriques détaillées (AUC-ROC, F1, MCC, etc.)  
- Des visualisations comparatives des modèles  
- Des exports (CSV, JSON, Excel)  

---

## 1. Décompression du projet
Téléchargez et décompressez l’archive :
```bash
unzip TracIA_Usecases.zip
cd TracIA_Usecases
```

---

## 2. Installation des dépendances
Créez un environnement virtuel (recommandé) puis installez les librairies :

```bash
python -m venv venv
source venv/bin/activate   # Linux 


pip install -r requirements.txt
```

---

## 3. Exécution du pipeline
Lancez le script principal :
```bash
python fixed_cv_binary_classification.py
```

Les résultats seront générés dans le dossier :
```
results_pipline/
```

---

## 4. Utilisation des notebooks
Vous pouvez aussi ouvrir les notebooks avec **Jupyter Lab / Notebook** :
- `02_run_binary_classification_pipeline.ipynb`
