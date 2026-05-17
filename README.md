# Projet Segmentation Marketing

## Description

Ce projet est une application Streamlit qui permet d'explorer un dataset de 1000 clients d'un site e-commerce. L'application repond a 4 questions principales :

1. Qui sont nos clients ? (distributions, stats descriptives)
2. Y a-t-il des relations entre les variables ? (correlations, scatter plots)
3. Existe-t-il des groupes de clients distincts ? (K-Means, 3 clusters)
4. Quelles sont les anomalies ? (boxplots, outliers)

## Donnees

Le dataset `segmentation_marketing_raw.csv` contient 5 variables :
- `revenu_annuel` : revenu annuel du client (en euros)
- `score_fidelite` : score de fidelite
- `frequence_achat` : nombre d'achats par mois
- `depenses_moyennes` : montant moyen depense par achat
- `anciennete_client` : nombre d'annees depuis le premier achat

### Nettoyage des donnees

Le dataset contient des valeurs negatives qui n'ont pas de sens dans ce contexte (un revenu ou une anciennete ne peuvent pas etre negatifs). La technique utilisee est :
1. Remplacer les valeurs negatives par NaN
2. Imputer les NaN par la mediane de chaque colonne

Cette methode est expliquee dans le notebook `segmentation_marketing_raw.ipynb`.

## Installation

1. Cloner ou telecharger le projet
2. Installer les dependances :

```bash
pip install -r requirements.txt
```

## Lancement de l'application

Placer le fichier `segmentation_marketing_raw.csv` dans le meme dossier que `app.py`, puis executer :

```bash
streamlit run app.py
```

L'application s'ouvrira dans le navigateur.

## Fichiers

- `app.py` : application Streamlit interactive
- `segmentation_marketing_raw.ipynb` : notebook d'analyse exploratoire (EDA)
- `segmentation_marketing_raw.csv` : dataset brut
- `requirements.txt` : liste des librairies necessaires

## Technologies utilisees

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn (K-Means)
