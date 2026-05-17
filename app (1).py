"""
App Streamlit pour l'analyse de segmentation marketing.
Auteur: [Ton nom]
Date: 2026
"""

# import des librairies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# configuration de la page
st.set_page_config(page_title="Analyse Clients", layout="wide")

# -------------------------------------------------
# EN-TETE AVEC IMAGE
# -------------------------------------------------

try:
    st.image("segmentation_header.png", use_container_width=True)
except:
    st.write("Image non trouvee. Placez segmentation_header.png dans le dossier.")

st.title("Analyse des Clients E-commerce")
st.write("Application interactive pour explorer le profil de nos 1000 clients. 📊")

# -------------------------------------------------
# CHARGEMENT ET NETTOYAGE DES DONNEES
# -------------------------------------------------

# on essaie plusieurs methodes de chargement pour etre robuste
df = None

try:
    df = pd.read_csv("segmentation_marketing_raw.csv", sep=";")
    if len(df.columns) == 1:
        raise ValueError("Mauvais separateur")
except:
    pass

try:
    df = pd.read_csv("segmentation_marketing_raw.csv", sep=",")
    if len(df.columns) == 1:
        raise ValueError("Mauvais separateur")
except:
    pass

try:
    df = pd.read_csv("segmentation_marketing_raw.csv", sep=None, engine="python")
except:
    pass

if df is None or len(df.columns) != 5:
    st.error("Erreur de chargement du CSV.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
cols_attendues = ["revenu_annuel", "score_fidelite", "frequence_achat", "depenses_moyennes", "anciennete_client"]
for col in cols_attendues:
    if col not in df.columns:
        st.error(f"Colonne manquante : {col}")
        st.stop()

df_clean = df.copy()
df_clean = df_clean.astype(float)

# nettoyage des valeurs negatives
for col in df_clean.columns:
    df_clean.loc[df_clean[col] < 0, col] = np.nan
    mediane = df_clean[col].median()
    df_clean[col].fillna(mediane, inplace=True)

# -------------------------------------------------
# SIDEBAR - FILTRES
# -------------------------------------------------

st.sidebar.header("Filtres 🔍")

# on utilise st.number_input au lieu de st.slider
# car les sliders ont des problemes de precision en ligne

# revenu
rev_min_val = float(df_clean["revenu_annuel"].min())
rev_max_val = float(df_clean["revenu_annuel"].max())

st.sidebar.write("Revenu annuel (€)")
col1, col2 = st.sidebar.columns(2)
with col1:
    revenu_min = st.number_input("Min", min_value=rev_min_val, max_value=rev_max_val, value=rev_min_val, key="rev_min")
with col2:
    revenu_max = st.number_input("Max", min_value=rev_min_val, max_value=rev_max_val, value=rev_max_val, key="rev_max")

# anciennete
anc_min_val = float(df_clean["anciennete_client"].min())
anc_max_val = float(df_clean["anciennete_client"].max())

st.sidebar.write("Anciennete client (annees)")
col3, col4 = st.sidebar.columns(2)
with col3:
    anciennete_min = st.number_input("Min", min_value=anc_min_val, max_value=anc_max_val, value=anc_min_val, key="anc_min")
with col4:
    anciennete_max = st.number_input("Max", min_value=anc_min_val, max_value=anc_max_val, value=anc_max_val, key="anc_max")

# score de fidelite
score_min_val = float(df_clean["score_fidelite"].min())
score_max_val = float(df_clean["score_fidelite"].max())

st.sidebar.write("Score de fidelite")
col5, col6 = st.sidebar.columns(2)
with col5:
    score_min = st.number_input("Min", min_value=score_min_val, max_value=score_max_val, value=score_min_val, key="score_min")
with col6:
    score_max = st.number_input("Max", min_value=score_min_val, max_value=score_max_val, value=score_max_val, key="score_max")

st.sidebar.markdown("---")
st.sidebar.info("Ajustez les filtres pour explorer differentes sous-populations.")

# application des filtres
mask = (
    (df_clean["revenu_annuel"] >= revenu_min) &
    (df_clean["revenu_annuel"] <= revenu_max) &
    (df_clean["anciennete_client"] >= anciennete_min) &
    (df_clean["anciennete_client"] <= anciennete_max) &
    (df_clean["score_fidelite"] >= score_min) &
    (df_clean["score_fidelite"] <= score_max)
)

df_filtre = df_clean[mask].copy()

st.write("Nombre de clients apres filtrage :", len(df_filtre), "👥")

# -------------------------------------------------
# SECTION 1 : QUI SONT NOS CLIENTS ?
# -------------------------------------------------

st.header("1. Qui sont nos clients ? 📈")

st.subheader("Statistiques descriptives")
st.write(df_filtre.describe().round(2))

st.subheader("Distributions des variables")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(df_filtre.columns):
    if i < 5:
        axes[i].hist(df_filtre[col], bins=30, color="steelblue", edgecolor="black")
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Nombre de clients")

axes[5].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

st.write("Interpretation : ces histogrammes montrent la repartition des clients sur chaque variable.")

# -------------------------------------------------
# SECTION 2 : RELATIONS ENTRE VARIABLES
# -------------------------------------------------

st.header("2. Y a-t-il des relations entre les variables ? 🔗")

st.subheader("Matrice de correlation")

corr = df_filtre.corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax2, fmt=".2f")
ax2.set_title("Correlations entre les 5 variables")
st.pyplot(fig2)

st.write("Interpretation : la matrice montre les forces des liens entre variables.")

st.subheader("Nuages de points")

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(df_filtre["revenu_annuel"], df_filtre["depenses_moyennes"], alpha=0.5, color="darkgreen")
ax3.set_xlabel("Revenu annuel")
ax3.set_ylabel("Depenses moyennes")
ax3.set_title("Revenu annuel vs Depenses moyennes")
st.pyplot(fig3)

st.write("Interpretation : ce graphique montre si les clients qui gagnent plus depensent aussi plus.")

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(df_filtre["frequence_achat"], df_filtre["anciennete_client"], alpha=0.5, color="darkorange")
ax4.set_xlabel("Frequence d'achat")
ax4.set_ylabel("Anciennete client")
ax4.set_title("Frequence d'achat vs Anciennete")
st.pyplot(fig4)

st.write("Interpretation : ce graphique montre si les clients anciens achetent plus souvent.")

# -------------------------------------------------
# SECTION 3 : CLUSTERING K-MEANS
# -------------------------------------------------

st.header("3. Existe-t-il des groupes de clients distincts ? 🎯")

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtre)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

df_filtre["cluster"] = clusters

st.subheader("Repartition des clusters")
st.write(df_filtre["cluster"].value_counts().sort_index())

st.subheader("Profils moyens par cluster")
st.write(df_filtre.groupby("cluster").mean().round(2))

st.subheader("Visualisation des clusters")

fig5, ax5 = plt.subplots(figsize=(10, 7))
colors = ["red", "blue", "green"]

for i in range(3):
    subset = df_filtre[df_filtre["cluster"] == i]
    ax5.scatter(subset["revenu_annuel"], subset["depenses_moyennes"],
                c=colors[i], label="Cluster " + str(i), alpha=0.6)

ax5.set_xlabel("Revenu annuel")
ax5.set_ylabel("Depenses moyennes")
ax5.set_title("Clusters de clients - Revenu vs Depenses")
ax5.legend()
st.pyplot(fig5)

st.write("Interpretation : K-Means a separe les clients en 3 groupes distincts.")

fig6, ax6 = plt.subplots(figsize=(10, 7))

for i in range(3):
    subset = df_filtre[df_filtre["cluster"] == i]
    ax6.scatter(subset["frequence_achat"], subset["score_fidelite"],
                c=colors[i], label="Cluster " + str(i), alpha=0.6)

ax6.set_xlabel("Frequence d'achat")
ax6.set_ylabel("Score de fidelite")
ax6.set_title("Clusters - Frequence vs Fidelite")
ax6.legend()
st.pyplot(fig6)

st.write("Interpretation : on peut voir si les clients qui achetent souvent ont un meilleur score de fidelite.")

# -------------------------------------------------
# SECTION 4 : OUTLIERS
# -------------------------------------------------

st.header("4. Quelles sont les anomalies ou valeurs extremes ? ⚠️")

st.subheader("Boxplots des variables")

fig7, axes7 = plt.subplots(2, 3, figsize=(15, 10))
axes7 = axes7.flatten()

for i, col in enumerate(df_filtre.columns):
    if i < 5 and col != "cluster":
        sns.boxplot(y=df_filtre[col], ax=axes7[i], color="lightcoral")
        axes7[i].set_title("Boxplot de " + col)

axes7[5].set_visible(False)
plt.tight_layout()
st.pyplot(fig7)

st.write("Interpretation : les points isoles au-dessus ou en-dessous des moustaches sont des outliers.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.write("---")
st.write("Application realisee avec Streamlit, Pandas, Matplotlib, Seaborn et Scikit-learn. ✨")
