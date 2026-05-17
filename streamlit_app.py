# import des librairies
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# configuration de la page
st.set_page_config(page_title="Analyse Clients", layout="wide")

# EN-TETE AVEC IMAGE

# image d'en-tete : segmentation client
try:
    st.image("segmentation_header.png", use_container_width=True)
except:
    st.write("Image non trouvee. Placez segmentation_header.png dans le dossier.")

# titre de l'application
st.title("Analyse des Clients E-commerce")
st.write("Application interactive pour explorer le profil de nos 1000 clients. 📊")

# CHARGEMENT ET NETTOYAGE DES DONNEES

# on charge le dataset
df = pd.read_csv("segmentation_marketing_raw.csv", sep=";")

# copie pour le nettoyage
df_clean = df.copy()

# les valeurs negatives n'ont pas de sens dans ce dataset
# (revenu, anciennete, score ne peuvent pas etre negatifs)
# on les remplace par NaN puis on impute par la mediane
for col in df_clean.columns:
    df_clean.loc[df_clean[col] < 0, col] = np.nan
    mediane = df_clean[col].median()
    df_clean[col].fillna(mediane, inplace=True)

# SIDEBAR - FILTRES

st.sidebar.header("Filtres 🔍")

# slider pour le revenu
min_rev = math.floor(df_clean["revenu_annuel"].min())
max_rev = math.ceil(df_clean["revenu_annuel"].max())
revenu_range = st.sidebar.slider(
    "Revenu annuel (€)",
    min_value=min_rev,
    max_value=max_rev,
    value=(min_rev, max_rev)
)

# slider pour l'anciennete
min_anc = math.floor(df_clean["anciennete_client"].min())
max_anc = math.ceil(df_clean["anciennete_client"].max())
anciennete_range = st.sidebar.slider(
    "Anciennete client (annees)",
    min_value=min_anc,
    max_value=max_anc,
    value=(min_anc, max_anc)
)

# slider pour le score de fidelite
min_score = math.floor(df_clean["score_fidelite"].min())
max_score = math.ceil(df_clean["score_fidelite"].max())
score_range = st.sidebar.slider(
    "Score de fidelite",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score)
)

st.sidebar.markdown("---")
st.sidebar.info("Ajustez les filtres pour explorer differentes sous-populations de clients.")

# application des filtres
mask = (
    (df_clean["revenu_annuel"] >= revenu_range[0]) &
    (df_clean["revenu_annuel"] <= revenu_range[1]) &
    (df_clean["anciennete_client"] >= anciennete_range[0]) &
    (df_clean["anciennete_client"] <= anciennete_range[1]) &
    (df_clean["score_fidelite"] >= score_range[0]) &
    (df_clean["score_fidelite"] <= score_range[1])
)

df_filtre = df_clean[mask].copy()

st.write("Nombre de clients apres filtrage :", len(df_filtre), "👥")

# SECTION 1 : QUI SONT NOS CLIENTS ?

st.header("1. Qui sont nos clients ? 📈")

# stats descriptives
st.subheader("Statistiques descriptives")
st.write(df_filtre.describe().round(2))

# histogrammes
st.subheader("Distributions des variables")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(df_filtre.columns):
    if i < 5:
        axes[i].hist(df_filtre[col], bins=30, color="steelblue", edgecolor="black")
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Nombre de clients")

# on cache le 6eme graphique car on a 5 variables
axes[5].set_visible(False)

plt.tight_layout()
st.pyplot(fig)

st.write("Interpretation : ces histogrammes montrent comment sont repartis les clients sur chaque variable. On voit que le revenu et les depenses sont assez etales, tandis que la frequence d'achat est plus concentree autour de 20 achats par mois.")

# SECTION 2 : RELATIONS ENTRE VARIABLES

st.header("2. Y a-t-il des relations entre les variables ? 🔗")

# matrice de correlation
st.subheader("Matrice de correlation")

corr = df_filtre.corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax2, fmt=".2f")
ax2.set_title("Correlations entre les 5 variables")
st.pyplot(fig2)

st.write("Interpretation : la matrice montre les forces des liens entre variables. Par exemple si deux variables ont une correlation proche de 1, elles augmentent ensemble. Ici on voit que les correlations ne sont pas tres fortes, ce qui signifie que les variables sont assez independantes.")

# scatter plots
st.subheader("Nuages de points")

# revenu vs depenses
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(df_filtre["revenu_annuel"], df_filtre["depenses_moyennes"], alpha=0.5, color="darkgreen")
ax3.set_xlabel("Revenu annuel")
ax3.set_ylabel("Depenses moyennes")
ax3.set_title("Revenu annuel vs Depenses moyennes")
st.pyplot(fig3)

st.write("Interpretation : ce graphique montre si les clients qui gagnent plus depensent aussi plus. On voit un nuage assez disperse, ce qui veut dire que ce n'est pas toujours le cas.")

# frequence vs anciennete
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(df_filtre["frequence_achat"], df_filtre["anciennete_client"], alpha=0.5, color="darkorange")
ax4.set_xlabel("Frequence d'achat")
ax4.set_ylabel("Anciennete client")
ax4.set_title("Frequence d'achat vs Anciennete")
st.pyplot(fig4)

st.write("Interpretation : ce graphique montre si les clients anciens achetent plus souvent. Le nuage est disperse donc l'anciennete ne determine pas forcement la frequence d'achat.")

# SECTION 3 : CLUSTERING K-MEANS

st.header("3. Existe-t-il des groupes de clients distincts ? 🎯")

# on fait le clustering sur les donnees filtrees
# standardisation
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtre)

# K-Means avec 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

# on ajoute les clusters au dataframe
df_filtre["cluster"] = clusters

st.subheader("Repartition des clusters")
st.write(df_filtre["cluster"].value_counts().sort_index())

# moyennes par cluster
st.subheader("Profils moyens par cluster")
st.write(df_filtre.groupby("cluster").mean().round(2))

# visualisation des clusters
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

st.write("Interpretation : K-Means a separe les clients en 3 groupes. Chaque couleur represente un profil different. On peut voir que certains clusters correspondent a des clients a haut revenu et hautes depenses, d'autres a des clients plus modestes.")

# deuxieme visualisation
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

st.write("Interpretation : ce graphique montre les clusters selon la frequence d'achat et le score de fidelite. On peut voir si les clients qui achetent souvent ont aussi un meilleur score de fidelite.")

# SECTION 4 : OUTLIERS

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

st.write("Interpretation : les boxplots montrent les valeurs extremes (outliers) sous forme de points isoles au-dessus ou en-dessous des moustaches. On voit qu'il y a beaucoup d'outliers sur le revenu annuel et les depenses moyennes. Ces clients ont des comportements atypiques qu'il faudrait peut-etre etudier separement.")

# FOOTER

st.write("---")
st.write("Application realisee avec Streamlit, Pandas, Matplotlib, Seaborn et Scikit-learn.")
