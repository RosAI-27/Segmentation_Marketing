import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# CSS PERSONNALISE
st.markdown("""
<style>
    /* Fond general */
    .main {
        background-color: #f8f9fa;
    }

    /* Titres */
    h1 {
        color: #1f4e79;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
        border-bottom: 3px solid #2e86de;
        padding-bottom: 10px;
    }

    h2 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
        margin-top: 30px;
        border-left: 4px solid #2e86de;
        padding-left: 12px;
    }

    h3 {
        color: #34495e;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
    }

    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #2c3e50;
    }

    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }

    /* Texte dans sidebar */
    .css-1v3fvcr {
        color: white;
    }

    /* Boutons et widgets */
    .stSlider > div > div > div > div {
        background-color: #2e86de;
    }

    /* Cards pour les sections */
    .stAlert {
        border-radius: 8px;
        border: none;
    }

    /* Interpretations */
    .stMarkdown {
        font-size: 15px;
        line-height: 1.6;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }

    /* Style des graphiques matplotlib */
    .stImage {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Info box */
    .stInfo {
        background-color: #e8f4f8;
        border-left: 4px solid #2e86de;
        border-radius: 4px;
    }

    /* Tableaux */
    .dataframe {
        font-size: 13px;
        border-collapse: collapse;
    }

    .dataframe th {
        background-color: #2e86de;
        color: white;
        font-weight: 500;
        padding: 8px;
    }

    .dataframe td {
        padding: 6px;
        border-bottom: 1px solid #ddd;
    }

    .dataframe tr:hover {
        background-color: #f1f1f1;
    }
</style>
""", unsafe_allow_html=True)

# CONFIGURATION PAGE
st.set_page_config(page_title="Analyse Clients", layout="wide")

# EN-TETE
try:
    st.image("segmentation_header.png", use_container_width=True)
except:
    st.write("")

st.title("Analyse des Clients E-commerce")
st.write("Application interactive pour explorer le profil de nos 1000 clients. 📊")

# CHARGEMENT ET NETTOYAGE
df = pd.read_csv("segmentation_marketing_raw.csv", sep=";")

df_clean = df.copy()

# nettoyage des valeurs negatives
for col in df_clean.columns:
    df_clean.loc[df_clean[col] < 0, col] = np.nan
    mediane = df_clean[col].median()
    df_clean[col].fillna(mediane, inplace=True)

# SIDEBAR - FILTRES
st.sidebar.header("Filtres 🔍")

# SOLUTION BUG SLIDER : utiliser des entiers pour min/max
# et step=1 pour eviter les problemes de float

# revenu
rev_min = int(df_clean["revenu_annuel"].min())
rev_max = int(df_clean["revenu_annuel"].max()) + 1
revenu_range = st.sidebar.slider(
    "Revenu annuel (€)",
    min_value=rev_min,
    max_value=rev_max,
    value=(rev_min, rev_max),
    step=1
)

# anciennete
anc_min = int(df_clean["anciennete_client"].min())
anc_max = int(df_clean["anciennete_client"].max()) + 1
anciennete_range = st.sidebar.slider(
    "Anciennete client (annees)",
    min_value=anc_min,
    max_value=anc_max,
    value=(anc_min, anc_max),
    step=1
)

# score de fidelite
score_min = int(df_clean["score_fidelite"].min())
score_max = int(df_clean["score_fidelite"].max()) + 1
score_range = st.sidebar.slider(
    "Score de fidelite",
    min_value=score_min,
    max_value=score_max,
    value=(score_min, score_max),
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info("Ajustez les filtres pour explorer differentes sous-populations de clients.")

# APPLICATION FILTRES
mask = (
    (df_clean["revenu_annuel"] >= revenu_range[0]) &
    (df_clean["revenu_annuel"] <= revenu_range[1]) &
    (df_clean["anciennete_client"] >= anciennete_range[0]) &
    (df_clean["anciennete_client"] <= anciennete_range[1]) &
    (df_clean["score_fidelite"] >= score_range[0]) &
    (df_clean["score_fidelite"] <= score_range[1])
)

df_filtre = df_clean[mask].copy()

# METRIQUES EN HAUT
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Clients filtres", len(df_filtre))
with col2:
    st.metric("Revenu moyen", f"{df_filtre['revenu_annuel'].mean():.0f} €")
with col3:
    st.metric("Score fidelite moyen", f"{df_filtre['score_fidelite'].mean():.1f}")
with col4:
    st.metric("Frequence moyenne", f"{df_filtre['frequence_achat'].mean():.1f}")

st.markdown("---")

# SECTION 1 : QUI SONT NOS CLIENTS
st.header("1. Qui sont nos clients ? 📈")

st.subheader("Statistiques descriptives")
st.write(df_filtre.describe().round(2))

st.subheader("Distributions des variables")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(df_filtre.columns):
    if i < 5:
        axes[i].hist(df_filtre[col], bins=30, color="#2e86de", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel("Nombre", fontsize=9)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

axes[5].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

st.info(" **Interpretation** : ces histogrammes montrent la repartition des clients sur chaque variable. Le revenu et les depenses sont assez etales, tandis que la frequence d'achat est concentree autour de 20 achats par mois.")

# SECTION 2 : RELATIONS
st.header("2. Y a-t-il des relations entre les variables ? 🔗")

st.subheader("Matrice de correlation")

corr = df_filtre.corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax2, fmt=".2f",
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax2.set_title("Correlations entre les 5 variables", fontsize=12, fontweight='bold')
st.pyplot(fig2)

st.info(" **Interpretation** : les correlations ne sont pas tres fortes, ce qui signifie que les variables sont assez independantes. C'est interessant car chaque variable apporte une information differente sur les clients.")

st.subheader("Nuages de points")

col_left, col_right = st.columns(2)

with col_left:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(df_filtre["revenu_annuel"], df_filtre["depenses_moyennes"], 
                alpha=0.5, color="#27ae60", s=30)
    ax3.set_xlabel("Revenu annuel", fontsize=10)
    ax3.set_ylabel("Depenses moyennes", fontsize=10)
    ax3.set_title("Revenu vs Depenses", fontsize=11, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    st.pyplot(fig3)

with col_right:
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(df_filtre["frequence_achat"], df_filtre["anciennete_client"], 
                alpha=0.5, color="#e67e22", s=30)
    ax4.set_xlabel("Frequence d'achat", fontsize=10)
    ax4.set_ylabel("Anciennete client", fontsize=10)
    ax4.set_title("Frequence vs Anciennete", fontsize=11, fontweight='bold')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    st.pyplot(fig4)

st.info(" **Interpretation** : les nuages de points sont disperses, ce qui confirme que les variables sont peu correlees. Les clients qui gagnent plus ne depensent pas forcement plus, et l'anciennete ne determine pas la frequence d'achat.")

# SECTION 3 : CLUSTERING
st.header("3. Existe-t-il des groupes de clients distincts ? 🎯")

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_filtre)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df_scaled)

df_filtre["cluster"] = clusters

col_c1, col_c2, col_c3 = st.columns(3)
for i in range(3):
    count = (df_filtre["cluster"] == i).sum()
    pct = count / len(df_filtre) * 100
    with [col_c1, col_c2, col_c3][i]:
        st.metric(f"Cluster {i}", f"{count} clients", f"{pct:.1f}%")

st.subheader("Profils moyens par cluster")
st.write(df_filtre.groupby("cluster").mean().round(2))

st.subheader("Visualisation des clusters")

fig5, ax5 = plt.subplots(figsize=(10, 7))
colors = ["#e74c3c", "#3498db", "#2ecc71"]

for i in range(3):
    subset = df_filtre[df_filtre["cluster"] == i]
    ax5.scatter(subset["revenu_annuel"], subset["depenses_moyennes"],
                c=colors[i], label=f"Cluster {i}", alpha=0.6, s=40)

ax5.set_xlabel("Revenu annuel", fontsize=11)
ax5.set_ylabel("Depenses moyennes", fontsize=11)
ax5.set_title("Clusters - Revenu vs Depenses", fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
st.pyplot(fig5)

fig6, ax6 = plt.subplots(figsize=(10, 7))

for i in range(3):
    subset = df_filtre[df_filtre["cluster"] == i]
    ax6.scatter(subset["frequence_achat"], subset["score_fidelite"],
                c=colors[i], label=f"Cluster {i}", alpha=0.6, s=40)

ax6.set_xlabel("Frequence d'achat", fontsize=11)
ax6.set_ylabel("Score de fidelite", fontsize=11)
ax6.set_title("Clusters - Frequence vs Fidelite", fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
st.pyplot(fig6)

st.info(" **Interpretation** : K-Means a identifie 3 profils distincts. Chaque cluster represente un type de client different que le marketing peut cibler avec des strategies adaptees.")

# SECTION 4 : OUTLIERS
st.header("4. Quelles sont les anomalies ou valeurs extremes ? ⚠️")

st.subheader("Boxplots des variables")

fig7, axes7 = plt.subplots(2, 3, figsize=(15, 10))
axes7 = axes7.flatten()

for i, col in enumerate(df_filtre.columns):
    if i < 5 and col != "cluster":
        bp = axes7[i].boxplot(df_filtre[col], patch_artist=True,
                               boxprops=dict(facecolor="#ff6b6b", alpha=0.7),
                               medianprops=dict(color="white", linewidth=2),
                               whiskerprops=dict(color="#2c3e50"),
                               capprops=dict(color="#2c3e50"))
        axes7[i].set_title(f"Boxplot de {col}", fontsize=11, fontweight='bold')
        axes7[i].set_ylabel(col, fontsize=9)
        axes7[i].spines['top'].set_visible(False)
        axes7[i].spines['right'].set_visible(False)

axes7[5].set_visible(False)
plt.tight_layout()
st.pyplot(fig7)

st.info(" **Interpretation** : les points au-dela des moustaches sont des outliers. On observe beaucoup de valeurs extremes sur le revenu et les depenses. Ces clients atypiques meritent une analyse separee pour comprendre leurs comportements specifiques.")

# FOOTER
st.markdown("---")
st.markdown("<center><p style='color: #7f8c8d; font-size: 13px;'>Application realisee avec Streamlit, Pandas, Matplotlib, Seaborn et Scikit-learn</p></center>", unsafe_allow_html=True)
