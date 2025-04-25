# 🩺 Analyse & Prédiction du Risque Cardiaque

Cette application Streamlit permet d'explorer, d'analyser et de prédire la colonne `Risk` à partir d'un fichier CSV (par défaut `high.csv`).

## Fonctionnalités principales
- **Exploration interactive** des données (statistiques, graphiques dynamiques, corrélations)
- **Upload de fichier CSV** personnalisé
- **Modélisation automatique** (classification ou régression selon la colonne Risk)
- **Prédiction personnalisée** avec formulaire interactif
- **Animations Lottie** et interface moderne, responsive, avec transitions et effets visuels
- **Menu latéral** pour une navigation fluide

## Lancer l'application en local

1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancez Streamlit :
   ```bash
   streamlit run app.py
   ```

## Déploiement sur Streamlit Cloud
- Uploadez les fichiers `app.py`, `requirements.txt`, `high.csv` et `README.md` sur [Streamlit Cloud](https://share.streamlit.io/)
- L'application sera automatiquement déployée et accessible en ligne

## Personnalisation
- Vous pouvez utiliser votre propre fichier CSV via l'interface d'upload
- Les animations Lottie sont personnalisables dans le code (`app.py`)

---

**Auteur :** VotreNom 