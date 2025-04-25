import streamlit as st
import plotly.express as px
from streamlit_lottie import st_lottie

from scripts.data_utils import DataManager
from scripts.visualization import DataVisualizer
from scripts.model_utils import RiskModel
from scripts.lottie_utils import LottieLoader

# Configuration de la page
st.set_page_config(
    page_title="Analyse & Prédiction du Risque",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lottie Animations
lottie_analytics = LottieLoader.load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_ktwnwv5m.json")
lottie_ml = LottieLoader.load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")
lottie_upload = LottieLoader.load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json")

# Menu latéral
st.sidebar.title("Navigation 🧭")
menu = st.sidebar.radio("Aller à :", ["Accueil", "Exploration des données", "Modélisation", "Prédiction personnalisée"])

# Upload de fichier
st.sidebar.markdown("---")
st.sidebar.header("Charger un fichier CSV")
user_file = st.sidebar.file_uploader("Uploader votre fichier CSV", type=["csv"])

data_manager = DataManager(user_file, "./data/high.csv")
df = data_manager.df
task_type = data_manager.detect_task_type("Risk")
visualizer = DataVisualizer(df)

# Accueil
if menu == "Accueil":
    st_lottie(lottie_analytics, height=200, key="analytics")
    st.title("🩺 Analyse & Prédiction du Risque Cardiaque")
    st.markdown("""
    <style>
    .big-font {font-size:30px !important; font-weight: bold; color: #4F8BF9;}
    .fade-in {animation: fadeIn 2s;}
    @keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
    </style>
    <div class="big-font fade-in">Bienvenue sur l'application d'analyse et de prédiction du risque cardiaque !</div>
    """, unsafe_allow_html=True)
    st.write("\nCette application vous permet d'explorer vos données, de visualiser des statistiques et de prédire le risque à partir de vos propres fichiers CSV.")
    st_lottie(lottie_upload, height=120, key="upload")
    st.info("Utilisez le menu latéral pour naviguer entre les sections.")

# Exploration des données
elif menu == "Exploration des données":
    st_lottie(lottie_analytics, height=120, key="explore")
    st.header("🔎 Exploration des données")
    head_df, desc_df = visualizer.get_data_overview()
    st.write("Aperçu du jeu de données :")
    st.dataframe(head_df, use_container_width=True)
    st.markdown("---")
    st.subheader("Statistiques descriptives")
    st.dataframe(desc_df, use_container_width=True)
    st.markdown("---")
    st.subheader("Distribution de la variable cible (Risk)")
    fig_risk = visualizer.get_risk_distribution()
    st.plotly_chart(fig_risk, use_container_width=True)
    st.markdown("---")
    st.subheader("Corrélation entre les variables")
    fig_corr = visualizer.get_correlation()
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("---")
    st.subheader("Visualisation interactive")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Axe X", df.columns, index=1)
    with col2:
        y_axis = st.selectbox("Axe Y", df.columns, index=2)
    fig_scatter = visualizer.get_interactive_scatter(x_axis, y_axis)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Modélisation
elif menu == "Modélisation":
    st_lottie(lottie_ml, height=120, key="ml")
    st.header("🤖 Modélisation du Risque")
    st.write("Sélectionnez les paramètres pour entraîner le modèle :")
    features = st.multiselect("Variables explicatives", [col for col in df.columns if col != "Risk"],
                              default=[col for col in df.columns if col != "Risk"])
    test_size = st.slider("Taille du jeu de test (%)", 10, 50, 20)
    random_state = st.slider("Random State", 0, 100, 42)
    st.markdown("---")
    if st.button("Entraîner le modèle", type="primary"):
        risk_model = RiskModel(task_type)
        acc, cm, report, mse, r2 = risk_model.train(df, features, test_size, random_state)
        st.session_state['risk_model'] = risk_model
        if task_type == 'classification':
            st.success(f"Précision du modèle : {acc:.2%}")
            st.subheader("Matrice de confusion")
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)
            st.subheader("Rapport de classification")
            st.text(report)
        else:
            st.success(f"MSE : {mse:.2f} | R² : {r2:.2%}")
        st.balloons()
        st.success("Modèle entraîné et prêt pour la prédiction personnalisée !")

# Prédiction personnalisée
elif menu == "Prédiction personnalisée":
    st_lottie(lottie_ml, height=120, key="ml_pred_page")
    st.header("🔮 Prédiction personnalisée du risque")
    risk_model = st.session_state.get('risk_model', None)
    if not risk_model:
        st.warning("Veuillez d'abord entraîner un modèle dans la section 'Modélisation'.")
    else:
        input_data = {}
        for feat in risk_model.features:
            val = st.number_input(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()), key=f"custom_{feat}")
            input_data[feat] = val
        if st.button("Prédire le risque", type="primary", key="predict_custom"):
            pred = risk_model.predict(input_data)
            if risk_model.task_type == "classification":
                st.markdown(f"<h2 style='color:#4F8BF9;'>Résultat : {'<span style=\"color:red;\">Risque élevé</span>' if pred else '<span style=\"color:green;\">Risque faible</span>'} ({pred})</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color:#4F8BF9;'>Valeur prédite : <span style='color:orange;'>{pred:.2f}</span></h2>", unsafe_allow_html=True)
            st_lottie(lottie_ml, height=100, key="ml_pred_result")
