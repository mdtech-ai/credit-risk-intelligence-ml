# 1 Bon (Faible risque)   0 Mauvais (Risque élevé) 

# Alors qu'ici, j'ai: 
# Bon        291   /  0    291
# Mauvais    231   /  1    231

import streamlit as st 
import pandas as pd 
import joblib 

SEUIL = 0.34   # seuil retenu dans le notebook (à placer en haut du fichier)

# Chargement du modèle et des encodeurs 
model = joblib.load("xgb_credit_model.pkl")
encodeurs = {col: joblib.load(f"{col}_encoder.pkl") for col in ["sexe", "logement", "compte_epargne", "compte_courant"]}

target_encoder = joblib.load("target_encoder.pkl")


# Initialisation de l'état de session
if "resultat" not in st.session_state:
    st.session_state.resultat = None

st.title("Application de prédiction du risque de crédit")
st.write("Entrer les informations du demandeur pour prédire si le risque de crédit est bon ou mauvais")

# --------------------------------------------------
# 1. FORMULAIRE DE SAISIE
# --------------------------------------------------

with st.form("formulaire_credit"):

    st.subheader("Informations du demandeur")

    age = st.number_input(
        "Âge",
        min_value=19,
        max_value=75,
        value=30
    )

    sexe = st.selectbox(
        "Sexe",
        encodeurs["sexe"].classes_
    )

    EMPLOIS = {"Sans emploi / non qualifié non résident": 0, "Non qualifié résident": 1,
           "Qualifié": 2, "Très qualifié / cadre": 3}
    
    emploi_label = st.selectbox("Qualification de l'emploi", list(EMPLOIS))

    logement = st.selectbox(
        "Logement",
        encodeurs["logement"].classes_
    )

    ORDRE_EPARGNE = ["Faible", "Modérée", "Assez élevée", "Élevée"]
    ORDRE_COURANT = ["Faible", "Modéré", "Élevé"]

    compte_epargne = st.selectbox("Compte épargne",
        [o for o in ORDRE_EPARGNE if o in encodeurs["compte_epargne"].classes_])
    
    compte_courant = st.selectbox("Compte courant",
        [o for o in ORDRE_COURANT if o in encodeurs["compte_courant"].classes_])


    montant_credit = st.number_input(
        "Montant du crédit (CFA)",
        min_value=85_000,
        value=800_000
    )

    duree = st.number_input(
        "Durée (mois)",
        min_value=6,
        max_value=72, 
        value=18
    )

    submitted = st.form_submit_button(
        "Prédire le risque",
        use_container_width=True
    )


# --------------------------------------------------
# 2. TRAITEMENT APRÈS SOUMISSION
# --------------------------------------------------

if submitted:

    input_df = pd.DataFrame({
        "age": [age],
        "sexe": [encodeurs["sexe"].transform([sexe])[0]],
        "qualification_emploi": [EMPLOIS[emploi_label]],
        "logement": [encodeurs["logement"].transform([logement])[0]],
        "compte_epargne": [encodeurs["compte_epargne"].transform([compte_epargne])[0]],
        "compte_courant": [encodeurs["compte_courant"].transform([compte_courant])[0]],
        "montant_credit": [montant_credit],
        "duree": [duree]
    })

    # Respecter l'ordre des variables utilisé à l'entraînement
    if hasattr(model, "feature_names_in_"):
        input_df = input_df[model.feature_names_in_]

    # Prédiction
    pred = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    # Identification des classes
    idx_good = list(model.classes_).index(
        target_encoder.transform(["Bon"])[0]
    )

    idx_bad = list(model.classes_).index(
        target_encoder.transform(["Mauvais"])[0]
    )

    proba_good = probabilities[idx_good]
    proba_bad = probabilities[idx_bad]


    # dans le bloc "if submitted:", à la place de pred = model.predict(...)
    mauvais = bool(proba_bad >= SEUIL)


    # Sauvegarde des résultats
    st.session_state.resultat = {
        "mauvais": mauvais,
        "proba_good": proba_good,
        "proba_bad": proba_bad
    }


# --------------------------------------------------
# 3. AFFICHAGE DES RÉSULTATS
# --------------------------------------------------

if st.session_state.resultat is not None:

    resultat = st.session_state.resultat

    st.divider()

    st.subheader("Résultat de l'évaluation")

    # dans l'affichage
    if not resultat["mauvais"]:
        st.success(f"Profil plutôt favorable : probabilité de défaut sous le seuil de {SEUIL:.0%}.")
    else:
        st.error(f"Profil à examiner : probabilité de défaut au-dessus du seuil de {SEUIL:.0%}.")

    st.progress(float(resultat["proba_bad"]),
            text=f"Probabilité estimée de défaut : {resultat['proba_bad']:.0%}")
    st.caption("Estimation indicative issue d'un modèle pédagogique : ce n'est pas une décision de crédit. "
           "Une décision réelle suppose des critères métier, des coûts et une validation.")


    st.subheader("Probabilités estimées")

    col1, col2 = st.columns(2)

    col1.metric(
        "Bon risque",
        f"{resultat['proba_good']:.1%}"
    )

    col2.metric(
        "Mauvais risque",
        f"{resultat['proba_bad']:.1%}"
    )