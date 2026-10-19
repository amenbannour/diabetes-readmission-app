import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# Chargement du modèle et du préprocesseur
# --------------------------------------------------

preprocessor = joblib.load("preprocessor.pkl")

model = XGBClassifier()
model.load_model("xgb_readmission_model.json")

# --------------------------------------------------
# Titre de l'application
# --------------------------------------------------

st.title("🏥 Prédiction de la réadmission hospitalière")

st.write(
    "Cette application estime le risque de réadmission d'un patient "
    "diabétique dans les 30 jours suivant sa sortie."
)

st.divider()

# --------------------------------------------------
# Interface utilisateur
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.selectbox(
        "Tranche d'âge",
        [
            "[0-10)",
            "[10-20)",
            "[20-30)",
            "[30-40)",
            "[40-50)",
            "[50-60)",
            "[60-70)",
            "[70-80)",
            "[80-90)",
            "[90-100)"
        ],
        index=6
    )

    gender = st.selectbox(
        "Genre",
        [
            "Female",
            "Male"
        ]
    )

    race = st.selectbox(
        "Race",
        [
            "Caucasian",
            "AfricanAmerican",
            "Hispanic",
            "Asian",
            "Other",
            "Unknown"
        ]
    )

    admission_type_id = st.number_input(
        "Type d'admission ID",
        min_value=1,
        max_value=8,
        value=1
    )

    admission_source_id = st.number_input(
        "Source d'admission ID",
        min_value=1,
        max_value=25,
        value=7
    )

    time_in_hospital = st.slider(
        "Durée d'hospitalisation (jours)",
        min_value=1,
        max_value=14,
        value=4
    )

    medical_specialty = st.text_input(
        "Spécialité médicale",
        value="Unknown"
    )

    num_lab_procedures = st.number_input(
        "Nombre de procédures laboratoire",
        min_value=0,
        value=40
    )

    num_procedures = st.number_input(
        "Nombre de procédures",
        min_value=0,
        value=1
    )

    num_medications = st.number_input(
        "Nombre de médicaments",
        min_value=0,
        value=15
    )

    number_outpatient = st.number_input(
        "Consultations ambulatoires précédentes",
        min_value=0,
        value=0
    )

    number_emergency = st.number_input(
        "Passages aux urgences précédents",
        min_value=0,
        value=0
    )

    number_inpatient = st.number_input(
        "Hospitalisations précédentes",
        min_value=0,
        value=0
    )

with col2:

    number_diagnoses = st.number_input(
        "Nombre de diagnostics",
        min_value=1,
        value=8
    )

    max_glu_serum = st.selectbox(
        "Maximum glucose serum",
        [
            "None",
            "Norm",
            ">200",
            ">300"
        ]
    )

    A1Cresult = st.selectbox(
        "Résultat HbA1c",
        [
            "None",
            "Norm",
            ">7",
            ">8"
        ]
    )

    insulin = st.selectbox(
        "Insuline",
        [
            "No",
            "Steady",
            "Up",
            "Down"
        ]
    )

    metformin = st.selectbox(
        "Metformin",
        [
            "No",
            "Steady",
            "Up",
            "Down"
        ]
    )

    glipizide = st.selectbox(
        "Glipizide",
        [
            "No",
            "Steady",
            "Up",
            "Down"
        ]
    )

    glyburide = st.selectbox(
        "Glyburide",
        [
            "No",
            "Steady",
            "Up",
            "Down"
        ]
    )

    change = st.selectbox(
        "Modification du traitement",
        [
            "No",
            "Ch"
        ]
    )

    diabetesMed = st.selectbox(
        "Traitement antidiabétique",
        [
            "Yes",
            "No"
        ]
    )

    diag_1_group = st.selectbox(
        "Diagnostic principal",
        [
            "Circulatory",
            "Respiratory",
            "Digestive",
            "Diabetes",
            "Injury",
            "Musculoskeletal",
            "Genitourinary",
            "Neoplasms",
            "Other",
            "Unknown"
        ]
    )

    diag_2_group = st.selectbox(
        "Diagnostic secondaire",
        [
            "Circulatory",
            "Respiratory",
            "Digestive",
            "Diabetes",
            "Injury",
            "Musculoskeletal",
            "Genitourinary",
            "Neoplasms",
            "Other",
            "Unknown"
        ]
    )

    diag_3_group = st.selectbox(
        "Diagnostic complémentaire",
        [
            "Circulatory",
            "Respiratory",
            "Digestive",
            "Diabetes",
            "Injury",
            "Musculoskeletal",
            "Genitourinary",
            "Neoplasms",
            "Other",
            "Unknown"
        ]
    )

# --------------------------------------------------
# Création du DataFrame patient
# --------------------------------------------------

patient = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "race": race,
    "admission_type_id": admission_type_id,
    "admission_source_id": admission_source_id,
    "time_in_hospital": time_in_hospital,
    "medical_specialty": medical_specialty,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "max_glu_serum": max_glu_serum,
    "A1Cresult": A1Cresult,
    "insulin": insulin,
    "metformin": metformin,
    "glipizide": glipizide,
    "glyburide": glyburide,
    "change": change,
    "diabetesMed": diabetesMed,
    "diag_1_group": diag_1_group,
    "diag_2_group": diag_2_group,
    "diag_3_group": diag_3_group
}])

# --------------------------------------------------
# Prédiction
# --------------------------------------------------

st.divider()

if st.button(
    "Analyser le risque de réadmission",
    type="primary",
    use_container_width=True
):

    patient_encoded = preprocessor.transform(patient)

    probability = model.predict_proba(
        patient_encoded
    )[0][1]

    # seuil final retenu
    threshold = 0.50

    prediction = int(
        probability >= threshold
    )

    st.subheader("Résultat")

    st.metric(
        "Probabilité estimée de réadmission < 30 jours",
        f"{probability * 100:.1f} %"
    )

    if prediction == 1:

        st.error(
            "⚠️ Patient identifié comme présentant "
            "un risque de réadmission dans les 30 jours."
        )

    else:

        st.success(
            "✅ Patient non identifié comme présentant "
            "un risque élevé de réadmission dans les 30 jours."
        )

    st.caption(
        "Cette prédiction est issue d'un modèle de Machine Learning "
        "et ne constitue pas une décision médicale."
    )

# --------------------------------------------------
# Informations sur le modèle
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ Informations sur le modèle"):

    st.write(
        """
        **Modèle utilisé :** XGBoost Classifier

        **Type de problème :** Classification binaire

        **Target :**
        - 0 : Pas de réadmission dans les 30 jours
        - 1 : Réadmission dans les 30 jours

        **Seuil de décision :** 0.50

        **Performances du modèle :**
        - Accuracy : environ 66 %
        - Recall classe 1 : environ 55 %
        - F1-score classe 1 : environ 27 %
        - ROC-AUC : environ 0.66
        - PR-AUC : environ 0.21
        """
    )