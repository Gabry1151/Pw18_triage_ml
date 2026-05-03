import streamlit as st
import pandas as pd
import joblib
import re, string
import numpy as np
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Triage Ticket ML — PW18", layout="centered")

# Pulisce testo: minuscole, rimuove URL, punteggiatura e spazi extra.
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Carica modelli salvati da disk.
@st.cache_resource
def load_models():
    vec = joblib.load("models/vectorizer.joblib")
    clf_cat = joblib.load("models/clf_category.joblib")
    clf_pri = joblib.load("models/clf_priority.joblib")
    return vec, clf_cat, clf_pri

# Restituisce la classe predetta, le probabilità e le parole più influenti nel testo dato.
def top_influential_words(vec, clf, text, k=5):
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else None
    pred = clf.predict(X)[0]
    classes = list(clf.classes_)
    ci = classes.index(pred)

    if hasattr(clf, "coef_"):
        inds = X.nonzero()[1]
        coefs = clf.coef_[ci, inds]
        feats = np.array(vec.get_feature_names_out())[inds]
        order = np.argsort(-coefs)  # ordina decrescente
        influential = [(feats[i], float(coefs[i])) for i in order[:k]]
    else:
        influential = []

    return pred, proba, influential
    
# Grafica orizzontale delle probabilità per ciascuna classe.
def plot_probabilities(probs, classes, title):
    fig, ax = plt.subplots()
    ax.barh(classes, probs, color='skyblue')
    ax.set_xlabel('Probabilità')
    ax.set_title(title)
    return fig

# Genera il CSV delle predizioni batch e ritorna filename + byte string.
def create_batch_predictions_csv(df: pd.DataFrame) -> tuple[str, bytes]:
    export_df = df[['title', 'body', 'categoria', 'priorita']].copy()
    csv_str = export_df.to_csv(index=False)
    return "predizioni_ticket.csv", csv_str.encode("utf-8")

vec, clf_cat, clf_pri = load_models()

st.title("Triage automatico dei ticket")

tab1, tab2 = st.tabs(["Predizione singola", "Predizione batch da file"])

with tab1:
    st.write("Inserisci **titolo** e **descrizione**, poi ottieni **categoria** e **priorità** suggerite.")
    title = st.text_input("Titolo", "Errore login dopo aggiornamento")
    body = st.text_area("Descrizione", "Dopo l'update ricevo errore e non riesco ad accedere. In allegato screenshot. Serve risposta oggi.")

    if st.button("Classifica"):
        text = f"{title} {body}"
        cat, pcat, infl_cat = top_influential_words(vec, clf_cat, clean_text(text), k=5)
        pri, ppri, infl_pri = top_influential_words(vec, clf_pri, clean_text(text), k=5)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Categoria")
            st.subheader(cat)
            if pcat is not None:
                fig_cat = plot_probabilities(pcat, clf_cat.classes_, "Probabilità Categoria")
                st.pyplot(fig_cat)
            if infl_cat:
                st.write("Parole più influenti (cat):", ", ".join([w for w, _ in infl_cat]))

        with col2:
            st.header("Priorità")
            st.subheader(pri)
            if ppri is not None:
                fig_pri = plot_probabilities(ppri, clf_pri.classes_, "Probabilità Priorità")
                st.pyplot(fig_pri)
            if infl_pri:
                st.write("Parole più influenti (prio):", ", ".join([w for w, _ in infl_pri]))

with tab2:
    st.write("Carica un file CSV con colonne 'title' e 'body' per fare predizioni in batch.")
    
    # Sezione per il download di file esistenti (Template o risultati precedenti)
    col_down1, col_down2 = st.columns(2)
    with col_down1:
        if os.path.exists("data/tickets_synth.csv"):
            with open("data/tickets_synth.csv", "rb") as f:
                st.download_button("Scarica file CSV sintetico", f, file_name="template_tickets.csv", mime="text/csv")
    
    with col_down2:
        if os.path.exists("data/predictions.csv"):
            with open("data/predictions.csv", "rb") as f:
                st.download_button("Scarica file batch prediction", f, file_name="predictions.csv", mime="text/csv")
        else:
            st.info("Nessun file di batch prediction trovabile in data/predictions.csv.")

    uploaded_file = st.file_uploader("Carica CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'title' in df.columns and 'body' in df.columns:
            df['text'] = df['title'].astype(str) + " " + df['body'].astype(str)
            df['clean_text'] = df['text'].apply(clean_text)
            df['categoria'] = df['clean_text'].apply(lambda x: clf_cat.predict(vec.transform([x]))[0])
            df['priorita'] = df['clean_text'].apply(lambda x: clf_pri.predict(vec.transform([x]))[0])
            st.write("Risultati predizione batch:")
            st.dataframe(df[['title', 'body', 'categoria', 'priorita']])
            file_name, csv_bytes = create_batch_predictions_csv(df)
            st.download_button("Scarica risultati CSV", csv_bytes, file_name=file_name, mime="text/csv")
        else:
            st.error("Il file deve contenere le colonne 'title' e 'body'.")

    st.caption("PW18 — prototipo con funzionalità di predizione singola, batch e visualizzazione grafica.")
