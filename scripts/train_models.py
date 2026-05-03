#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import re, string

# Pulisce il testo: minuscole, rimuove URL, punteggiatura, spazi extra.
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s
# Salva una matrice di confusione come immagine.
def plot_confusion(cm, labels, out_path):
    fig = plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

# Esegue il training dei modelli per categoria e priorità, salva i modelli e genera report metriche e matrici di confusione.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/tickets_synth.csv",
                        help="Path al file CSV del dataset sintetico")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory per salvare i modelli")
    parser.add_argument("--reports_dir", type=str, default="reports",
                        help="Directory per salvare report e immagini")
    args = parser.parse_args()

    data_path = Path(args.data)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Carica dataset e prepara testo
    df = pd.read_csv(data_path, engine="python")
    df["text"] = (df["title"].astype(str) + " " + df["body"].astype(str)).map(clean_text)

    # Split dati con stratificazione sulla categoria
    X_train, X_test, ycat_train, ycat_test, ypri_train, ypri_test = train_test_split(
        df["text"], df["category"], df["priority"],
        test_size=0.2, random_state=42, stratify=df["category"]
    )

    # Vectorizer TF-IDF con n-grammi (1,2) e rimozione termini rari (min_df=2)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # Regressioni logistiche per categoria (bilanciata) e priorità
    clf_category = LogisticRegression(max_iter=200, n_jobs=1, class_weight="balanced")
    clf_priority = LogisticRegression(max_iter=200, n_jobs=1)
    clf_category.fit(Xtr, ycat_train)
    clf_priority.fit(Xtr, ypri_train)

    # Salva i modelli
    joblib.dump(vectorizer, models_dir / "vectorizer.joblib")
    joblib.dump(clf_category, models_dir / "clf_category.joblib")
    joblib.dump(clf_priority, models_dir / "clf_priority.joblib")

    # Valutazione modelli
    ycat_pred = clf_category.predict(Xte)
    ypri_pred = clf_priority.predict(Xte)
    acc_cat = accuracy_score(ycat_test, ycat_pred)
    f1_cat = f1_score(ycat_test, ycat_pred, average="macro")
    acc_pri = accuracy_score(ypri_test, ypri_pred)
    f1_pri = f1_score(ypri_test, ypri_pred, average="macro")

    # Salva metriche in file di testo
    with open(reports_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Category — accuracy: {acc_cat:.3f}, F1 macro: {f1_cat:.3f}\n")
        f.write(f"Priority — accuracy: {acc_pri:.3f}, F1 macro: {f1_pri:.3f}\n")

    # Calcola e salva matrici di confusione
    categories = sorted(df['category'].unique())
    priorities = ["bassa", "media", "alta"]
    cm_cat = confusion_matrix(ycat_test, ycat_pred, labels=categories)
    cm_pri = confusion_matrix(ypri_test, ypri_pred, labels=priorities)
    plot_confusion(cm_cat, categories, reports_dir / "confusion_category.png")
    plot_confusion(cm_pri, priorities, reports_dir / "confusion_priority.png")

    print("Saved models and reports. Metrics:")
    print((reports_dir / "metrics.txt").read_text())

if __name__ == "__main__":
    main()
