#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path
import joblib
import re, string

 # Pulisce il testo: minuscole, rimuove URL, punteggiatura, spazi extra.
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Esegue la predizione batch su file CSV di input e salva il file con le predizioni.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="CSV con colonne: id, title, body (category/priority opzionali)")
    parser.add_argument("--output", type=str, required=True,
                        help="CSV di output con predizioni aggiunte")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory contenente i modelli salvati")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    # Caricamento modelli
    vec = joblib.load(models_dir / "vectorizer.joblib")
    clf_cat = joblib.load(models_dir / "clf_category.joblib")
    clf_pri = joblib.load(models_dir / "clf_priority.joblib")

    # Caricamento dati input
    df = pd.read_csv(args.input)
    # Unione e pulizia testo
    df["text"] = (df["title"].astype(str) + " " + df["body"].astype(str)).map(clean_text)

    # Trasformazione testo e predizione
    X = vec.transform(df["text"])
    df["pred_category"] = clf_cat.predict(X)
    df["pred_priority"] = clf_pri.predict(X)

    # Salvataggio output
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"Scritto: {out}")

if __name__ == "__main__":
    main()
