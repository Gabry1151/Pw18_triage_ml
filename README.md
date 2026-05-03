# PW18 — Triage automatico dei ticket con ML

Un prototipo minimale per il triage automatico di ticket in tre categorie (Amministrazione, Tecnico, Commerciale) e stima della priorità (bassa, media, alta) usando machine learning.

---

## Sommario
- [Struttura](#struttura)
- [Requisiti](#requisiti)
- [Come riprodurre](#come-riprodurre)
- [Output attesi](#output-attesi)
- [Tecnologie utilizzate](#tecnologie-utilizzate)
- [Problemi comuni e soluzioni](#problemi-comuni-e-soluzioni)

---

## Struttura

```
pw18_triage_ml/
├── data/
│   ├── tickets_synth.csv
│   └── predictions.csv
├── models/
│   ├── vectorizer.joblib
│   ├── clf_category.joblib
│   └── clf_priority.joblib
├── reports/
│   ├── metrics.txt
│   ├── confusion_category.png
│   └── confusion_priority.png
├── scripts/
│   ├── dataset_generator.py
│   ├── train_models.py
│   └── batch_predict.py
├── dashboard/
│   └── dashboard.py
├── requirements.txt
├── README.md
└── report.md
```

---

## Requisiti

- Python 3.10+
- Installare dipendenze:
`pip install -r requirements.txt`

## Come riprodurre

1. **Genera il dataset sintetico** (200–500 ticket):
   `python scripts/dataset_generator.py --n 400`
   
2. **Addestra i modelli** e salva metriche + grafici:
   `python scripts/train_models.py`

3. **Esegui predizioni in batch** su `data/tickets_synth.csv` (o su un CSV tuo con colonne: `id,title,body`):
   `python scripts/batch_predict.py --input data/tickets_synth.csv --output data/predictions.csv`

4. **Avvia la dashboard** (Streamlit) per provare il modello su singoli ticket:
   `streamlit run dashboard/dashboard.py`

---

## Output attesi

- File `predictions.csv` con categorie e priorità predette
- Metriche di valutazione nei file di testo e matrici di confusione come immagini nella cartella `reports/`
- Dashboard interattiva per esplorare dati e risultati, con possibilita' di download delle predizioni batch dalla dashboard in file CSV

---

## Tecnologie utilizzate

- Python 3.10+
- Scikit-learn per ML
- Joblib per serializzazione modelli
- Streamlit per dashboard

---

## Problemi comuni e soluzioni

- Errore versioni Python: usare Python 3.10+
- Dipendenze mancanti: reinstallare con `pip install -r requirements.txt`
- File input non trovato: verificare percorso o creare dataset sintetico
- Streamlit non trovato: se installato solo nell'ambiente virtuale, usare `python -m streamlit run dashboard/dashboard.py` invece di `streamlit run dashboard/dashboard.py`
