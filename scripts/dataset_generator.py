#!/usr/bin/env python3

import argparse
import random
import csv
from pathlib import Path

CATEGORIES = ["Amministrazione", "Tecnico", "Commerciale"]
PRIORITIES = ["bassa", "media", "alta"]

# Lessico specifico per categoria
LEXICON = {
    "Amministrazione": [
        "fattura", "pagamento", "scadenza", "bonifico", "rimborso", "IVA",
        "estratto conto", "doppio addebito", "preventivo", "ordine", "nota di credito"
    ],
    "Tecnico": [
        "errore", "bug", "bloccante", "server", "timeout", "login", "crash",
        "aggiornamento", "API", "database", "rete", "installazione", "ticket", "sistema"
    ],
    "Commerciale": [
        "offerta", "sconto", "listino", "contratto", "ordine", "preventivo",
        "consegna", "disponibilità", "proposta", "demo", "assistenza commerciale"
    ],
}

# Parole chiave indicative di priorità
PRIORITY_HINTS = {
    "alta": ["bloccante", "urgente", "impossibile", "fermi", "critico", "non funziona"],
    "media": ["ritardo", "verifica", "sollecito", "anomalia", "incongruenza"],
    "bassa": ["informazione", "richiesta dettagli", "chiarimento", "domanda"]
}

# Titoli tipici per categoria
TITLES = {
    "Amministrazione": [
        "Problema pagamento fattura", "Richiesta nota di credito", "Doppio addebito su carta",
        "Chiarimenti su IVA e scadenze", "Errore importo preventivo"
    ],
    "Tecnico": [
        "Errore login dopo aggiornamento", "API in timeout", "Crash applicazione su avvio",
        "Impossibile accedere al server", "Bug su form di registrazione"
    ],
    "Commerciale": [
        "Richiesta offerta personalizzata", "Disponibilità prodotto e tempi di consegna",
        "Sconto su ordine ricorrente", "Domande su contratto", "Richiesta demo piattaforma"
    ],
}

# Frasi extra per il corpo del ticket
BODIES_EXTRA = [
    "Potete verificare?",
    "In allegato screenshot dell'errore.",
    "Serve una risposta entro oggi.",
    "Il problema si presenta su più utenti.",
    "Abbiamo già aperto un ticket in passato.",
    "Richiedo cortese riscontro.",
    "Se necessario possiamo fare una call."
]

def random_priority(category_words):
    """
    Assegna priorità basata sulle parole chiave presenti.
    Restituisce alta, bassa o media a seconda delle hint trovate.
    """
    text = " ".join(category_words).lower()
    if any(k in text for k in PRIORITY_HINTS["alta"]):
        return "alta"
    if any(k in text for k in PRIORITY_HINTS["bassa"]):
        return "bassa"
    return "media"

def synth_ticket():
    """
    Genera un singolo ticket sintetico con titolo, corpo, categoria e priorità.
    Il corpo contiene da 1 a 3 parole chiave, eventualmente con hint di priorità.
    """
    cat = random.choice(CATEGORIES)
    title = random.choice(TITLES[cat])
    # Scegliere fino a 3 parole chiave per categoria
    words = random.sample(LEXICON[cat], k=min(3, len(LEXICON[cat])))

    # Con probabilità del 35%, inserisce un indizio di priorità alta o bassa
    if random.random() < 0.35:
        if random.random() < 0.6:
            words.append(random.choice(PRIORITY_HINTS["alta"]))
        else:
            words.append(random.choice(PRIORITY_HINTS["bassa"]))

    # Corpo testo finale
    body = f"{title}. {' '.join(words)}. {random.choice(BODIES_EXTRA)}"
    prio = random_priority(words)
    return title, body, cat, prio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400, help="Numero di ticket da generare (200-500 consigliati)")
    parser.add_argument("--out", type=str, default="data/tickets_synth.csv", help="File di output CSV")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "body", "category", "priority"])
        for i in range(args.n):
            title, body, cat, prio = synth_ticket()
            writer.writerow([i+1, title, body, cat, prio])

    print(f"Scritto: {out_path}")

if __name__ == "__main__":
    main()

