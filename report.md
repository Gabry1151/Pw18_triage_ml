# PW18 - Report progetto triage automatico ticket con ML

Il progetto nasce dalla necessità di automatizzare la gestione dei ticket aziendali, che arrivano quotidianamente tramite e-mail o moduli online. Ogni ticket, composto da un titolo e una descrizione, deve essere classificato in una specifica categoria (Amministrazione, Tecnico o Commerciale) e assegnato a una priorità che può essere bassa, media o alta.

Il flusso prevede che, una volta ricevuto il ticket, il sistema applichi un modello di machine learning per identificare automaticamente la categoria e la priorità. Successivamente i ticket vengono smistati nei reparti corrispondenti, dove gli operatori li gestiscono ordinandoli secondo la priorità assegnata, così da ottimizzare tempi e risorse. Questo processo mira a rendere più rapido e preciso il triage, riducendo il carico sul personale e migliorando il servizio di assistenza.

Per il modello sono stati utilizzati due algoritmi di **Regressione Logistica** indipendenti: uno dedicato alla classificazione della categoria e l'altro alla classificazione della priorità.

La regressione logistica è stata scelta per le seguenti ragioni:

- **Semplicità ed efficienza:** è un algoritmo leggero e veloce da addestrare, ideale per un prototipo con dataset sintetico di medie dimensioni.
- **Interpretabilità:** i coefficienti associati alle feature permettono di identificare le parole più influenti nella decisione, facilitando l’analisi e il debugging.
- **Efficacia su dati testuali con TF-IDF:** in presenza di una buona rappresentazione TF-IDF, la regressione logistica spesso raggiunge ottime performance su compiti di classificazione testuale.

La rappresentazione testuale è stata effettuata tramite TF-IDF con n-grammi di ordine 1 e 2, in modo da cogliere sia parole singole sia brevi sequenze di termini, migliorando la capacità predittiva.

Questa scelta tecnica bilancia quindi la necessità di un sistema rapido, interpretabile e riproducibile con le caratteristiche del dataset sintetico.

L'addestramento si è svolto con una divisione train/test 80/20 stratificata, con parametri impostati per assicurare la convergenza e un bilanciamento di peso delle classi nella classificazione delle categorie.

I risultati mostrano una classificazione della categoria praticamente perfetta, con accuracy e F1 macro vicine a 1.0, segno che il modello coglie efficacemente i segnali distintivi nel dataset sintetico.

Per la priorità, le metriche sono più modeste: accuracy intorno al 60% e F1 macro di 0.40, indicativi della maggiore complessità di predire la priorità tramite segnali lessicali meno netti.

Sono state generate matrici di confusione per categoria e priorità, incluse nel progetto come immagini, che rappresentano visivamente i risultati e aiutano a identificare errori sistematici.

Nella dashboard interattiva, è stata implementata una funzionalità dedicata per eseguire predizioni batch su un insieme di ticket caricati tramite file CSV. Questo è realizzato nel secondo tab della dashboard.

L'utente può caricare un file CSV contenente le colonne obbligatorie `title` e `body`, che rappresentano rispettivamente il titolo e la descrizione di ogni ticket. Una volta caricato, il sistema concatena e pulisce il testo di ogni ticket attraverso la funzione di preprocessing, eliminando URL, punteggiatura e normalizzando i caratteri in minuscolo.

Ogni testo pulito viene poi trasformato in vettori TF-IDF usando il modello di vettorizzazione precedentemente addestrato. Successivamente, i due modelli di regressione logistica, uno per la categoria e uno per la priorità, vengono utilizzati per predire le etichette corrispondenti a ogni ticket in batch.

I risultati vengono mostrati in tabella nella dashboard, permettendo all'utente di esaminare titolo, descrizione e relative predizioni di categoria e priorità. Infine, è possibile scaricare un file CSV contenente tutte le informazioni e le predizioni per utilizzo esterno o report.

Questa funzionalità consente di processare in modo efficiente grandi volumi di ticket, integrando facilmente il sistema in flussi di lavoro aziendali reali senza necessità di input manuale singolo.

Il progetto presenta alcune limitazioni importanti:

- L’utilizzo di un dataset sintetico, seppur utile per un prototipo, non interpreta completamente la complessità linguistica dei ticket reali, limitando la generalizzabilità.
- Il preprocessing è molto basilare: non sono state utilizzate tecniche come rimozione approfondita di stopword specifiche, lemmatizzazione o stemming.
- Il modello di classificazione della priorità si basa su segnali lessicali intermittenti e regole semplici, che possono risultare poco robusti in contesti reali.

Possibili miglioramenti futuri includono:

- Integrare preprocessing più avanzato con stopword italiane, lemmatizzazione e tecniche di pulizia del testo più sofisticate.
- Addestrare e testare il sistema su dati reali anonimizzati per aumentare la validità e robustezza.
- Sperimentare modelli ML complessi come reti neurali, transformer o modelli pretrained NLP.
- Implementare soglie probabilistiche dinamiche per ottimizzare l’assegnazione della priorità.
- Arricchire la dashboard con funzionalità di feedback, analisi errori e visualizzazioni più approfondite.
- Utilizzo di tecniche di data augmentation o oversampling per bilanciare meglio le classi.
- Collegamento con sistemi di ticketing reali per test in ambienti produttivi.
- Migliorare la gestione delle dipendenze e la documentazione tecnica per facilitare l’uso e la manutenzione.


