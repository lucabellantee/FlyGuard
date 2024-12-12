# Parametri Utilizzati nei Sistemi di Data-Driven Fault Detection per Motori a Reazione di Elicotteri

## Descrizione delle Grandezze

### 1. `id`
- **Descrizione**: Identificativo univoco per ciascun campione o sessione di dati.
- **Uso**: Serve per tracciare i dati o associare un determinato set di misurazioni a un motore o a un volo specifico.

### 2. `trq_measured` (Torque Misurato)
- **Descrizione**: Coppia misurata generata dal motore.
- **Unità di misura internazionale**: Newton-Metro(Nm)
- **Uso**: Indica la potenza trasmessa dal motore al rotore; varia in base alle condizioni operative e alla richiesta di potenza.

### 3. `oat` (Outside Air Temperature)
- **Descrizione**: Temperatura dell'aria esterna.
- **Unità di misura internazionale**: Gradi Celsius(C°)
- **Uso**: Influisce sulle prestazioni del motore e sulle condizioni di combustione. È un parametro chiave per correggere i valori misurati.

### 4. `mgt` (Measured Gas Temperature)
- **Descrizione**: Temperatura dei gas di scarico misurata (o temperatura del gas principale).
- **Unità di misura internazionale**: Gradi Celsius 
- **Uso**: Monitorare il calore generato dalla combustione. Anomalie in questa misura possono indicare problemi di combustione o di sovraccarico del motore.

### 5. `pa` (Pressure Altitude)
- **Descrizione**: Altitudine basata sulla pressione atmosferica misurata. Corrisponde all'altitudine del veivolo quando l'altimetro è settato su 29.92 inHg .
- **Unità di misura internazionale**: Feat(ft)
- **Uso**: Importante per valutare le prestazioni del motore in diverse condizioni atmosferiche e di altitudine.

### 6. `ias` (Indicated Airspeed)
- **Descrizione**: Velocità indicata dell'elicottero rispetto all'aria.
- **Unità di misura internazionale**: knots(kn)
- **Uso**: Aiuta a correlare le condizioni del motore con il comportamento aerodinamico dell'elicottero.

### 7. `np` (Power Turbine Speed)
- **Descrizione**: Velocità della turbina di potenza, misurata in giri al minuto (RPM).
- **Unità di misura internazionale**: Revolutions per minute(RPM)  ---> Dai dati, si osserva che non può essere questa l'unità di misura corretta. Probabilmente, sarà un suo multiplo.
- **Uso**: Indica la velocità con cui la turbina di potenza ruota. Fluttuazioni possono suggerire problemi meccanici o instabilità.

### 8. `ng` (Gas Generator Speed)
- **Descrizione**: Velocità del generatore di gas, misurata in RPM.
- **Unità di misura internazionale**: Revolutions per minute(RPM)
- **Uso**: Indica la velocità della turbina primaria del motore responsabile della generazione dei gas di scarico.

### 9. `faulty`
- **Descrizione**: Indicatore binario o categorico che identifica se il sistema è in stato di guasto (`1`) o funzionante correttamente (`0`).
- **Uso**: Variabile target per il sistema di fault detection.

### 10. `trq_margin` (Torque Margin)
- **Descrizione**: Margine disponibile tra la coppia attuale e il limite massimo di coppia consentito.
- **Formula**: torque margin (%) = 100 * (torque measured – torque target) / torque target . Questa variabile è dunque espressa in percentuale(%).
- **Uso**: Indica quanto margine rimane, in %, prima che il motore raggiunga il limite massimo di coppia. Può essere un indicatore chiave per evitare sovraccarichi o stress del motore.


