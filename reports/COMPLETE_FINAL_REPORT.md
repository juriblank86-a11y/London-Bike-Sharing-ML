# London Bike Rental Forecasting - Gesamtbericht mit Evolutionsanalyse
## Machine Learning für asymmetrische Business-Anforderungen
### Kontinuierliche Entwicklung vom Anfang bis zur Production-Ready Solution

**Projektdatum:** 12. Dezember 2025  
**Team:** René Lemke, Finn Hößler, Shiar Hido, Juri Blank  
**Datensatz:** Transport for London (TfL) Bike-Sharing Dataset (17.366 Stunden)  
**Zeitraum:** 04.01.2015 - 03.01.2017 (2 Jahre)  
**Status:** Production-Ready

**Aufgaben:**

| Name | Hauptaufgaben |
| -----| --------------------------------------------------------------------- |
| René | Feature Engineering (sin/cos), GridSearch, Hyperparameter-Tuning BayesRidgeRegression, Visualisierungen, Code-Dokumentation |
| Finn | Custom Scoring Function, OOF-Optimizer, Visualisierungen, Hyperparameter-Tuning RandomForestRegressor |
| Shiar| Post-Processing, Hyperparameter-Tuning KNeighborsRegressor, Visualisierungen, Datenanalyse |
| Juri | Systematische Modell-Evaluierung (13 Algorithmen), Code-Dokumentation, Hyperparameter-Tuning GradienBoostRegressorQ85, Visualisierungen, Datenanalyse |

---

##  Zusammenfassung

Dieses Projekt dokumentiert die **iterative Entwicklung eines hochoptimierten Machine-Learning-Systems** zur Vorhersage der stündlichen Nachfrage nach Leihfahrrädern in London, mit detaillierter Analyse des kontinuierlichen Lernprozesses über drei Phasen.

### Kernresultate:

| Metrik | Phase 1 | Phase 2 | Phase 3 (Final) | Status |
|--------|---------|---------|-----------------|--------|
| **Gewinner-Modell** | Random Forest | GradBoost (MSE) | **KNN Custom + PP** |  |
| **R² Score** | 0,932 | 0,927 | **0,9180** |  Exzellent |
| **RMSE (Fahrräder)** | 293,6 | 306 | **322,9** |  Gut |
| **Asymmetrischer Score** | 36,7% | 72,7% | **84,01%** |  Übertroffen |
| **Kritische Rate** | 46,4% | ~26% | **17,4%** |  Akzeptabel |
| **Business-Performance** |  Untauglich |  OK |  Production-Ready | **2,2× Verbesserung!** |

**Die zentrale Innovation:** Ein **Asymmetric Weighted Scoring Function**, der Unterversorgung (fehlende Räder = verlorene Kunden) 2,5× höher gewichtet als Überversorgung (extra Räder = managebar). Dies führte zu einer kontinuierlichen Verbesserung von **36,7% → 84,01%** – eine **2,2× Steigerung des Business-Scores**.

---

## 1. Aufgabenstellung und Geschäftlicher Kontext

### 1.1.1 Scenario

Der Auftraggeber ist ein Online-Fahrradverleih, der in der Stadt verteilt Fahrräder bereitstellt, die per App gebucht werden können.

### 1.1.2 Ziel der Untersuchung

Erstellung einer Lösung zur präzisen Ermittlung des stündlichen Bedarfs an Fahrrädern auf Basis von Wetter- und Kalenderdaten mit dem Ziel eine bedarfsgerechte Bereitstellung zu ermöglichen. 

### 1.1.3 Arbeits-Hypothese

Anhand von Jahreszeiten und Wetterdaten wird eine Prognose möglich sein. Bei Regen und besonders niedrigen Temperaturen werden vermutlich deutlich weniger Fahrräder geliehen. An Wochenenden und Feiertagen wird die Leihe für Freizeitaktivitäten besonders attraktiv sein.

### 1.1 Geschäftliches Problem (Supply-Chain-Optimierung)

Das London Bike-Sharing System muss jeden Tag hundertausende Fahrradvermietungen bewerkstelligen. Die zentrale Frage:

> **Wie viele Leihfahrräder sollten wir zu jeder Stunde an den richtigen Orten bereitstellen?**

**Zwei Fehlertypen, unterschiedliche Kosten:**

| Fehlertyp | Business-Impact | Kosten | Gewichtung |
|-----------|-----------------|--------|-----------|
| **Unterversorgung** | Kunden finden keine Räder, Umsatz verloren, Reputationsverlust | ~10€ pro fehlendem Rad |  **2,5×** |
| **Überversorgung** | Unnötige Wartungskosten, Parkplatzprobleme | ~2€ pro überschüssigem Rad |  **0,8×** |
| **Optimal** | Richtige Flottenmenge, zufriedene Kunden | 0€ |  **1.0** |

Die **asymmetrische Kostenstruktur** ist das Kernproblem: Unterversorgung ist typischerweise **5-10× teurer** als Überversorgung.

### 1.2 Methodische Innovation: Vom Standard- zum Business-fokussierten ML

**Klassisches ML-Denken:**
```
Trainiere auf Standard-Metrik (R²) → Hope, dass das Geschäft funktioniert
Resultat: ~50% der Vorhersagen verfehlen die Business-Anforderung
```

**Unser Ansatz:**
```
Codiere Business-Logik mathematisch in Loss-Funktion
Optimiere direkt auf Business-Metrik mit Custom Scoring
GridSearchCV sucht Parameter, die Geschäftserfolg maximieren
Resultat: 84% Business-Performance (vs. ~73% mit Standard-Optimierung)
```

### 1.3 Qualitätskriterien (Multi-Dimensionale Bewertung)

| Metrik | Typ | Berechnung | Geschäfts-Relevanz |
|:---|:---|:---|:---|
| **R² Score** | Statistisch | Erklärvarianz (0–1) | Akademischer Baseline-Vergleich |
| **MAE** | Fehler | Ø\|y−ŷ\| in Fahrrädern | Absolute Vorhersage-Güte |
| **MAPE** | Fehler (%) | Ø\|y−ŷ\|/y in % | Prozentuale Abweichung |
| **Asymmetric Weighted Score** | **Business-spezifisch** | Toleranzband −5% bis +20% mit asymmetrischer Gewichtung |  **PRIMÄRMETRIK** |

Folgende Metriken wurden für die Bewertung der Modellergebnisse definiert:
• R² Score (Primärmetrik): Zielmarke R² > 0,90 für ein praxistaugliches Modell.
• Mean Absolute Error (MAE): Durchschnittlicher absoluter Fehler in Fahrrädern pro Stunde.
• Root Mean Squared Error (RMSE): Wurzel der quadrierten Fehler, besonders empfindlich gegenüber Ausreißern.
• Symmetrischer Toleranz-Score: Anteil der Stunden, in denen die Vorhersage im Bereich ±15 % liegt.
• Asymmetrischer Business-Score: Anteil der Stunden mit −5 % bis +20 % Toleranz (asymmetrisch, da zu viele Räder akzeptabel sind, zu wenige aber kritisch).

Für den Auftraggeber ist es besonders wichtig, die Nachfrage bedienen zu können. Der Dienst soll für den Kunden verfügbar und zuverlässig wirken. Die Fehlertolleranz nach oben erlaubt eine Abweichung von 20%. Eine um bis 5% zu geringe Voraussage ist gerade noch akzeptabel. Für den Auftraggeber ist es wichtiger, mehr Fahrräder bereitzustellen als zu wenige

### 1.4 Roadmap

Das Projekt wurde in folgende Phasen strukturiert:
• Phase 1–2: Datenbereinigung und Feature Engineering.
• Phase 3–4: Explorative Datenanalyse (EDA) und Split-Strategie.
• Phase 5–6: Systematisches Training mit GridSearch, vier verschiedenen Skalern und acht Algorithmen.
• Phase 7: Ensemble-Methoden (Voting Regressor).
• Phase 8: Detaillierte Business-Analyse und Risikobewertung.

Daten teilen in Trainings- und Testmenge
Datenvorverarbeitung (Imputer? Scaler? Encoder?)
Algorithums testen (fit, predict für einige Werte, score?)
Auswertung des Algorithmus und ggf anpassen (Parameter?)
Leistung verbessern mit GridSearch evtl
Predicts aller Algorithmen zusammenfassen → Voting (welcher ist der beste?)
Testdaten vorbereiten (skalieren?) und predict darauf mit der besten Methode

---

## 2. Daten und Feature Engineering

### 2.1 Datenquelle und Bereinigung

- **Quelle:** Transport for London (TfL) Open Data API
- **Original-Umfang:** 17.414 stündliche Beobachtungen
- **Nach Bereinigung:** 17.366 Stunden (48 Zeilen entfernt)
- **Bereinigung:** U-Bahn-Streiks am 09.07.2015 und 06.08.2015 (atypische Nachfrage-Spikes)
- **Zeitraum:** 04.01.2015 - 03.01.2017 (exakt 2 Jahre), fehlende Stunden verteilt im gesamten Datensatz, welche für die Analyse keinen Einfluss auf unsere Ergebnisse haben
- **Datenqualität:** Keine fehlenden Werte, sehr sauber

### 2.2 Feature Engineering (10 → 17 Features)

**Ursprüngliche 10 Features:**
- Temperatur: `t1`, `t2`
- Wetter: `hum`, `wind_speed`, `weather_code`
- Temporal: `timestamp`, `is_holiday`, `is_weekend`, `season`, `cnt` (Target)

**Neu erzeugte 7 Temporal-Features:**

| Feature | Typ | Erklärung | Wertebereich |
|:---|:---|:---|:---:|
| `hour` | Integer | Stunde des Tages | 0–23 |
| `month` | Integer | Monat des Jahres | 1–12 |
| `day_of_week` | Integer | Wochentag (0=Mo, 6=So) | 0–6 |
| `hour_sin`, `hour_cos` | Float | Sinus/Cosinus-Kodierung (Zirkularität) | −1 bis +1 |
| `month_sin`, `month_cos` | Float | Saisonale Zirkularität | −1 bis +1 |

**Zyklische Kodierung (essentiell!):**
- Mathematisches Problem: Stunde 23 ist näher bei Stunde 0 als bei Stunde 12
- Sinus/Cosinus-Transformation macht diese geometrische Nähe explizit
- Resultat: KNN, neuronale Netze und Tree-Splits funktionieren besser

### 2.3 Explorative Datenanalyse: Kernmuster

**Tageszeit-Struktur (Pendler-dominiert):**
```
06:00–10:00 Uhr (Morgenspitze):  2.000–3.500 Fahrräder/h (zur Arbeit)
12:00–16:00 Uhr (Mittag):        1.200–1.800 Fahrräder/h (Rückgang)
17:00–20:00 Uhr (Abendspitze):   2.500–4.000 Fahrräder/h (Rückweg + Freizeit) ← PEAK!
21:00–06:00 Uhr (Nacht):         <500 Fahrräder/h (minimal)
```

**Wochenend-Effekt:**
- Werktage: Zwei scharfe Pendler-Peaks (Doppel-Peak-Struktur)
- Wochenenden: Breitere, flachere Verteilung (Freizeitverkehr, keine Pendler)
- Durchschnitt Werktag: 1.203 Räder/h
- Durchschnitt Wochenende: 977 Räder/h (−18,8%)

**Wetter-Einfluss (schwächer):**
- Temperatur: Moderat positiv (wärmer → +2–3% mehr Ausleihen)
- Luftfeuchtigkeit: Schwach negativ (nass → −1–2% weniger)
- Windgeschwindigkeit: Minimal (Wind hat wenig Effekt)

**Saisonalität:**
- Winter: 800–850 Räder/h | Frühling: 900–1.100 | Sommer: 1.400–1.500 ← Peak | Herbst: 1.200–1.350

---

## 3. Datenvorbereitung und Validierungsstrategie

### 3.1 Train/Test Split (Chronologisch, nicht zufällig)

```
Trainingsmenge: 13.892 Stunden (80%)  → 04.01.2015 - 07.08.2016
Testmenge:       3.474 Stunden (20%)  → 07.08.2016 - 03.01.2017
```

**Warum chronologisch?** Dies ist eine Zeitreihe. Zufälliges Shuffling würde zu Data Leakage führen: Das Modell könnte Zukunftsdaten zur Vergangenheits-Vorhersage nutzen. Ein chronologischer Split simuliert die Realität: **Trainiere auf der Vergangenheit, teste auf der Zukunft.**

### 3.2 Cross-Validation: TimeSeriesSplit (5 Folds)

```
Fold 1: Train [0–20%]   → Validate [20–40%]
Fold 2: Train [0–40%]   → Validate [40–60%]
Fold 3: Train [0–60%]   → Validate [60–80%]
Fold 4: Train [0–80%]   → Validate [80–100%]
Test Set bleibt unberührt für finale Evaluation
```

**Vorteil:** Das Modell schaut niemals in die Zukunft. Jeder Fold simuliert realistisches Forecasting.

---

## 4. Modell-Experimente: 13 Algorithmen × 4 Scaler

### 4.1 Systematischer Vergleich: Alle Kategorien

#### A. Lineare Modelle (Baselines)

| Modell | Beste Config | R² | Asym-Score | Status |
|--------|--------------|-----|--------|---------|
| LinearRegression | Alle Scaler | 0,42 | 43% |  Völlig untauglich |
| Ridge (α=0,1-100) | Standard/MinMax/Robust | 0,42 | 43% |  Zu simpel |
| Lasso (α=0,1-10) | Standard/MinMax/Robust | 0,42 | 44% |  Sparsity hilft nicht |
| BayesianRidge | Standard/MinMax/Robust | 0,42 | 42% |  Noch schlechter |

**Fazit:** Alle linearen Modelle stagnieren bei R² ≈ 0,42. Die Bike-Sharing-Nachfrage ist **stark nicht-linear** (Schwelleffekte bei Stoßzeiten, diskontinuierliche Wochenend-Effekte).

#### B. Baum-basierte Modelle (Dominant!)

| Modell | Beste Config | R² | Asym-Score |
|--------|--------------|-----|--------|
| DecisionTree | Kein Scaler, max_depth=20 | 0,9025 | 71,4% |
| RandomForest | Kein Scaler, 200 Bäume | 0,9223 | 74,0% |
| ExtraTrees | Kein Scaler | 0,9173 | 73,3% |
| GradBoost (Standard MSE) | Kein Scaler | 0,9266 | 72,7% |
| **GradBoost (Quantile 85%)** | **Kein Scaler** | **0,9129** | **80,4%**  |

**Erkenntnis:** Tree-basierte Modelle dominieren linear um Faktoren. Quantile-Regression schlägt Standard-Regression beim Business-Score (+7,7% ohne R²-Verlust von 1,4%).

#### C. Distance-based Modelle

| Modell | Beste Config | R² | Asym-Score |
|--------|--------------|-----|--------|
| KNN | MinMaxScaler, k=10 | 0,8388 | 69,0% |
| SVR (RBF) | MinMaxScaler | 0,5341 | 63,4% |

#### D. Neuronale Netze

| Modell | Beste Config | R² | Asym-Score |
|--------|--------------|-----|--------|
| MLP (50,) | StandardScaler | 0,9272 | 70,4% |
| MLP (100,50) | RobustScaler | 0,9278 | 69,5% |

#### E. Ensemble-Modelle

| Modell | Zusammensetzung | R² | Asym-Score |
|--------|-----------------|-----|--------|
| Voting Regressor | MLP + GradBoost + RandomForest | 0,9307 | 74,2% |

#### F. Post-Processed Modelle (Advanced)

| Modell | Basis-Algo | R² | Asym-Score | Verbess. |
|--------|-----------|-----|--------|----------|
| **KNN Custom + PP** | KNN (MinMax) | **0,9180** | **84,01%** | **+10,4%**  |
| RandomForest + PP | RF | 0,9230 | 82,8% | +8,8% |
| GradBoost_Q85 + PP | GradBoost | 0,9038 | 81,4% | +1,0% |

### 4.2 Top 10 Modelle (Final Ranking)

| Rang | Modell | R² | RMSE | MAE | MAPE | Asym-Score |
|:----:|--------|-----|------|-----|------|--------|
|  1 | **KNN Custom + Post-Process** | **0,9180** | **323** | **186** | **27,6%** | **84,01%**  |
|  2 | RandomForest + Post-Process | 0,9230 | 313 | 187 | 27,0% | 82,83% |
|  3 | GradBoost_Q85 + Post-Process | 0,9038 | 350 | 210 | 32,5% | 81,42% |
| 4 | GradBoost_Q85 (Standard) | 0,9129 | 333 | 190 | 29,5% | 80,40% |
| 5 | RandomForest (Standard) | 0,9223 | 314 | 188 | 27,5% | 73,97% |
| 6 | **Voting Ensemble** | **0,9307** | **297** | **180** | **25,1%** | 73,90% |
| 7 | KNN Custom (Standard) | 0,9218 | 315 | 189 | 28,0% | 73,62% |
| 8 | ExtraTrees | 0,9173 | 324 | 191 | 29,2% | 73,35% |
| 9 | GradBoost Standard (MSE) | 0,9266 | 306 | 182 | 26,2% | 72,66% |
| 10 | DecisionTree | 0,9025 | 352 | 207 | 30,8% | 71,37% |

---

## 5. Custom Scoring: Das Kernstück der Innovation

### 5.1 Das Problem mit Standard-Metriken

Klassische Metriken (R², RMSE, MAE) behandeln Fehler **symmetrisch**:

```python
# Mathematisch äquivalent:
Fehler_1 = |4000 - 3900| = 100  (100 zu wenig)
Fehler_2 = |4000 - 4100| = 100  (100 zu viel)
# Beide werden GLEICH schwer bestraft!
```

**Aber geschäftlich:**

```python
# Szenario 1: 100 Räder zu wenig (Underestimation)
Kosten = 100 Räder × 1€/Rad = 100€
Schaden = verlorene Kunden, Reputationsverlust

# Szenario 2: 100 Räder zu viel (Overestimation)
Kosten = 100 Räder × 0.20€/Rad = 20€
Schaden = Wartungskosten, managebar
```

**Das Verhältnis ist 5:1, nicht 1:1!**

### 5.2 Asymmetric Weighted Scoring Function (Implementiert)

```python
def bike_score_asymmetric_weighted(y_true, y_pred):
    """
    Business-fokussierter Score mit kontinuierlicher Gewichtung.
    
    Logik:
    1. Toleranzband: -5% bis +20%
    2. Innerhalb: Score = 1.0 (perfekt)
    3. Außerhalb: Kontinuierliche Penalty
       - Unterversorgung: 2.5× Penalty
       - Überversorgung: 0.8× Penalty
    4. Gewichtung nach Demand-Magnitude: sqrt(y_true)
    """
    # Toleranzbänder
    lower = y_true * 0.95   # -5%
    upper = y_true * 1.20   # +20%
    
    # Score-Berechnung
    score = np.ones_like(y_true, dtype=float)
    
    # Innerhalb Toleranz: perfekt (1.0)
    in_band = (y_pred >= lower) & (y_pred <= upper)
    
    # Außerhalb: gewichtete Penalties
    err_ratio = |y_pred - y_true| / y_true
    weight = sqrt(y_true)  # Gewichtung nach Magnitude
    
    # Unterversorgung (Stockout): 2.5× teurer
    under = y_pred < lower
    score[under] -= 2.5 * err_ratio[under] * weight[under]
    
    # Überversorgung (Verschwendung): 0.8× Penalty
    over = y_pred > upper
    score[over] -= 0.8 * err_ratio[over] * weight[over]
    
    # Clipping auf [0.0, 1.0]
    score = clip(score, 0.0, 1.0)
    
    return mean(score)  # Durchschnitt über alle Samples
```

### 5.3 Integration mit GridSearchCV (Production-ML)

**Die Revolution:**

```python
# Standard ML:
gs = GridSearchCV(model, params, cv=cv, scoring='r2')
# → Sucht Parameter, die R² maximieren

# Business ML (Unser Ansatz):
bike_scorer = make_scorer(bike_score_asymmetric_weighted, 
                          greater_is_better=True)
gs = GridSearchCV(model, params, cv=cv, scoring=bike_scorer)
# → Sucht Parameter, die Geschäftserfolg maximieren!
```

**Das bedeutet:** GridSearchCV optimiert Hyperparameter nicht auf "statistisch korrekt", sondern auf "geschäftlich erfolgreich".

---

## 6. Die Evolution: Phase 1 → Phase 2 → Phase 3

### 6.1 Phase 1: Statistik-zentrierte Baseline

**Gewinner-Modell:** Random Forest (R² = 0,932, Asym = 36,7%)

**Zentrale Erkenntnisse:**
-  Lineare Modelle versagen völlig (R² ≈ 0,42)
-  Tree-basierte Modelle dominieren
-  TimeSeriesSplit ist essentiell
-  **KRITISCHES PROBLEM:** Asym-Score nur 36,7%!
-  Das Modell unterschätzt in **46,4% der Stunden** (untauglich!)

**Wendepunkt:** Das Team erkannte: "High R² ≠ Geschäftserfolg"

### 6.2 Phase 2: Der Paradigmenwechsel

**Gewinner-Modell:** GradBoost mit Standard MSE (R² = 0,927, Asym = 72,7%)

**Hauptfortschritt:**
-  Multi-dimensionale Bewertung implementiert (R², MAE, MAPE, Asym)
-  Asymmetrische Kostenstruktur mathematisch codiert
-  Asym-Score stieg um **36 Prozentpunkte** (36,7% → 72,7%)!
-  Unterversorgungsrate halbiert (46,4% → 26%)
-  Erkenntnis: "Asymmetrische Kosten brauchen asymmetrische Optimierung"

**Nächster Gedanke:** "Warum nicht direkt auf Asym-Score optimieren statt auf R²?"

### 6.3 Phase 3: Der Durchbruch – Business-fokussierte Optimierung

**Gewinner-Modell:** KNN Custom + Post-Processing (R² = 0,9180, Asym = **84,01%**)

**Die Innovation:**
-  Custom Scoring Function mit sklearn-Integration
-  GridSearchCV mit `make_scorer()` für Business-Optimierung
-  Post-Processing mit OOF-Multiplikator-Optimierung
-  **Asym-Score stieg auf 84,01%** (+11,4% von Phase 2!)
-  **Unterversorgungsrate auf 17,4%** (kritische Rate akzeptabel)
-  **2,2× Gesamtverbesserung** vom Anfang (36,7% → 84,01%)

**Resultat:** Production-Ready Modell mit mathematisch codierter Business-Logik

### 6.4 Numerische Progression

```
Asymmetrischer Business-Score:

Phase 1 (RF):        36,7%   ████
Phase 2 (GB-Std):    72,7%   ████████████
Phase 3 (KNN+PP):    84,01%  █████████████  FINAL

Gesamtverbesserung: +47,31 Prozentpunkte (+129% Steigerung!)
```

---

## 7. Das Gewinner-Modell: KNN Custom + Post-Processing

### 7.1 Architektur und Komponenten

**KNN Custom Transformer:**
```python
# Komponenten:
- workday Feature (kombiniert is_holiday & is_weekend)
- Zyklische hour/day/month Features
- MinMaxScaler auf transformierte Features
- KNeighborsRegressor(n_neighbors=10, weights='distance', metric='minkowski', p=1)
```

**Post-Processing (Bias-Korrektur):**
```
1. OOF-Vorhersagen: Trainiere mit 5-Fold TimeSeriesSplit
2. Multiplier-Suche: Finde Factor m ∈ [0.85, 1.15] für max. Asym-Score
3. Bester Multiplikator: m = 1.0372 (+3,72% alle Vorhersagen)
4. Test-Anwendung: y_pred_final = y_pred_raw × 1.0372
```

### 7.2 Geschäftliche Kategorisierung

Das Gewinner-Modell teilt seine Vorhersagen in drei Business-Kategorien ein:

```
 OK (−5% bis +20%):        1.748 Stunden (50,3%)
   → Flotte kann optimal geplant werden
   
  ZU VIEL (> +20%):        1.123 Stunden (32,3%)
   → Überversorgung, aber managebar (Wartungskosten)
   
 ZU WENIG (< −5%):         603 Stunden (17,4%)
   → Kritisch, aber akzeptabel (mit Mitigation)
```

**Vergleich mit anderen Top-Modellen:**
- Quantile-GB Standard: ~26% kritisch
- RandomForest Standard: ~23% kritisch
- **KNN Custom + PP: 17,4% kritisch** ← **Deutlich besser!**

### 7.3 Worst-Case vs. Best-Case Analyse

**Top 5 größte Unterschätzungen (Worst-Case):**

| Actual | Predicted | Fehler % | Kontext |
|:---:|:---:|:---:|:---|
| 3.999 | 1.263 | −68,4% | Unerwarteter massiver Spike |
| 3.710 | 1.241 | −66,5% | Massiver Spike (Event?) |
| 895 | 289 | −67,7% | Unerwartet hohe Nacht-Nachfrage |
| 149 | 39 | −73,9% | Spät nachts, extremer Spike |
| 49 | 16 | −67,3% | Sehr geringe Ausgangszahlen |

**Observation:** Die größten Fehler sind **unerwartete Spikes** (Konzerte, Großveranstaltungen), die nicht im Trainings-Datensatz vertreten sind. Das ist **unvermeidbar** und normal.

**Top 5 beste Vorhersagen (Best-Case):**

| Actual | Predicted | Fehler % |
|:---:|:---:|:---:|
| 2.048 | 2.048 | +0,003% |
| 4.657 | 4.658 | +0,020% |
| 956 | 956 | +0,010% |
| 985 | 985 | +0,021% |
| 265 | 265 | −0,011% |

**Observation:** In **stabilen, wiederkehrenden Mustern** (normale Werktage, Standard-Stoßzeiten) ist das Modell quasi **perfekt** (Fehler < 0,02%).

---

## 8. Feature Importance und Business-Insights

### 8.1 Wichtigste Features (nach Random Forest)

| Rang | Feature | Importance | Interpretation |
|:---:|---------|------------|-----------------|
| 1 | `hour` | **42%** | **Dominierend:** Tageszeit ist stärkster Prädiktor |
| 2 | `is_weekend` | **18%** | **Stark:** Wochenend-Effekt signifikant |
| 3 | `month` | **14%** | **Moderat:** Saisonalität (Winter vs. Sommer) |
| 4 | `day_of_week` | **8%** | Wochentag-Feinstruktur |
| 5 | `hum` | **8%** | Schwacher Wetter-Effekt |
| 6 | `t1` | **7%** | Temperatur-Effekt |
| 7+ | Weitere | **<5%** | Negligible |

**Geschäftliche Interpretation:**
- **Pendlerverkehr dominiert:** 42% (hour) + 18% (is_weekend) = **60% Importance!**
- **Modell versteht Kernlogik:** Rush-Hour morgens/abends, ruhig nachts
- **Wetter ist sekundär:** Selbst schlechtes Wetter stoppt nicht das Pendeln zur Arbeit
- **Marketing-Implikation:** Kapazitäts-Planung sollte sich auf **Stoßzeiten** fokussieren

---

## 9. Produktions-Empfehlungen und Deployment

### 9.1 Primary Model: KNN Custom + Post-Processing

**Deployment-Konfiguration:**

```python
# 1. Model Persistence:
joblib.dump({
    'model': knn_model,
    'transformer': knn_transformer,
    'scaler': scaler,
    'multiplier': 1.0372,
    'metadata': {
        'R2': 0.9180,
        'Asym_Score': 84.01,
        'Training_Date': '2025-12-12',
        'Critical_Rate': 0.174
    }
}, 'best_model_knn_custom_pp.pkl')

# 2. Inference:
model_pack = joblib.load('best_model_knn_custom_pp.pkl')
X_transformed = model_pack['transformer'].transform(X_new)
X_scaled = model_pack['scaler'].transform(X_transformed)
y_pred_raw = model_pack['model'].predict(X_scaled)
y_pred_final = y_pred_raw * model_pack['multiplier']
```

### 9.2 Risk Mitigation für verbleibende 17,4% Unterversorgung

**Option 1: Statischer Safety Buffer**
```python
y_pred_safe = y_pred_final * 1.05  # +5% Extra-Puffer
# Reduziert kritische Rate: 17,4% → ~12%
# Trade-off: +5% Überversorgung (akzeptabel)
```

**Option 2: Adaptive Puffer nach Tageszeit (EMPFOHLEN)**
```python
buffer = {
    (6, 10): 1.02,    # Morgenspitze: Modell ist präzise
    (10, 17): 1.05,   # Mittag/Nachmittag: Mittel-Puffer
    (17, 21): 1.12,   # Abendspitze: Großer Puffer (höchste Fehlerquote)
    (21, 6): 1.01     # Nacht: Minimal (sehr stabil)
}
y_pred_adaptive = y_pred_final * buffer[hour]
```

### 9.3 Monitoring und Retraining

**Tägliches Monitoring:**
- Vergleich Vorhersage vs. Actual
- Tracking von Stockout-Events
- Alert wenn Asym-Score < 75%

**Wöchentliches Review:**
- Performance-Metriken (R², MAE, Asym-Score)
- Drift-Detection
- Fehleranalyse

**Monatliche Updates:**
- Feature-Drift Analysis
- Multiplikator-Rekalibrierung

**Quartalsweise Retraining:**
- Vollständiges Retraining mit neuesten Daten
- Hyperparameter-Retuning
- Modell-Vergleich gegen Baselines

---

## 10. Lessons Learned und Wissenschaftliche Erkenntnisse

### 10.1 Technische Erkenntnisse

1. **Asymmetrische Optimization schlägt symmetrische** 
   - Klassisches ML: Optimiere auf R² (symmetrisch) → 36-73% Business-Score
   - Business ML: Optimiere auf Custom Score (asymmetrisch) → 84% Business-Score
   - **Resultat:** Direkte Kodierung von Business-Logik ist essentiell

2. **Custom Scorers sind Production-Grade Tools**
   - Nicht "Hacky", sondern das **richtige Tool** für Business-Anforderungen
   - sklearn macht das elegant mit `make_scorer()`
   - GridSearchCV respektiert Custom Scorer vollständig

3. **Quantile Regression > Standard Regression**
   - Standard (MSE): Optimiert auf Mittelwert (symmetrisch)
   - Quantile: Optimiert auf 85. Perzentil (konservativ)
   - **Resultat:** 80,4% vs. 72,7% Asym-Score

4. **Explizite Features > Komplexe Modelle**
   - KNN + gute Features: 84% Business-Score
   - Deep MLP ohne Features: 70% Business-Score
   - Zyklische Features (sin/cos) sind kritisch

5. **TimeSeriesSplit ist nicht optional**
   - Zufälliger CV-Split → Data Leakage, unrealistische Metriken
   - Chronologischer Split → Fair Evaluation, realistisch

### 10.2 Geschäftliche Erkenntnisse

6. **80% Optimal ist nicht 100% Statistik**
   - Gewinner: R² 0.918 + Asym 84% (exzellent)
   - Das ist der **richtige Trade-off**
   - Perfectionism würde zu Overfitting führen

7. **Underestimation ist teurer als Overestimation**
   - 2,5× höhere Gewichtung ist nicht paranoid
   - Verlorene Kunden kosten viel mehr als Wartung
   - Business-Logik gehört IN den Algorithmus

8. **Vorhersagefehler sind unvermeidbar**
   - Selbst perfekte Modelle können gegen unerwartete Ereignisse nicht vorhersagen
   - Das Modell funktioniert **exzellent auf bekannten Mustern**

9. **Pendlerverkehr dominiert**
   - 60% Feature-Importance für Tageszeit + Wochenende
   - Marketing und Planung sollten sich auf **Stoßzeiten** fokussieren

10. **Early Deployment > Perfectionism**
    - Ein gutes Modell mit 80% jetzt ist besser als perfektes in 6 Monaten
    - Mit Live-Daten schnell zu 85%+ kommen

### 10.3 Methodologische Erkenntnisse

11. **Iteratives Lernen ist essentiell**
    - Phase 1 → 2 → 3 zeigt kontinuierlichen Fortschritt
    - Jede Phase baut auf Einsichten der vorherigen auf

12. **Paradoxes lösen führt zu Innovationen**
    - Das Paradox "R² hoch, aber Business-Score niedrig" war der Auslöser
    - Paradoxes aufzulösen → Durchbruch

13. **Daten-Kategorisierung ist aufschlussreich**
    - "OK / Zu viel / Zu wenig" Kategorisierung zeigte erst, was das Modell wirklich tat
    - Tiefe Datenanalyse ist essentiell

---

## 11. Abschließendes Fazit

### 11.1 Zusammenfassung der Evolution

| Aspekt | Phase 1 | Phase 2 | Phase 3 (Final) |
|--------|---------|---------|-----------------|
| **Fokus** | Statistik-zentiert | Hybrid-Denken | Business-zentiert  |
| **Gewinner-Modell** | Random Forest | GradBoost (MSE) | KNN Custom + PP |
| **R² Score** | 0,932 | 0,927 | 0,9180 |
| **Asym-Score** | 36,7% | 72,7% | **84,01%** |
| **Unterversorgungsrate** | 46,4% | ~26% | **17,4%** |
| **Produktion-Readiness** |  Nein |  Bedingt |  **JA** |

**Gesamtverbesserung: +47,31 Punkte Asym-Score (+129% Steigerung!)**

### 11.2 Die Kernbotschaft

> **Klassisches ML:** Optimiere auf Standard-Metrik (R²), hoffe, dass das Geschäft funktioniert.
>
> **Business-fokussiertes ML (Diese Arbeit):** Codiere Business-Anforderung in Loss-Funktion, optimiere direkt auf Business-Metrik.
>
> **Resultat:** Viel bessere Geschäftsergebnisse (84% vs. 50–73%), ohne Kompromiss bei statistischer Rigor (R² = 0,918 ist exzellent).

### 11.3 Erreichung der Ziele

| Ziel | Erreicht | Evidenz |
|------|----------|---------|
| R² > 0,90 |  **0,9180** | Übertroffen |
| Asym-Score > 80% |  **84,01%** | Übertroffen |
| Kritische Rate < 20%  | **17,4%** | Erfüllt |
| Praktisches Modell | Ja | Deployment-ready |
| Production-Readiness | Ja | Vollständiger Plan |

---

## 12. Abschluss

Mit einem **R² von 0,9180** und einem **Asymmetrischen Score von 84,01%** ist das Modell **produktionsreif** und kann sofort für operative Flotten-Planung eingesetzt werden.

Das Projekt zeigt:
- **Technische Exzellenz:** 13 Algorithmen, 4 Scaler, Custom Scoring, TimeSeriesSplit
- **Wissenschaftliche Reife:** These-Antithese-Synthese über 3 Phasen dokumentiert
- **Business-Verständnis:** Asymmetrische Kosten mathematisch codiert
- **Produktions-Bereitschaft:** Deployment-Plan, Risk Mitigation, Monitoring

**Status:**  Production-Ready

---

**Dokumentation erstellt:** 12. Dezember 2025  
**Datensatz:** london_merged.csv (17.366 Beobachtungen, 2015–2017)  
**Gewinner-Modell:** KNN Custom + Post-Processing  
**Finale Metriken:** R² = 0,9180 | Asym-Score = **84,01%**   