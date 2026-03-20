# 🚴‍♂️ London Bike Sharing - Machine Learning & Predictive Analytics

## Projektübersicht
Dieses Data Science / Machine Learning Projekt demonstriert den Aufbau einer umfassenden Predictive Analytics Pipeline. Ziel ist es, die Auslastung (Nachfrage) des "London Bike Sharing" Systems basierend auf historischen, meteorologischen und kalendarischen Daten vorherzusagen (Regression).

Das Projekt umfasst den gesamten Data Science Lifecycle: Von der explorativen Datenanalyse (EDA) über intensives Feature Engineering, Outlier-Handling bis hin zum Training und der Evaluierung multipler Machine Learning Algorithmen (Ensemble-Methoden).

## Tech Stack & Skills
- **Machine Learning:** XGBoost, Gradient Boosting, K-Nearest Neighbors (KNN)
- **Data Science Workflow:** Scikit-Learn (Pipelines, Custom Transformers), Hyperparameter-Tuning
- **Datenverarbeitung & Analyse:** Pandas, Numpy
- **Evaluierung:** Residuenanalyse, Post-Processing für Constraints (z.B. keine negativen Vorhersagen)
- **Modell-Persistierung:** Pickle (`.pkl`)

## Repository Struktur
Das Repository ist wie folgt organisiert:

- `notebooks/` - Beinhaltet alle Jupyter Notebooks für EDA, Feature Engineering und das Modell-Training (`XGBoost.ipynb`, `Gradient_Boost.ipynb` etc.).
- `data/` - (Lokal ignoriert) Enthält die Trainingsdaten (`london_merged.csv`) und Post-Processing-Exports (Residuen-Analyse).
- `models/` - (Lokal ignoriert) Serialisierte, fertige und getunte ML-Modelle für den direkten Einsatz (`.pkl`).
- `reports/` - Dokumentationen, Projektberichte und HTML-Exporte der Notebooks.

## Kernphasen des Projekts

### 1. Explorative Datenanalyse (EDA) & Preprocessing
Analyse der Verteilungen und Saisonalitäten im Fahrradverleih. Identifizierung und Bereinigung von Ausreißern in Wetter- und Nutzungsdaten (z.B. über Residuenanalyse im Vorfeld).

### 2. Predictive Modeling (Regression)
Vergleich verschiedener Algorithmen zur präzisen Vorhersage der Bike-Rental-Counts:
- **XGBoost & Gradient Boosting:** Hervorragend für tabellarische Daten und komplexe nicht-lineare Zusammenhänge.
- **K-Nearest Neighbors (KNN):** Als Baseline und zur Erkennung lokaler Muster.

### 3. Custom Transformations & Post-Processing
Implementierung von Scikit-Learn Custom Transformers, um spezielle Skalierungen oder Datentransformationen direkt in einer ML-Pipeline (`Pipeline`) abzuwickeln. Zudem ein Custom Post-Processing (z.B. asymmetrische Weighted Scores), um Ausreißer in den Vorhersagen zu korrigieren.

## Business Value (Warum dieses Projekt?)
- **Real-World Problem:** Vorhersage der Nachfrage optimiert Logistik (z.B. Umverteilung von Fahrrädern an Stationen).
- **Advanced ML:** Der Einsatz von Custom Pipelines, Gradient Boosting und tiefgehender Residuenanalyse zeigt ein Seniory Level im Umgang mit Model-Limitations und Overfitting abseits von einfachen `fit() / predict()` Skripten.
