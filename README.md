\## Live Demo

Try the app here: https://your-streamlit-url.streamlit.app



\# Hospital Readmission Predictor



A machine learning project to predict 30-day hospital readmissions

using the UCI Diabetes 130-US Hospitals dataset (101,766 patients).



\## Problem framing

Each year, roughly 20% of Medicare patients in the US are readmitted

to hospital within 30 days of discharge — many of them preventably.

This project builds a binary classifier to flag high-risk patients

before discharge.



\## Tech stack

\- Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit



\## Project structure

\- notebooks/ — Jupyter notebooks for each phase

\- src/ — helper functions

\- data/ — raw and cleaned datasets (not tracked in git)



\## Evaluation metric

ROC-AUC and Recall — not accuracy, due to class imbalance (11.2%

positive rate).

