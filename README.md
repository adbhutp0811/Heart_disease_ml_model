# Heart Stroke Prediction

A Streamlit web app that predicts heart disease risk using a K-Nearest Neighbors
classifier trained on the Heart Stroke Dataset. Users enter patient details
through a form and the app returns a high-risk or low-risk prediction.

## Features

- Interactive Streamlit form covering all 11 model features
- One-hot encoding for categorical inputs, aligned to the training column order
- Feature scaling with the fitted `StandardScaler` before inference
- Instant high-risk / low-risk result

## Files

| File | Description |
| --- | --- |
| `main.py` | Streamlit application |
| `knn_heart_model.pkl` | Trained KNN classifier (joblib) |
| `heart_scaler.pkl` | Fitted `StandardScaler` (joblib) |
| `heart_columns.pkl` | Expected feature column order (joblib) |

## Requirements

- Python 3.8+
- `streamlit`
- `pandas`
- `scikit-learn`
- `joblib`

## Usage

Install the dependencies and run the app from this directory:

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** The `.pkl` artifacts are loaded from the current working directory, so
> `main.py` must be run from the folder that contains them.

## Disclaimer

This project is for educational purposes only and is not a substitute for
professional medical advice, diagnosis, or treatment.
