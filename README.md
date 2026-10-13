# 🌸 Flower Species Prediction

## 1. Project Overview

Flower Species Prediction is a machine learning project that identifies the species of an Iris flower based on its physical measurements. It uses a Random Forest classification model and provides an interactive web interface built with Streamlit.

## 2. Objective

To train and evaluate a machine learning classification model that predicts whether a flower belongs to the Setosa, Versicolor, or Virginica species using sepal and petal measurements.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 4. Dataset

The project uses the Iris dataset provided by Scikit-learn.

- Total samples: 150
- Features: 4
- Target classes: 3
- Samples per species: 50

### Input Features

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

### Output Classes

- Setosa
- Versicolor
- Virginica

## 5. Machine Learning Algorithm

**Random Forest Classifier**

The dataset is split into:
- 80% training data
- 20% testing data

The model is trained on the training data and evaluated on unseen test data.

## 6. Model Evaluation Results

Test samples: 30

**Accuracy: 90%**

| Species | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 0.82 | 0.90 | 0.86 |
| Virginica | 0.89 | 0.80 | 0.84 |
| **Macro average** | **0.90** | **0.90** | **0.90** |

### Confusion Matrix

| Actual / Predicted | Setosa | Versicolor | Virginica |
|---|---:|---:|---:|
| Setosa | 10 | 0 | 0 |
| Versicolor | 0 | 9 | 1 |
| Virginica | 0 | 2 | 8 |

The model correctly classified 27 out of 30 test samples. Setosa was classified correctly in all 10 test cases. Most errors occurred between Versicolor and Virginica.

## 7. Interactive Web Application

The Streamlit application allows users to enter four flower measurements and receive:

- Predicted flower species
- Estimated probabilities for all three species
- A visual probability chart

## 8. How to Run the Project

### Install dependencies

```bash
python -m pip install pandas scikit-learn matplotlib streamlit joblib
```

### Train and evaluate the model

```bash
python flower_prediction.py
```

This generates the saved model (`flower_model.pkl`) and confusion matrix (`confusion_matrix.png`).

### Launch the web application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## 9. Project Files

- `flower_prediction.py` — dataset loading, model training, and evaluation
- `iris_dataset.csv` — Iris dataset saved as a CSV file
- `app.py` — interactive Streamlit application
- `flower_model.pkl` — saved trained model
- `confusion_matrix.png` — model evaluation visualization

## 10. Limitations

The model was evaluated on a small test set of 30 samples. Performance on new flower measurements may differ from the reported results. Predicted probabilities are model estimates, not guarantees.

## 11. Conclusion

The Random Forest classifier achieved 90% accuracy on the held-out test set. The project demonstrates how machine learning can classify flower species from numerical measurements and how a Streamlit application can make predictions accessible through a simple user interface.

## Author 
Jothi A