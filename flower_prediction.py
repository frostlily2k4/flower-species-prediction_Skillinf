
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load the dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
y = iris.target

print("FLOWER SPECIES PREDICTION")
print("=" * 45)
print("Total samples:", len(X))
print("Species:", list(iris.target_names))

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Predict on unseen test data
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("-" * 45)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# 6. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()
plt.title("Flower Species Prediction")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 7. Save the trained model and feature names
joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
        "species_names": list(iris.target_names)
    },
    "flower_model.pkl"
)

print("\nModel saved as flower_model.pkl")
print("Confusion matrix saved as confusion_matrix.png")
