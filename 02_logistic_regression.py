# Experiment 2: Logistic Regression
# Iris classification with a reproducible train/test split.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42, stratify=iris.target
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("LOGISTIC REGRESSION - IRIS CLASSIFICATION")
print("Test samples:", len(y_test))
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("First 5 actual classes:", y_test[:5].tolist())
print("First 5 predicted classes:", pred[:5].tolist())
print("Class names:", iris.target_names.tolist())
