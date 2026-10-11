# Experiment 4: Naive Bayes Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
)
model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("NAIVE BAYES CLASSIFICATION")
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
print("Class names:", data.target_names.tolist())
