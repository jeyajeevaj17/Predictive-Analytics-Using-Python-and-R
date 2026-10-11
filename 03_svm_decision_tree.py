# Experiment 3: SVM and Decision Tree Classification
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
)
models = {
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=42)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)
}
print("SVM / DECISION TREE CLASSIFICATION")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name} accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
print("Target classes:", data.target_names.tolist())
