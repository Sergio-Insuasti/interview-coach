import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise']
def train_classical_baselines(X: np.ndarray, y: np.ndarray):
    """
    X: (N, 1764) HOG feature matrix
    y: (N,) integer emotion labels
    Trains all classifiers, prints comparison table, saves best.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "k-NN (k=5)":          KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(max_depth=20),
        "SVM (RBF)":           SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        val_acc = clf.score(X_val, y_val)
        results[name] = val_acc
        print(f"{name:25s}  val_acc={val_acc:.3f}")

    best_name = max(results, key=results.get)
    best_clf = classifiers[best_name]
    joblib.dump({"clf": best_clf, "scaler": scaler}, "models/classical_emotion.pkl")
    print(f"\nBest: {best_name} ({results[best_name]:.3f}) — saved.")
    return best_clf, scaler