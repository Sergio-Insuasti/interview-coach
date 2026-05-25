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

# backend/pipeline/emotion.py  (CNN section)
import torch
import torch.nn as nn

class EmotionCNN(nn.Module):
    """
    Input:  (B, 1, 48, 48) — grayscale face crops
    Output: (B, 7)         — logits over 7 emotion classes
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # → (B,32,48,48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (B,32,24,24)
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (B,64,24,24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (B,64,12,12)
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# → (B,128,12,12)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → (B,128,6,6)
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_cnn(model, train_loader, val_loader, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()          # backpropagation
            optimiser.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                correct += (model(X).argmax(1) == y).sum().item()
                total += len(y)
        val_acc = correct / total
        scheduler.step(1 - val_acc)
        print(f"Epoch {epoch+1:3d}  val_acc={val_acc:.3f}")

    torch.save(model.state_dict(), "models/emotion_cnn.pt")