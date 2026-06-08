import joblib
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Load dataset (same split)
data = fetch_olivetti_faces()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Load model
model = joblib.load('savedmodel.pth')
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")