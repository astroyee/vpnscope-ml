import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------------
# 1. Load ARFF file
# -------------------------------------
file_path = "ml/data/processed/TimeBasedFeatures-Dataset-15s-VPN.arff"

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Strings in ARFF become bytes after loading, need to decode
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# -------------------------------------
# 2. Split features and label
# -------------------------------------
X = df.drop("class1", axis=1)
y = df["class1"]

# Encode labels (Non-VPN / VPN -> numeric)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------------
# 3. Train-test split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------------
# 4. Decision Tree Model
# -------------------------------------
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,        # adjustable
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

dt.fit(X_train, y_train)

# -------------------------------------
# 5. Prediction and Evaluation
# -------------------------------------
y_pred = dt.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------
# 6. Decision Tree Rules
# -------------------------------------
print("\nDecision Tree Rules (partial):")
rules = export_text(dt, feature_names=list(X.columns))
# print(rules)

# -------------------------------------
# 7. Feature Importance Analysis
# -------------------------------------
importances = pd.Series(dt.feature_importances_, index=X.columns)
print("\nFeature Importances (sorted):")
print(importances.sort_values(ascending=False))
