import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('ready_sales_dataset.csv')

target = 'ReturnStatus_Returned' 
features = ['TotalPrice', 'Discount', 'Quantity', 'UnitPrice']  

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)

#rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Prediction
y_pred = rf_clf.predict(X_test_scaled)
y_pred_proba = rf_clf.predict_proba(X_test_scaled)[:, 1]

# Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.2f}")

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['No Return', 'Returned'], cmap='Blues')
plt.title("Матрица ошибок (Random Forest)")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Случайное угадывание')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Feature importance
feature_importances = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=False)
print("Feature Importances:")
print(feature_importances)

# Plotting
plt.figure(figsize=(8, 6))
feature_importances.plot(kind='bar', color='skyblue')
plt.title('Важность признаков (Feature Importances)')
plt.xlabel('Признаки')
plt.ylabel('Важность')
plt.grid(True)
plt.show()
