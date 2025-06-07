import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_resampled, y_resampled)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_resampled, y_resampled)

    # Evaluation
    for name, model in [("XGBoost", xgb), ("RandomForest", rf)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"\n===== {name} Evaluation =====")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))
        print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

    # Save best model (XGBoost in this case)
    joblib.dump(xgb, "models/fraud_xgb_model.pkl")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()
    train_and_evaluate(X_train, y_train, X_test, y_test)