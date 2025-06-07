import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
    pd.Series(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.Series(y_test).to_csv("data/processed/y_test.csv", index=False)