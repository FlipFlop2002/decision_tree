import numpy as np
from sklearn.preprocessing import LabelEncoder

def encode_data(path: str):
    with open(path, 'r') as file:
        lines = file.readlines()

    # Podział danych na atrybuty i etykiety
    X = []
    y = []
    for line in lines:
        parts = line.strip().split(',')
        X.append(parts[:-1])  # atrybuty
        y.append(parts[-1])   # etykieta

    # Konwersja danych na numeryczne
    label_encoder = LabelEncoder()
    X_encoded = []
    for i in range(len(X[0])):
        feature_values = [row[i] for row in X]
        feature_encoded = label_encoder.fit_transform(feature_values)
        X_encoded.append(feature_encoded)
    # Transponowanie tablicy dla poprawnego kształtu
    X_encoded = np.array(X_encoded).T
    # Konwersja etykiet klas na wartości liczbowe
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded




