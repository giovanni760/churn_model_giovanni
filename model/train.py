# Librerías
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

# 1. Cargar dataset
df = pd.read_csv(pathlib.Path("data/Churn_Modelling.csv"))

# 2. Eliminar columnas irrelevantes
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# 3. Codificar variables categóricas (Geography, Gender)
df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True,  # evita multicolinealidad
    dtype=int
)

# 4. Separar features (X) y target (y)
X = df.drop("Exited", axis=1)   # "Exited" = cliente se fue o no
y = df["Exited"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 6. Definir y entrenar modelo
model = RandomForestClassifier(
    n_estimators=100,
    bootstrap=True,
    max_features="sqrt",
    random_state=0
)
model.fit(X_train, y_train)

# 7. Evaluar modelo
print("Accuracy en test:", model.score(X_test, y_test))
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 8. Guardar modelo entrenado
dump(model, pathlib.Path("model/churn-model-v1.joblib"))
    
