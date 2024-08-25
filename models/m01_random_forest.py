# m01_random_forest.py
# Se utiliza Random Forest para realizar la clasificacion binaria y se obtienen las métricas correspondientes

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# Cargar la base de datos
df = pd.read_csv("dataset_Caso_1.csv")

# Separar la target del resto del dataset
X = df.drop('target', axis=1)
y = df['target']

# Cambiar tipo de datos de columnas de tipo object
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Separar data en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# RandomForest
rf_classifier = RandomForestClassifier(random_state=42)

# Buscar mejores hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros
mejores_params = grid_search.best_params_
print("Parámetros: ", mejores_params)

# Usar el mejor modelo encontrado
rf_classifier = grid_search.best_estimator_

# Training
rf_classifier.fit(X_train, y_train)

# Predicción
y_pred = rf_classifier.predict(X_test)

# Obtener probabilidades
y_pred_proba_test = rf_classifier.predict_proba(X_test)
y_pred_proba_train = rf_classifier.predict_proba(X_train)

# Calcular las métricas
auc_test = roc_auc_score(y_test, y_pred_proba_test[:, 1])
auc_train = roc_auc_score(y_train, y_pred_proba_train[:, 1])
f1_test = f1_score(y_test, y_pred)

print('AUC Train:', auc_train)
print('AUC Test:', auc_test)
print('F1-Score Test:', f1_test)
print(classification_report(y_test, y_pred, zero_division = 1))
