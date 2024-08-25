# e01_eda.py
# Análisis exploratorio del dataset del caso

import pandas as pd
import numpy as np

# Se carga la base de datos
df = pd.read_csv("dataset_Caso_1.csv")

# Se obtiene el tamaño del dataset
df.shape

# Se revisan columnas
df.columns
df.head()

# Se revisa si target está balanceada
df['target'].value_counts()
df['target'].value_counts(normalize = True)

# Se analiza el tipo de datos de las variables independientes
df.info()
df.nunique()
