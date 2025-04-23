import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Cargar datos
df = sns.load_dataset("mpg").dropna()
df = df[["horsepower", "weight", "acceleration", "displacement", "mpg"]]

X = df.drop("mpg", axis=1)
y = df["mpg"]

# Entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Guardar modelo
joblib.dump(model, "modelo_mpg.pkl")