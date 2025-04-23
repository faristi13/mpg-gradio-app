import gradio as gr
import joblib
import numpy as np

# Cargar modelo entrenado
model = joblib.load("modelo_mpg.pkl")

# Función predictiva
def predict_mpg(horsepower, weight, acceleration, displacement):
    data = np.array([[horsepower, weight, acceleration, displacement]])
    return round(model.predict(data)[0], 2)

# Interfaz
interface = gr.Interface(
    fn=predict_mpg,
    inputs=[
        gr.Number(label="Horsepower"),
        gr.Number(label="Weight"),
        gr.Number(label="Acceleration"),
        gr.Number(label="Displacement")
    ],
    outputs="number",
    title="MPG Estimator - Real Model"
)

# Para Cloud Run
interface.launch(server_name="0.0.0.0", server_port=8080)
