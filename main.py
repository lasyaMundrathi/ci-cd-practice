from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

@app.get("/", response_class=HTMLResponse)
def form_get():
    return """
    <html>
      <body>
        <h2>Enter Features</h2>
        <form action="/" method="post">
          <input name="f1" type="text" placeholder="Feature 1"><br>
          <input name="f2" type="text" placeholder="Feature 2"><br>
          <input name="f3" type="text" placeholder="Feature 3"><br>
          <input name="f4" type="text" placeholder="Feature 4"><br>
          <input type="submit" value="Predict">
        </form>
      </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
def form_post(
    f1: float = Form(...),
    f2: float = Form(...),
    f3: float = Form(...),
    f4: float = Form(...),
):
    data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(data)[0]
    return f"""
    <html>
      <body>
        <h2>Enter Features</h2>
        <form action="/" method="post">
          <input name="f1" type="text" value="{f1}"><br>
          <input name="f2" type="text" value="{f2}"><br>
          <input name="f3" type="text" value="{f3}"><br>
          <input name="f4" type="text" value="{f4}"><br>
          <input type="submit" value="Predict">
        </form>
        <h3>Prediction: {prediction}</h3>
      </body>
    </html>
    """
