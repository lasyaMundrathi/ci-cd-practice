import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

def train():
    iris = load_iris()
    X, y = iris.data, iris.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    joblib.dump(model, 'model.joblib')
    print("Model trained and saved as model.joblib")

if __name__ == "__main__":
    train()
