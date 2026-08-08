from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json()["message"] == "Wine Quality Prediction API is running"


def test_prediction():
    data = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.00,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "ph": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_quality" in result
    assert isinstance(result["predicted_quality"], float)