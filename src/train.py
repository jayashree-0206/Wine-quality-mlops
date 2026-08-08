import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/winequality-red.csv"

data = pd.read_csv(DATA_PATH, sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]


# --------------------------------------------------
# 2. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# --------------------------------------------------
# 4. MLflow experiment
# --------------------------------------------------

mlflow.set_experiment("Wine Quality Prediction")


# --------------------------------------------------
# 5. Train and evaluate each model
# --------------------------------------------------

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(
            y_test,
            predictions
        ) ** 0.5
        r2 = r2_score(y_test, predictions)

        # Print results
        print(f"\n{model_name}")
        print("-" * 30)
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model parameters
        if model_name == "Random Forest":
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 10)

        elif model_name == "Gradient Boosting":
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_param("max_depth", 3)

        elif model_name == "Linear Regression":
            mlflow.log_param("model_type", "Linear Regression")

        # Log model
        if model_name == "Random Forest":
            mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="WineQualityModel"
            )
            joblib.dump(
             model,
            "models/best_model.pkl"
            )

        else:
            mlflow.sklearn.log_model(
            model,
            name="model"
            )

        # Save model locally
        os.makedirs("models", exist_ok=True)

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".pkl"
        )

        joblib.dump(
            model,
            f"models/{filename}"
        )


print("\nTraining completed successfully!")