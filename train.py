
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

df = pd.read_csv("data/dataset.csv")

X = df.drop(["total_time","total_cost"], axis=1)
y = df[["total_time","total_cost"]]

cat = ["terrain","project_type"]
num = [c for c in X.columns if c not in cat]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(), cat),
    ("num", "passthrough", num)
])

X_p = pre.fit_transform(X)

joblib.dump(pre, "models/preprocessor.pkl")

models = {
    "linear": MultiOutputRegressor(LinearRegression()),
    "rf": MultiOutputRegressor(RandomForestRegressor()),
    "lgbm": MultiOutputRegressor(LGBMRegressor())
}

for name, model in models.items():
    model.fit(X_p, y)
    joblib.dump(model, f"models/{name}.pkl")
