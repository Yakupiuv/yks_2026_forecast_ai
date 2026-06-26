import pandas as pd
import numpy as np
import os

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

pd.options.mode.chained_assignment = None

# Verileri Yüklüyopm
if os.path.exists("features_cache.parquet"):
    df = pd.read_parquet("features_cache.parquet")
else:
    df = pd.read_parquet("data/data.parquet")

    df["key"] = (
        df["universite"] + "_" +
        df["fakulte"] + "_" +
        df["bolum"]
    )

    df["yil"] = df["yil"].astype("int16")
    df["kontenjan"] = df["kontenjan"].astype("float32")
    df["puan"] = df["puan"].astype("float32")
    df["siralama"] = df["siralama"].astype("float32")

    df = df.sort_values(["key", "yil"])

    g = df.groupby("key")["puan"]
    df["lag1"] = g.shift(1)
    df["lag2"] = g.shift(2)
    df["trend"] = df["lag1"] - df["lag2"]

    kg = df.groupby("key")
    df["kontenjan_prev"] = kg["kontenjan"].shift(1)
    df["kontenjan_change"] = df["kontenjan"] - df["kontenjan_prev"]
    df["kontenjan_ratio"] = df["kontenjan"] / (df["kontenjan_prev"] + 1e-6)

    bg = df.groupby("bolum")["puan"]
    df["bolum_avg"] = bg.transform("mean")

    df["bolum_trend"] = bg.transform(lambda x: x.diff())

    df["demand"] = df.groupby("key")["siralama"].shift(1) / (df["kontenjan"] + 1)

    year_mean = df.groupby("yil")["puan"].transform("mean")
    df["year_strength"] = year_mean.max() - year_mean

    df["score_per_slot"] = df["lag1"] / (df["kontenjan"] + 1e-6)
    df["trend_acceleration"] = df["lag1"] - 2 * df["lag2"]

    df.to_parquet("features_cache.parquet", index=False)

if "key" not in df.columns:
    df["key"] = (
        df["universite"] + "_" +
        df["fakulte"] + "_" +
        df["bolum"]
    )

df = df.dropna(subset=["puan", "lag1", "lag2"])

features = [
    "yil",
    "kontenjan",
    "kontenjan_change",
    "kontenjan_ratio",
    "lag1",
    "lag2",
    "trend",
    "bolum_avg",
    "bolum_trend",
    "demand",
    "year_strength",
    "score_per_slot",
    "trend_acceleration"
]

# Train / Test Verileri
train_df = df[df["yil"] != 2025]
test_df = df[df["yil"] == 2025]

X_train = train_df[features]
y_train = train_df["puan"]

X_test = test_df[features]
y_test = test_df["puan"]

# Model
model = CatBoostRegressor(
    iterations=5000,
    depth=8,
    learning_rate=0.01,
    loss_function="MAE",
    random_seed=42,
    verbose=500,
    subsample=0.8,
    rsm=0.8
)

model.fit(X_train, y_train)

# 2025 Tahmini
pred_2025 = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_2025)
rmse = np.sqrt(mean_squared_error(y_test, pred_2025))

accuracy = 100 - (mae / y_test.mean() * 100)

print("\n2025 Sonucu : ")
print(f"Doğruluk : %{accuracy:.2f}")

# 2026 Tahmini
base_2025 = df[df["yil"] == 2025].copy()
base_2025["yil"] = 2026
base_2025["puan"] = np.nan
base_2025["siralama"] = np.nan

for temp_df in [df, base_2025]:
    if "key" not in temp_df.columns:
        temp_df["key"] = temp_df["universite"] + "_" + temp_df["fakulte"] + "_" + temp_df["bolum"]

future = pd.concat([df, base_2025], ignore_index=True)
future = future.sort_values(["key", "yil"])

future = future.reset_index(drop=True) 

if "key" not in future.columns:
    future["key"] = future["universite"] + "_" + future["fakulte"] + "_" + future["bolum"]

future = future.groupby("key", group_keys=False).apply(lambda x: x.ffill())

g = future.groupby("key")["puan"]
future["lag1"] = g.shift(1)
future["lag2"] = g.shift(2)
future["trend"] = future["lag1"] - future["lag2"]

kg = future.groupby("key")
future["kontenjan_prev"] = kg["kontenjan"].shift(1)
future["kontenjan_change"] = future["kontenjan"] - future["kontenjan_prev"]
future["kontenjan_ratio"] = future["kontenjan"] / (future["kontenjan_prev"] + 1e-6)

bg = future.groupby("bolum")["puan"]
future["bolum_avg"] = bg.transform("mean")
future["bolum_trend"] = bg.transform(lambda x: x.diff())

future["demand"] = future.groupby("key")["siralama"].shift(1) / (future["kontenjan"] + 1)

year_mean = future.groupby("yil")["puan"].transform("mean")
future["year_strength"] = year_mean.max() - year_mean

future["score_per_slot"] = future["lag1"] / (future["kontenjan"] + 1e-6)
future["trend_acceleration"] = future["lag1"] - 2 * future["lag2"]

future = future.dropna(subset=features)

X_2026 = future[future["yil"] == 2026][features]

pred_2026 = model.predict(X_2026)

pred_2026 = pd.Series(pred_2026)
pred_2026 = pred_2026.ewm(alpha=0.3).mean()

output = future[future["yil"] == 2026][
    ["universite", "fakulte", "bolum"]
].copy()

output["tahmini_puan"] = pred_2026.values

output.to_csv("universite_2026.csv", index=False)

print("\nBaşarıyla Tamamlandı :3")