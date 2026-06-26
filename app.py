import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Verileri Hazırlıyom
df = pd.read_parquet("data/data.parquet")
df["key"] = df["universite"] + "_" + df["fakulte"] + "_" + df["bolum"]

def create_features(data):
    df = data.copy().sort_values(["key", "yil"])
    for i in [1, 2, 3]:
        df[f'lag_{i}'] = df.groupby('key')['puan'].shift(i)
    
    # Trend
    df['trend'] = df['lag_1'] - df['lag_2']
    # Kontenjan değişim etkisi
    df['kont_change'] = df.groupby('key')['kontenjan'].diff()
    # Ortalama Puan 
    df['bolum_mean'] = df.groupby('bolum')['puan'].transform('mean')
    return df

df_feat = create_features(df).fillna(0)

train = df_feat[df_feat['yil'] < 2025]
test = df_feat[df_feat['yil'] == 2025]

X_train, y_train = train.drop(['puan', 'key', 'universite', 'fakulte', 'bolum'], axis=1), train['puan']
X_test, y_test = test.drop(['puan', 'key', 'universite', 'fakulte', 'bolum'], axis=1), test['puan']

cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0)
cat.fit(X_train, y_train)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)

# 2025 TAHMİNİ 
pred_cat = cat.predict(X_test)
pred_ridge = ridge.predict(X_test_sc)
final_preds = 0.7 * pred_cat + 0.3 * pred_ridge 

print(f"2025 Başarı Skoru : {mean_absolute_error(y_test, final_preds):.4f}")
print(f"R2 Skoru : {r2_score(y_test, final_preds):.4f}")

# 2026 TAHMİNİ
last_year_data = df_feat[df_feat['yil'] == 2025].copy()
last_year_data['yil'] = 2026
X_2026 = last_year_data.drop(['puan', 'key', 'universite', 'fakulte', 'bolum'], axis=1)

pred_2026 = 0.7 * cat.predict(X_2026) + 0.3 * ridge.predict(scaler.transform(X_2026))
last_year_data['tahmini_puan'] = pred_2026

print("2026 tahminleri hazırlandı.")