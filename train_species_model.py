"""
train_species_model.py — TreeAge AI v3.1
Entraîne le classificateur sur 8 espèces algériennes.
Usage: python train_species_model.py
→ Génère models/species_model.pkl
"""
import os, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

np.random.seed(42)

# Profils synthétiques réalistes par espèce
# [file_size_mean/std, width_m/s, height_m/s, brightness_m/s, R_m/s, G_m/s, B_m/s]
PROFILES = {
    "Pin_d_Alep":  [1_100_000,200_000, 3264,400,2448,300, 115,20, 125,18,105,16, 95,14],
    "Cedre_Atlas": [2_800_000,400_000, 4032,300,3024,200,  90,18,  95,15, 85,14, 75,12],
    "Chene_Vert":  [1_900_000,300_000, 3840,400,2160,300, 100,20, 105,15, 98,14, 80,12],
    "Chene_Liege": [2_200_000,350_000, 4000,350,3000,250, 135,22, 145,20,120,18, 95,14],
    "Eucalyptus":  [3_200_000,500_000, 4608,400,3456,300, 170,20, 170,15,165,14,155,12],
    "Genevrier":   [  500_000,120_000, 2560,300,1920,200,  80,18, 100,16, 75,14, 60,12],
    "Thuya":       [  750_000,150_000, 2976,300,2232,200, 105,20, 130,18, 95,16, 70,12],
    "Peuplier":    [1_600_000,250_000, 3840,400,2880,300, 155,22, 150,18,155,18,130,16],
}
N = 60  # échantillons par espèce

def make_data():
    X, y = [], []
    for sp, p in PROFILES.items():
        fs_m,fs_s, w_m,w_s, h_m,h_s, br_m,br_s, r_m,r_s, g_m,g_s, b_m,b_s = p
        for _ in range(N):
            X.append([
                max(50_000, np.random.normal(fs_m,fs_s)),
                max(640,    np.random.normal(w_m,w_s)),
                max(480,    np.random.normal(h_m,h_s)),
                np.clip(np.random.normal(br_m,br_s),0,255),
                np.clip(np.random.normal(r_m,r_s),0,255),
                np.clip(np.random.normal(g_m,g_s),0,255),
                np.clip(np.random.normal(b_m,b_s),0,255),
            ])
            y.append(sp)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("="*55)
    print("🌲 TreeAge AI — Entraînement Modèle Espèces (8 DZ)")
    print("="*55)
    X, y = make_data()
    print(f"Dataset: {X.shape[0]} échantillons × {X.shape[1]} features")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200,max_depth=5,
                                           learning_rate=0.1,random_state=42))
    ])
    cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5,shuffle=True,random_state=42))
    print(f"CV Accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    pipe.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/species_model.pkl")
    print(f"✅ Modèle sauvegardé → models/species_model.pkl")
    print(f"   Classes: {list(pipe.classes_)}")
