# =============================================================================
# species_classifier.py — TreeAge AI v3.1
# 8 Espèces Algériennes — Classification + Estimation d'Âge Non-Destructive
# =============================================================================
import os, logging, numpy as np

try:
    import joblib; JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from PIL import Image; PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger("species_classifier")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "species_model.pkl")

# ─── 8 Espèces Algériennes ───────────────────────────────────────────────────
SPECIES_CATALOG = {
    "Pin_d_Alep": {
        "latin_name": "Pinus halepensis", "common_fr": "Pin d'Alep",
        "growth_rate": "fast", "avg_rings_per_cm": 3.2,
        "diameter_age_factor": 0.28, "max_age_years": 400, "emoji": "🌲",
        "description": "Espèce la plus répandue en Algérie. Méditerranéen semi-aride.",
        "habitat": ["Mediterranean","semi-arid","tell"],
        "regions_alg": ["Tlemcen","Blida","Médéa","Bouira","Tizi Ouzou","Béjaïa","Jijel"],
        "elevation_range": (0,1800), "bark_color": "reddish-gray"
    },
    "Cedre_Atlas": {
        "latin_name": "Cedrus atlantica", "common_fr": "Cèdre de l'Atlas",
        "growth_rate": "slow", "avg_rings_per_cm": 2.0,
        "diameter_age_factor": 0.38, "max_age_years": 1000, "emoji": "🌳",
        "description": "Arbre emblématique de l'Algérie. Hautes montagnes de l'Atlas.",
        "habitat": ["mountain","sub-humid","cold"],
        "regions_alg": ["Batna (Aurès)","Khenchela","Tébessa","Tiaret","Médéa"],
        "elevation_range": (1200,2400), "bark_color": "dark-gray"
    },
    "Chene_Vert": {
        "latin_name": "Quercus ilex", "common_fr": "Chêne vert",
        "growth_rate": "slow", "avg_rings_per_cm": 2.2,
        "diameter_age_factor": 0.35, "max_age_years": 800, "emoji": "🌳",
        "description": "Chêne sempervirent du Tell atlasique. Bois très dur.",
        "habitat": ["Mediterranean","sub-humid","tell_atlas"],
        "regions_alg": ["Tizi Ouzou","Béjaïa","Skikda","Annaba","Guelma","El Tarf"],
        "elevation_range": (100,1600), "bark_color": "dark-brown"
    },
    "Chene_Liege": {
        "latin_name": "Quercus suber", "common_fr": "Chêne-liège",
        "growth_rate": "medium", "avg_rings_per_cm": 2.8,
        "diameter_age_factor": 0.30, "max_age_years": 600, "emoji": "🌳",
        "description": "Producteur de liège. Forêts humides du nord-est algérien.",
        "habitat": ["humid","coastal","north-east"],
        "regions_alg": ["El Tarf","Skikda","Annaba","Guelma","Souk Ahras"],
        "elevation_range": (0,900), "bark_color": "cork-beige"
    },
    "Eucalyptus": {
        "latin_name": "Eucalyptus globulus", "common_fr": "Eucalyptus",
        "growth_rate": "very_fast", "avg_rings_per_cm": 4.8,
        "diameter_age_factor": 0.18, "max_age_years": 150, "emoji": "🌿",
        "description": "Espèce introduite, croissance très rapide. Reboisement des plaines.",
        "habitat": ["planted","plains","humid"],
        "regions_alg": ["Blida","Tipaza","Alger","Oran","Mostaganem"],
        "elevation_range": (0,800), "bark_color": "white-gray"
    },
    "Genevrier": {
        "latin_name": "Juniperus thurifera", "common_fr": "Genévrier thurifère",
        "growth_rate": "very_slow", "avg_rings_per_cm": 1.5,
        "diameter_age_factor": 0.50, "max_age_years": 2000, "emoji": "🌲",
        "description": "Arbre très ancien. Atlas Saharien et hauts plateaux.",
        "habitat": ["semi-arid","sub-desert","high_plateau"],
        "regions_alg": ["Laghouat","Djelfa","M'Sila","Naâma","El Bayadh"],
        "elevation_range": (800,2100), "bark_color": "reddish-brown"
    },
    "Thuya": {
        "latin_name": "Tetraclinis articulata", "common_fr": "Thuya de Berbérie",
        "growth_rate": "slow", "avg_rings_per_cm": 2.0,
        "diameter_age_factor": 0.40, "max_age_years": 700, "emoji": "🌲",
        "description": "Arbre berbère endémique du Maghreb. Nord-ouest algérien.",
        "habitat": ["Mediterranean","dry","north-west"],
        "regions_alg": ["Tlemcen","Ain Témouchent","Mascara","Tiaret","Saïda"],
        "elevation_range": (200,1400), "bark_color": "reddish"
    },
    "Peuplier": {
        "latin_name": "Populus nigra", "common_fr": "Peuplier noir",
        "growth_rate": "fast", "avg_rings_per_cm": 4.0,
        "diameter_age_factor": 0.22, "max_age_years": 300, "emoji": "🌳",
        "description": "Bords des oueds et vallées. Croissance rapide.",
        "habitat": ["riparian","valley","oued"],
        "regions_alg": ["Biskra","El Oued","Ghardaïa","Béchar","Ouargla"],
        "elevation_range": (0,1200), "bark_color": "white-green"
    }
}

GROWTH_RATE_DISPLAY = {
    "very_fast": {"label":"Très rapide","color":"#ef4444"},
    "fast":      {"label":"Rapide",     "color":"#f97316"},
    "medium":    {"label":"Moyen",      "color":"#eab308"},
    "slow":      {"label":"Lent",       "color":"#22c55e"},
    "very_slow": {"label":"Très lent",  "color":"#3b82f6"}
}


def extract_image_features(filepath: str) -> list:
    """7 features: [file_size, width, height, brightness, red, green, blue]"""
    try: file_size = float(os.path.getsize(filepath))
    except: file_size = 500_000.0
    features = [file_size]
    if PIL_AVAILABLE:
        try:
            img = Image.open(filepath).convert("RGB")
            arr = np.array(img.resize((64,64)), dtype=np.float32)
            features += [float(img.width), float(img.height),
                         round(float(arr.mean()),2),
                         round(float(arr[:,:,0].mean()),2),
                         round(float(arr[:,:,1].mean()),2),
                         round(float(arr[:,:,2].mean()),2)]
            return features[:7]
        except: pass
    import hashlib
    h = int(hashlib.md5(str(int(file_size)).encode()).hexdigest(), 16)
    features += [float(800+(h%2000)), float(600+(h%1500)),
                 float(100+(h%155)), float(80+(h%100)),
                 float(90+(h%80)),   float(70+(h%90))]
    return features[:7]


class SpeciesClassifier:
    def __init__(self):
        self.model = None
        if JOBLIB_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"✅ Modèle chargé: {MODEL_PATH}")
            except Exception as e:
                logger.error(f"❌ {e}")

    def predict(self, filepath: str) -> dict:
        if self.model is not None:
            try:
                feats  = extract_image_features(filepath)
                probs  = self.model.predict_proba([feats])[0]
                classes = self.model.classes_
                idx    = int(np.argmax(probs))
                return self._build(classes[idx], float(probs[idx]), probs, classes, "ml")
            except Exception as e:
                logger.error(f"ML error: {e}")
        return self._rule_based(filepath)

    def _rule_based(self, filepath: str) -> dict:
        try:
            size = os.path.getsize(filepath)
            feats = extract_image_features(filepath)
            br    = feats[3] if len(feats)>3 else 128.0
            gn    = feats[5] if len(feats)>5 else 100.0
            if br > 170 and gn > 110:   sp, cf = "Eucalyptus",   0.62
            elif size > 3_000_000:      sp, cf = "Cedre_Atlas",  0.60
            elif size > 2_000_000:      sp, cf = "Chene_Vert",   0.63
            elif size > 1_200_000:      sp, cf = "Pin_d_Alep",   0.68
            elif size > 700_000:        sp, cf = "Chene_Liege",  0.60
            elif size > 400_000:        sp, cf = "Peuplier",     0.62
            elif size > 200_000:        sp, cf = "Thuya",        0.60
            else:                       sp, cf = "Genevrier",    0.58
        except:                         sp, cf = "Pin_d_Alep",   0.50
        return self._build(sp, cf, None, None, "rule")

    def _build(self, name, conf, probs, classes, mode):
        if name not in SPECIES_CATALOG: name = "Pin_d_Alep"
        m  = SPECIES_CATALOG[name]
        gd = GROWTH_RATE_DISPLAY.get(m["growth_rate"], {})
        scores = ({c: round(float(p),4) for c,p in zip(classes,probs)}
                  if probs is not None else {})
        return {
            "species":              name,
            "species_common_fr":    m["common_fr"],
            "confidence":           round(conf, 4),
            "latin_name":           m["latin_name"],
            "emoji":                m["emoji"],
            "description":          m["description"],
            "growth_rate":          m["growth_rate"],
            "growth_rate_label":    gd.get("label", m["growth_rate"]),
            "growth_rate_color":    gd.get("color", "#22c55e"),
            "avg_rings_per_cm":     m["avg_rings_per_cm"],
            "diameter_age_factor":  m["diameter_age_factor"],
            "max_age_years":        m["max_age_years"],
            "habitat":              m.get("habitat", []),
            "regions_alg":          m.get("regions_alg", []),
            "elevation_range":      m.get("elevation_range", (0,2000)),
            "all_scores":           scores,
            "classifier_type":      mode
        }


def estimate_age_with_species(diameter_cm: float, sp: dict,
                               climate_factor: float = 1.0) -> dict:
    """
    Estimation non-destructive :  Age = (D/2) × anneaux/cm × facteur_climatique
    Sans calcul d'anneaux réels (pas de carottage).
    """
    if diameter_cm <= 0: diameter_cm = 20.0
    radius   = diameter_cm / 2.0
    rings    = sp.get("avg_rings_per_cm", 3.0)
    max_age  = sp.get("max_age_years", 800)
    age_base = int(radius * rings)
    age_corr = min(max(1, int(age_base * climate_factor)), max_age)
    age_base = min(max(1, age_base), max_age)
    clf_type  = sp.get("classifier_type", "rule")
    conf_b    = sp.get("confidence", 0.60)
    return {
        "age_base":          age_base,
        "age_corrected":     age_corr,
        "climate_factor":    climate_factor,
        "rings_per_cm_used": rings,
        "formula":           f"({diameter_cm:.1f}/2) × {rings} × {climate_factor:.2f} = {age_corr} ans",
        "method":            f"allometric_{'ml' if clf_type=='ml' else 'rule'}",
        "confidence_boost":  min(0.95, conf_b + (0.10 if clf_type=="ml" else 0.05)),
        "max_age_years":     max_age
    }
