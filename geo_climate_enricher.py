"""
geo_climate_enricher.py — TreeAge AI v3.1
Enrichissement Géo-Climatique — Algérie (Nord/Sud/Est/Ouest)
Sources: Open-Meteo (gratuit, sans clé API)
"""
import requests, logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

KOPPEN_ZONES = {
    "Csa":  {"label":"Méditerranéen chaud",       "growth_modifier":1.00,
              "description":"Étés chauds secs, hivers doux pluvieux",
              "regions":["Alger","Oran","Tell côtier","Béjaïa","Skikda"]},
    "Csb":  {"label":"Méditerranéen montagnard",  "growth_modifier":1.08,
              "description":"Altitude, bonnes précipitations",
              "regions":["Kabylie","Tell atlasique","Atlas blidéen"]},
    "BSh":  {"label":"Semi-aride chaud (steppe)", "growth_modifier":0.78,
              "description":"Précipitations faibles, stress hydrique",
              "regions":["Hauts Plateaux sud","Pré-Sahara","Biskra"]},
    "BSk":  {"label":"Semi-aride froid (steppe)", "growth_modifier":0.85,
              "description":"Froid en hiver, sécheresse estivale",
              "regions":["Sétif","Batna","Djelfa","Tiaret","Aurès"]},
    "BWh":  {"label":"Désertique chaud (Sahara)", "growth_modifier":0.40,
              "description":"Hyper-aride, chaleur extrême",
              "regions":["Grand Erg","Tamanrasset","Adrar","Illizi"]},
    "BWk":  {"label":"Désertique froid",          "growth_modifier":0.45,
              "description":"Aride, hivers froids, Sahara septentrional",
              "regions":["Ghardaïa","Ouargla","El Oued"]},
    "Cfa":  {"label":"Continental humide chaud",  "growth_modifier":1.25,
              "description":"Humide toute l'année, croissance optimale",
              "regions":["Jijel","El Tarf humide"]},
    "Dfb":  {"label":"Continental froid humide",  "growth_modifier":1.10,
              "description":"Hivers enneigés, étés frais",
              "regions":["Djurdjura","Aurès hauts","Chélia"]},
}

SOIL_PROFILES = {
    "clay":       {"label_fr":"Sol Argileux",           "growth_modifier":1.15,
                   "description":"Retient l'eau, riche. Tell est-central."},
    "sandy":      {"label_fr":"Sol Sableux",            "growth_modifier":0.78,
                   "description":"Drainage rapide. Erg et Sahara Nord."},
    "loam":       {"label_fr":"Sol Limoneux (Fertile)", "growth_modifier":1.22,
                   "description":"Équilibre optimal. Vallées et Tell côtier."},
    "rocky":      {"label_fr":"Sol Rocheux",            "growth_modifier":0.68,
                   "description":"Peu de nutriments. Massifs montagneux."},
    "calcareous": {"label_fr":"Sol Calcaire (Rendzine)","growth_modifier":0.88,
                   "description":"pH élevé, typique Hauts Plateaux algériens."},
    "erg_sand":   {"label_fr":"Sable de Dune (Erg)",   "growth_modifier":0.35,
                   "description":"Erg saharien, quasi-stérile."},
    "alluvial":   {"label_fr":"Sol Alluvial (Oued)",   "growth_modifier":1.30,
                   "description":"Très fertile, dépôts alluviaux des oueds."},
}


class OpenMeteoClient:
    BASE_URL = "https://api.open-meteo.com/v1"
    TIMEOUT  = 8

    def get_current(self, lat, lng):
        try:
            r = requests.get(f"{self.BASE_URL}/forecast", timeout=self.TIMEOUT, params={
                "latitude":lat,"longitude":lng,"timezone":"auto",
                "current":["temperature_2m","relative_humidity_2m","wind_speed_10m","precipitation"]})
            r.raise_for_status()
            c = r.json().get("current",{})
            return {"temperature_c":c.get("temperature_2m"),"humidity_pct":c.get("relative_humidity_2m"),
                    "wind_speed_kmh":c.get("wind_speed_10m"),"precipitation_mm":c.get("precipitation")}
        except Exception as e:
            logger.warning(f"Open-Meteo current: {e}"); return None

    def get_annual(self, lat, lng):
        try:
            end   = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now()-timedelta(days=365)).strftime("%Y-%m-%d")
            r = requests.get(f"{self.BASE_URL}/archive", timeout=self.TIMEOUT, params={
                "latitude":lat,"longitude":lng,"start_date":start,"end_date":end,
                "timezone":"auto","daily":["temperature_2m_max","temperature_2m_min","precipitation_sum"]})
            r.raise_for_status()
            d = r.json().get("daily",{})
            tmax = [t for t in (d.get("temperature_2m_max") or []) if t is not None]
            tmin = [t for t in (d.get("temperature_2m_min") or []) if t is not None]
            prec = [p for p in (d.get("precipitation_sum") or []) if p is not None]
            am   = sum(tmax)/len(tmax) if tmax else None
            ami  = sum(tmin)/len(tmin) if tmin else None
            return {
                "avg_temp_max_c": round(am,1) if am else None,
                "avg_temp_min_c": round(ami,1) if ami else None,
                "avg_temp_c":     round((am+ami)/2,1) if am and ami else None,
                "annual_rainfall_mm": round(sum(prec),1) if prec else None,
                "dry_months": sum(1 for i,m in enumerate([sum(prec[j] for j in range(len(prec)) if (j%365)*12//365==i)
                                   for i in range(12)]) if m < 30)
            }
        except Exception as e:
            logger.warning(f"Open-Meteo archive: {e}"); return None


class SoilEstimator:
    def estimate(self, lat, lng, elev=0):
        t = self._classify(lat, lng, elev)
        p = dict(SOIL_PROFILES[t]); p["soil_type"] = t
        return p

    def _classify(self, lat, lng, elev):
        if lat < 24: return "erg_sand"
        if lat < 29: return "sandy"
        if lat < 31: return "calcareous" if elev>400 else "sandy"
        if lat < 33: return "rocky" if elev>1500 else "calcareous"
        if lat < 35:
            return "loam" if lng<1 else "calcareous"
        if lat < 36.5:
            if elev>1800: return "rocky"
            if elev>1000: return "clay"
            return "loam" if lng>6 else "clay"
        return "alluvial" if lng>5 else "loam"


class GeoClimateEnricher:
    def __init__(self):
        self.meteo = OpenMeteoClient()
        self.soil  = SoilEstimator()

    def enrich(self, lat: float, lng: float, elevation_m: float = 0) -> dict:
        logger.info(f"🌍 Enrich: {lat:.4f},{lng:.4f}")
        weather   = self.meteo.get_current(lat, lng)
        annual    = self.meteo.get_annual(lat, lng)
        soil_data = self.soil.estimate(lat, lng, elevation_m)
        zone      = self._koppen(lat, lng, annual, elevation_m)
        factor    = self._growth_factor(annual, zone, soil_data)
        ref       = self._ref(lat)
        return {
            "latitude": lat, "longitude": lng, "elevation_m": elevation_m,
            "algeria_region": self._region(lat, lng),
            "temperature_c":     (weather or {}).get("temperature_c") or ref["t"],
            "humidity_pct":      (weather or {}).get("humidity_pct"),
            "wind_speed_kmh":    (weather or {}).get("wind_speed_kmh"),
            "avg_temp_c":        (annual or {}).get("avg_temp_c") or ref["t"],
            "avg_temp_max_c":    (annual or {}).get("avg_temp_max_c"),
            "avg_temp_min_c":    (annual or {}).get("avg_temp_min_c"),
            "annual_rainfall_mm":(annual or {}).get("annual_rainfall_mm") or ref["r"],
            "dry_months":        (annual or {}).get("dry_months"),
            "climate_zone":      zone,
            "climate_zone_label":KOPPEN_ZONES.get(zone,{}).get("label","Inconnu"),
            "climate_zone_desc": KOPPEN_ZONES.get(zone,{}).get("description",""),
            "climate_regions":   KOPPEN_ZONES.get(zone,{}).get("regions",[]),
            "soil_type":         soil_data["soil_type"],
            "soil_label_fr":     soil_data["label_fr"],
            "soil_description":  soil_data["description"],
            "soil_growth_modifier": soil_data["growth_modifier"],
            "climate_growth_factor": factor,
            "growth_factor_explanation": self._explain(factor, self._region(lat,lng)),
            "data_sources": (["open-meteo-current"] if weather else []) +
                            (["open-meteo-archive"] if annual  else []) +
                            ["soil-pedology-dz","koppen-dz"],
            "enrichment_success": True
        }

    def enrich_fallback(self, lat=None, lng=None) -> dict:
        return {
            "latitude":lat,"longitude":lng,"elevation_m":None,
            "algeria_region":"Hauts Plateaux (estimation)",
            "temperature_c":16.0,"humidity_pct":None,"wind_speed_kmh":None,
            "avg_temp_c":14.5,"avg_temp_max_c":None,"avg_temp_min_c":None,
            "annual_rainfall_mm":380,"dry_months":5,
            "climate_zone":"BSk","climate_zone_label":"Semi-aride froid (Hauts Plateaux)",
            "climate_zone_desc":"Froid en hiver, sécheresse estivale",
            "climate_regions":["Sétif","Batna","Djelfa","Constantine"],
            "soil_type":"calcareous","soil_label_fr":"Sol Calcaire (Rendzine)",
            "soil_description":"Typique Hauts Plateaux algériens",
            "soil_growth_modifier":0.88,
            "climate_growth_factor":0.87,
            "growth_factor_explanation":"Conditions semi-arides — croissance modérée",
            "data_sources":["fallback"],"enrichment_success":False
        }

    def _region(self, lat, lng):
        if lat<24:  return "Sahara profond (Hoggar/Tassili)"
        if lat<27:  return "Grand Sud (Tamanrasset/Illizi)"
        if lat<29:  return "Sahara central (Adrar/In Salah)"
        if lat<31:  return "Sahara nord (Ghardaïa/El Oued)"
        if lat<33:  return "Atlas Saharien (Laghouat/Naâma)"
        if lat<34:  return "Hauts Plateaux sud (Djelfa/El Bayadh)"
        if lat<35:
            return ("Hauts Plateaux est (Batna/Aurès)" if lng>5
                    else "Hauts Plateaux ouest (Tiaret/Saïda)")
        if lat<36:
            if lng>6:   return "Tell sud-est (Sétif/Guelma)"
            if lng<1:   return "Tell sud-ouest (Tlemcen/Mascara)"
            return "Tell central (Médéa/Bouira)"
        if lat<36.8:
            if lng>7:   return "Tell est (Constantine/Annaba/Skikda)"
            if lng<0:   return "Tell ouest (Oran/Mostaganem)"
            if 3.5<lng<5.8: return "Kabylie (Tizi Ouzou/Béjaïa)"
            return "Tell centre (Alger/Blida/Tipaza)"
        return "Littoral méditerranéen (côte nord)"

    def _koppen(self, lat, lng, annual, elev=0):
        t = (annual or {}).get("avg_temp_c") or (32.0-(lat-20)*0.9 - elev*0.0065)
        r = (annual or {}).get("annual_rainfall_mm") or self._ref(lat)["r"]
        if lat<28: return "BWh"
        if lat<31: return "BWh" if t>20 else "BWk"
        if lat<33: return "BSh" if t>18 else "BSk"
        if lat<35: return "BSk"
        if lat<36 and elev>1500: return "Dfb" if r>700 else "Csb"
        if lat<36: return "Csb" if r>600 else "BSk"
        if lng>5.5 and r>700: return "Cfa" if (annual or {}).get("dry_months",4)<2 else "Csb"
        return "Csa"

    def _growth_factor(self, annual, zone, soil):
        zm = KOPPEN_ZONES.get(zone,{}).get("growth_modifier",1.0)
        r  = (annual or {}).get("annual_rainfall_mm") or 350
        rm = (0.40 if r<100 else 0.60 if r<200 else 0.80 if r<350
              else 1.00 if r<550 else 1.15 if r<800 else 1.25 if r<1200 else 1.30)
        sm = soil.get("growth_modifier",1.0)
        return round(zm*0.50 + rm*0.30 + sm*0.20, 3)

    def _explain(self, f, region):
        if f>=1.25: return f"Excellentes conditions ({region}) → croissance très rapide"
        if f>=1.10: return f"Conditions favorables ({region}) → bonne croissance"
        if f>=0.95: return f"Conditions normales ({region}) → croissance standard"
        if f>=0.75: return f"Conditions difficiles ({region}) → sécheresse, croissance ralentie"
        if f>=0.50: return f"Conditions arides ({region}) → croissance très lente"
        return f"Conditions désertiques ({region}) → croissance extrêmement lente"

    def _ref(self, lat):
        if lat<25: return {"t":35.0,"r":15}
        if lat<30: return {"t":28.0,"r":50}
        if lat<32: return {"t":22.0,"r":120}
        if lat<34: return {"t":18.0,"r":200}
        if lat<36: return {"t":15.5,"r":380}
        return {"t":17.5,"r":600}
