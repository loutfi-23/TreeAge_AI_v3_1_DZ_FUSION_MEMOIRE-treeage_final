"""
thesis_content.py — TreeAge AI v3.1
Module académique : intègre le mémoire de Master à l'application.

Auteur du mémoire : Bouchareb Loutfi
Université Badji Mokhtar – Annaba (UBMA)
Faculté Technologie — Département Informatique
Spécialité : Gestion et Analyse des Données Massives
Encadrants : Pr. Sari Toufik · Mme Oualhi Ouarda
Année : 2025/2026
"""
import json
import os

# ─────────────────────────────────────────────────────────────
# MÉTADONNÉES DU MÉMOIRE
# ─────────────────────────────────────────────────────────────
THESIS_META = {
    "title_fr": "Estimation Automatique de l'Âge d'Arbres à partir d'Images",
    "title_en": "Automatic Tree Age Estimation from Images",
    "author": "Bouchareb Loutfi",
    "university": "Université Badji Mokhtar – Annaba",
    "university_ar": "جامعة باجي مختار – عنابة",
    "faculty": "Faculté de Technologie",
    "department": "Département d'Informatique",
    "domain": "Mathématique-Informatique",
    "specialty": "Gestion et Analyse des Données Massives",
    "diploma": "Master",
    "year": "2025/2026",
    "supervisors": [
        {"name": "Sari Toufik", "title": "Professeur", "role": "Encadrant"},
        {"name": "Oualhi Ouarda", "title": "—", "role": "Co-encadrante"},
    ],
    "keywords_fr": [
        "Estimation d'âge des arbres",
        "Apprentissage profond (Deep Learning)",
        "Réseaux de neurones convolutionnels (CNN)",
        "Vision par ordinateur",
        "Traitement d'images",
        "MongoDB",
        "Flask",
        "Big Data",
        "Forêts algériennes",
        "Méthode allométrique",
        "Non-destructif",
    ],
}

ABSTRACT_FR = (
    "Dans un contexte marqué par les enjeux environnementaux et la nécessité d'une gestion "
    "durable des ressources forestières, l'estimation de l'âge des arbres constitue un indicateur "
    "essentiel pour l'analyse de la croissance, de la productivité et de la santé des écosystèmes. "
    "Les méthodes traditionnelles, telles que le comptage des cernes de croissance ou le carottage, "
    "bien que précises, présentent des limites importantes en raison de leur caractère invasif, "
    "coûteux et difficilement applicable à grande échelle.\n\n"
    "Le présent mémoire propose la conception et la réalisation d'un système intelligent "
    "d'estimation automatique de l'âge des arbres à partir d'images numériques, en s'appuyant "
    "sur les techniques d'intelligence artificielle, notamment l'apprentissage profond. "
    "L'approche adoptée repose sur l'extraction de caractéristiques visuelles pertinentes "
    "issues des images, telles que la texture de l'écorce et les motifs de croissance, "
    "suivie de leur exploitation par un modèle prédictif.\n\n"
    "Afin d'améliorer la précision des résultats, des données géospatiales telles que la "
    "localisation, les conditions climatiques et les caractéristiques du sol sont intégrées "
    "dans le processus d'analyse. Le système est implémenté sous forme d'une application web "
    "full-stack basée sur Flask, React et MongoDB, intégrant un système d'authentification et "
    "une interface utilisateur intuitive.\n\n"
    "Les résultats obtenus démontrent la pertinence de l'approche proposée, offrant une "
    "solution non destructive, automatisée et scalable pour la gestion intelligente des "
    "ressources forestières."
)

ABSTRACT_EN = (
    "In a context characterized by environmental challenges and the need for sustainable "
    "forest resource management, tree age estimation is a key indicator for analyzing growth, "
    "productivity, and ecosystem health. Traditional methods such as tree-ring counting and "
    "coring, although accurate, are invasive, costly, and difficult to apply at large scale.\n\n"
    "This thesis presents the design and implementation of an intelligent system for automatic "
    "tree age estimation based on digital images, using artificial intelligence techniques, "
    "particularly deep learning. The proposed approach relies on extracting relevant visual "
    "features such as bark texture and growth patterns, which are then processed by a "
    "predictive model.\n\n"
    "To enhance accuracy, geospatial data including location, climate conditions, and soil "
    "characteristics are integrated into the analysis. The system is implemented as a "
    "full-stack web application using Flask, React, and MongoDB, with an authentication "
    "system and a user-friendly interface.\n\n"
    "The results demonstrate the effectiveness of the proposed approach, providing a "
    "non-destructive, automated, and scalable solution for smart forest resource management."
)

# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES CHAPITRES
# ─────────────────────────────────────────────────────────────
_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_data.json")
_CHAPTERS = None


def _load():
    """Charge les chapitres depuis le JSON (lazy loading + cache)."""
    global _CHAPTERS
    if _CHAPTERS is None:
        try:
            with open(_DATA_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Convert string keys to int
            _CHAPTERS = {int(k): v for k, v in raw.items()}
        except FileNotFoundError:
            _CHAPTERS = {}
    return _CHAPTERS


def get_all_chapters():
    """Retourne tous les chapitres (dict {num: chapter})."""
    return _load()


def get_chapter(num: int):
    """Retourne un chapitre donné, ou None s'il n'existe pas."""
    return _load().get(num)


def get_chapter_titles():
    """Retourne la liste ordonnée [(num, title)] pour la navigation."""
    chapters = _load()
    return [(n, chapters[n]["title"]) for n in sorted(chapters.keys())]


def get_section(chapter_num: int, section_id: str):
    """Retourne une section spécifique d'un chapitre."""
    ch = get_chapter(chapter_num)
    if not ch:
        return None
    for sec in ch.get("sections", []):
        if sec.get("id") == section_id:
            return sec
    return None


# ─────────────────────────────────────────────────────────────
# RÉSUMÉ EXÉCUTIF — alignement mémoire ↔ application
# ─────────────────────────────────────────────────────────────
THESIS_TO_APP_MAPPING = [
    {
        "thesis_section": "Chap. 1 — Forêts en Algérie",
        "app_implementation": "Catalogue de 8 espèces algériennes (Pin d'Alep, Cèdre, Chêne vert, "
                              "Chêne-liège, Eucalyptus, Genévrier, Thuya, Peuplier) avec régions",
        "code_file": "species_classifier.py",
    },
    {
        "thesis_section": "Chap. 2.1 — Méthode du diamètre du tronc (DBH)",
        "app_implementation": "Estimation du diamètre par OpenCV (Canny + contours) — fonction get_diameter()",
        "code_file": "app.py",
    },
    {
        "thesis_section": "Chap. 2.3 — Approches modernes basées sur l'imagerie",
        "app_implementation": "Classifieur Gradient Boosting sur features visuels (8 espèces, ~92% accuracy)",
        "code_file": "species_classifier.py + train_species_model.py",
    },
    {
        "thesis_section": "Chap. 2.4 — Fondements du traitement d'images",
        "app_implementation": "Pipeline OpenCV : niveaux de gris, flou gaussien, détection de bords Canny",
        "code_file": "app.py (get_diameter)",
    },
    {
        "thesis_section": "Chap. 3.1 — Architecture 3-tiers",
        "app_implementation": "Flask (backend) + Jinja2/HTML (frontend) + MongoDB (data layer)",
        "code_file": "app.py + templates/",
    },
    {
        "thesis_section": "Chap. 3.5 — Conception MongoDB",
        "app_implementation": "Collections : users, estimations, forests, models_ia, about_profiles "
                              "+ index 2dsphere géospatial",
        "code_file": "init_db.py",
    },
    {
        "thesis_section": "Chap. 3.7 — Diagramme de cas d'utilisation",
        "app_implementation": "Rôles ADMIN / VIEWER avec décorateurs login_required / admin_required",
        "code_file": "app.py",
    },
    {
        "thesis_section": "Chap. 4 — Modélisation NoSQL + implémentation",
        "app_implementation": "Schéma documents MongoDB avec embedding (location, climate, species)",
        "code_file": "app.py (run_pipeline)",
    },
    {
        "thesis_section": "Données géospatiales (Chap. 1 + 3)",
        "app_implementation": "Enrichissement Open-Meteo : zones Köppen + sol + facteur de croissance",
        "code_file": "geo_climate_enricher.py",
    },
    {
        "thesis_section": "Méthode allométrique non-destructive",
        "app_implementation": "Âge = (D/2) × anneaux/cm × facteur_climatique — sans carottage",
        "code_file": "species_classifier.py (estimate_age_with_species)",
    },
]
