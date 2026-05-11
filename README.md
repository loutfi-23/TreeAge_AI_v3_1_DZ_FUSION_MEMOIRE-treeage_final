# 🌳 TreeAge AI v3.1 — DZ Edition (Fusion Mémoire + Application)
## Estimation Automatique de l'Âge des Arbres à partir d'Images

> **Mémoire Master 2 — Gestion et Analyse des Données Massives**
> Université Badji Mokhtar – Annaba — Faculté de Technologie — Département d'Informatique
> Présenté par : **Bouchareb Loutfi**
> Encadrants : Pr. Sari Toufik · Mme Oualhi Ouarda
> Année universitaire : **2025/2026**

---

## 🎯 À propos de cette version fusionnée

Cette release combine **deux artefacts** en une seule application déployable :

1. **TreeAge AI v3.1** — l'application web full-stack Flask + MongoDB d'estimation non-destructive d'âge des arbres en Algérie.
2. **Le mémoire académique** complet (Chapitres 1 à 4) intégré comme module navigable directement dans l'interface.

L'utilisateur authentifié peut donc consulter à la fois le système opérationnel **et** la documentation scientifique qui le sous-tend, depuis un seul et même produit. Une page « Page de garde » et une page par chapitre sont accessibles via le menu latéral, ainsi qu'une API JSON.

---

## 🚀 Installation rapide

```bash
# 1. Créer environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Installer dépendances
pip install -r requirements.txt

# 3. Initialiser MongoDB
python init_db.py

# 4. Lancer l'application
python app.py
```

Ouvrir : **http://localhost:5001**

---

## 🔑 Identifiants par défaut

| Rôle   | Email                    | Mot de passe         |
|--------|--------------------------|----------------------|
| Admin  | admin@treepage.fr        | AdminPassword@2026   |
| Viewer | lotfi@email.com          | 123                  |

---

## 📖 Module Mémoire académique

Une fois connecté, le menu latéral expose une nouvelle section **« Mémoire Master 2 »** :

| Route                          | Description                                              |
|--------------------------------|----------------------------------------------------------|
| `/memoire`                     | Page de garde + résumé bilingue + table des matières     |
| `/memoire/chapitre/1`          | Les forêts en Algérie et leur importance                 |
| `/memoire/chapitre/2`          | Méthodes d'estimation de l'âge + traitement d'images     |
| `/memoire/chapitre/3`          | Conception du système (UML, MongoDB, IA, UI)             |
| `/memoire/chapitre/4`          | Modélisation Relationnel + NoSQL + Implémentation        |
| `/api/memoire/chapter/<num>`   | API JSON pour récupérer un chapitre programmatiquement   |

Chaque page de chapitre offre :
- Sommaire latéral cliquable (ancres internes)
- Navigation entre chapitres (← précédent / suivant →)
- Carte d'auteur sticky en colonne droite
- Affichage typographique soigné (DM Serif + Inter)

La page d'accueil `/memoire` inclut une **table de correspondance Mémoire ↔ Application** qui montre, pour chaque section théorique, l'implémentation concrète dans le code source.

---

## 🌲 8 Espèces Algériennes supportées

| Espèce | Nom scientifique | Vitesse |
|--------|-----------------|---------|
| 🌲 Pin d'Alep | Pinus halepensis | Rapide |
| 🌳 Cèdre de l'Atlas | Cedrus atlantica | Lent |
| 🌳 Chêne vert | Quercus ilex | Lent |
| 🌳 Chêne-liège | Quercus suber | Moyen |
| 🌿 Eucalyptus | Eucalyptus globulus | Très rapide |
| 🌲 Genévrier thurifère | Juniperus thurifera | Très lent |
| 🌲 Thuya de Berbérie | Tetraclinis articulata | Lent |
| 🌳 Peuplier noir | Populus nigra | Rapide |

---

## 🔬 Pipeline d'Estimation (Non-Destructif)

```
Image arbre
    │
    ├─ EXIF → GPS automatique (lat/lng)
    │         OU coordonnées manuelles
    │
    ├─ OpenCV → Estimation diamètre tronc
    │
    ├─ CNN/ML → Classification espèce (8 espèces DZ)
    │           Accuracy: ~92% (cross-validation 5-fold)
    │
    ├─ Open-Meteo → Données climatiques réelles
    │   + Pédologie → Type de sol (Algérie)
    │   → Zone Köppen + Facteur de croissance
    │
    └─ Formule allométrique:
       Age = (Diamètre/2) × (anneaux/cm) × facteur_climatique

       SANS MESURE DES ANNEAUX RÉELS
       SANS CAROTTAGE DE L'ARBRE
```

---

## 📁 Structure du Projet (mise à jour)

```
treeage_final/
├── app.py                    # Application Flask principale (+ routes /memoire)
├── species_classifier.py     # Classificateur 8 espèces DZ
├── geo_climate_enricher.py   # Enrichissement géo-climatique Algérie
├── thesis_content.py         # 🆕 Module mémoire (métadonnées + loader)
├── thesis_data.json          # 🆕 Contenu structuré des 4 chapitres
├── train_species_model.py    # Ré-entraîner le modèle
├── init_db.py                # Initialiser MongoDB
├── requirements.txt
├── models/
│   └── species_model.pkl     # Modèle pré-entraîné (GradientBoosting)
├── templates/
│   ├── login.html
│   ├── dashboard.html        # (lien vers Mémoire ajouté dans la sidebar)
│   ├── about.html            # (lien vers Mémoire ajouté dans la sidebar)
│   ├── admin.html
│   ├── admin_users.html
│   ├── add_user.html
│   ├── validations.html
│   ├── map_view.html
│   ├── pricing.html
│   ├── memoire_home.html     # 🆕 Page de garde du mémoire
│   └── memoire_chapter.html  # 🆕 Vue par chapitre
├── static/
│   └── manifest.json
└── uploads/                  # Images uploadées (auto-créé)
```

---

## 🔗 Correspondance Mémoire ↔ Application

| Section du mémoire                  | Implémentation                                          | Fichier                          |
|-------------------------------------|---------------------------------------------------------|----------------------------------|
| Chap. 1 — Forêts en Algérie         | Catalogue de 8 espèces algériennes avec régions         | `species_classifier.py`          |
| Chap. 2.1 — Diamètre du tronc (DBH) | OpenCV (Canny + contours) — `get_diameter()`            | `app.py`                         |
| Chap. 2.3 — Imagerie moderne        | Gradient Boosting sur features visuels (~92% accuracy)  | `species_classifier.py`          |
| Chap. 2.4 — Traitement d'images     | Niveaux de gris, flou gaussien, détection de bords      | `app.py` (get_diameter)          |
| Chap. 3.1 — Architecture 3-tiers    | Flask + Jinja2/HTML + MongoDB                           | `app.py` + `templates/`          |
| Chap. 3.5 — Conception MongoDB      | Collections + index 2dsphere géospatial                 | `init_db.py`                     |
| Chap. 3.7 — Cas d'utilisation UML   | Rôles ADMIN/VIEWER + décorateurs                        | `app.py`                         |
| Chap. 4 — Modélisation NoSQL        | Schéma documents avec embedding                         | `app.py` (run_pipeline)          |
| Données géospatiales                | Open-Meteo + zones Köppen + sols                        | `geo_climate_enricher.py`        |
| Méthode allométrique                | `Age = (D/2) × anneaux/cm × facteur_climatique`         | `species_classifier.py`          |

---

## 🛠️ Ré-entraîner le modèle

```bash
python train_species_model.py
# → Sauvegarde models/species_model.pkl
# → Accuracy attendue: ~92%
```

---

## ⚙️ Variables d'environnement

```bash
MONGODB_URI=mongodb://localhost:27017/   # URI MongoDB
SECRET_KEY=votre-clé-secrète             # Clé Flask sessions
PORT=5001                                # Port (défaut 5001)
FLASK_DEBUG=false                        # Mode debug
```

---

## 📊 Base de données MongoDB

Collection `estimations` — champs clés après chaque prédiction :
- `estimated_age` — âge estimé (ans)
- `diameter_cm` — diamètre tronc
- `species` / `species_common_fr` — espèce
- `location.lat` / `location.lng` — coordonnées GPS
- `algeria_region` — région algérienne identifiée
- `climate_zone` / `climate_growth_factor` — données climatiques
- `soil_type` / `soil_label_fr` — type de sol
- `is_validated` / `corrected_age` — validation admin

---

## 🩺 Endpoint santé

`GET /health` → JSON avec statut MongoDB, modules IA, et présence du module mémoire :

```json
{
  "status": "ok",
  "db": "connected",
  "species_module": true,
  "climate_module": true,
  "thesis_module": true,
  "pipeline": "3.1",
  "...": "..."
}
```
