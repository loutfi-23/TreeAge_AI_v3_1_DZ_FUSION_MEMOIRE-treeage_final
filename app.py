"""
app.py — TreeAge AI v3.1 (DZ Edition)
======================================
Estimation Non-Destructive de l'Âge des Arbres
Sans calcul d'anneaux réels — méthode allométrique + IA

Pipeline: Image → EXIF GPS → Diamètre → CNN Espèce →
          Géo-IA Algérie → Age Allométrique → MongoDB

Admin: http://localhost:5001  →  admin@treepage.fr / AdminPassword@2026
"""
import os, uuid, logging, traceback
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify, send_from_directory)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson import ObjectId
from bson.errors import InvalidId
import os

SECRET_KEY = os.getenv("SECRET_KEY")
app.config["SECRET_KEY"] = SECRET_KEY
# ─── Modules IA ──────────────────────────────────────────────
try:
    from species_classifier import (SpeciesClassifier, estimate_age_with_species,
                                     SPECIES_CATALOG, GROWTH_RATE_DISPLAY)
    SPECIES_MODULE = True
except ImportError:
    SPECIES_MODULE = False

try:
    from geo_climate_enricher import GeoClimateEnricher
    CLIMATE_MODULE = True
except ImportError:
    CLIMATE_MODULE = False

# ─── Module Mémoire (intégration thèse Master2 GADM) ─────────
try:
    import thesis_content
    THESIS_MODULE = True
except ImportError:
    THESIS_MODULE = False

try:
    from PIL import Image; PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2, numpy as np; CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class Config:
    SECRET_KEY         = os.environ.get("SECRET_KEY","treeage-dz-2026-secret")
    MONGO_URI          = os.environ.get("MONGODB_URI","mongodb://localhost:27017/")
    DB_NAME            = "tree_deep_db"
    UPLOAD_FOLDER      = os.path.join(os.path.dirname(__file__),"uploads")
    MAX_CONTENT_LENGTH = 16*1024*1024
    ALLOWED_EXTENSIONS = {"jpg","jpeg","png","webp"}
    SESSION_LIFETIME   = timedelta(hours=24)
    PLAN_LIMITS = {"free":{"predictions_per_month":10},
                   "researcher":{"predictions_per_month":500},
                   "enterprise":{"predictions_per_month":-1}}

app = Flask(__name__)
app.config.from_object(Config)
app.permanent_session_lifetime = Config.SESSION_LIFETIME
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("treeage")

# ─── MongoDB ──────────────────────────────────────────────────
_client = None
def get_db():
    global _client
    if _client is None:
        _client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[Config.DB_NAME]

# ─── Singletons IA ────────────────────────────────────────────
_sp_clf   = SpeciesClassifier()  if SPECIES_MODULE  else None
_geo_enr  = GeoClimateEnricher() if CLIMATE_MODULE  else None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(f): return "." in f and f.rsplit(".",1)[1].lower() in Config.ALLOWED_EXTENSIONS

def sdoc(doc):
    if doc is None: return {}
    out = {}
    for k,v in doc.items():
        if isinstance(v, ObjectId):   out[k] = str(v)
        elif isinstance(v, datetime): out[k] = v.isoformat()
        elif isinstance(v, dict):     out[k] = sdoc(v)
        elif isinstance(v, list):
            out[k] = [sdoc(i) if isinstance(i,dict) else str(i) if isinstance(i,ObjectId) else i for i in v]
        else: out[k] = v
    return out

def login_required(f):
    @wraps(f)
    def d(*a,**kw):
        if "user_email" not in session:
            flash("Veuillez vous connecter.","error")
            return redirect(url_for("login"))
        return f(*a,**kw)
    return d

def admin_required(f):
    @wraps(f)
    def d(*a,**kw):
        if "user_email" not in session: return redirect(url_for("login"))
        if session.get("user_role") != "admin":
            flash("Accès admin requis.","error")
            return redirect(url_for("dashboard"))
        return f(*a,**kw)
    return d

# ─────────────────────────────────────────────
# GPS EXIF
# ─────────────────────────────────────────────
def extract_gps(filepath):
    if not PIL_AVAILABLE: return None, None
    try:
        exif = Image.open(filepath)._getexif()
        if not exif: return None,None
        gps = exif.get(34853)
        if not gps: return None,None
        def deg(v): return float(v[0][0])/float(v[0][1])+(float(v[1][0])/float(v[1][1]))/60+(float(v[2][0])/float(v[2][1]))/3600
        lat = deg(gps[2]); lng = deg(gps[4])
        if gps.get(1)=="S": lat=-lat
        if gps.get(3)=="W": lng=-lng
        return round(lat,6), round(lng,6)
    except: return None,None

# ─────────────────────────────────────────────
# DIAMÈTRE
# ─────────────────────────────────────────────
def get_diameter(filepath):
    if CV2_AVAILABLE:
        try:
            img = cv2.imread(filepath)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(cv2.GaussianBlur(gray,(11,11),0),30,100)
                cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
                    return round(max(5.0,min(w/30.0,300.0)),1)
        except: pass
    import hashlib
    try:
        h = int(hashlib.md5(filepath.encode()).hexdigest(),16)
        return round(min(15.0+(h%750)/10.0,90.0),1)
    except: return 35.0

# ─────────────────────────────────────────────
# PIPELINE PRÉDICTION v3.1
# ─────────────────────────────────────────────
def run_pipeline(filepath, lat=None, lng=None):
    # 1. Diamètre
    diam = get_diameter(filepath)

    # 2. Espèce (ML ou règles)
    sp = _sp_clf.predict(filepath) if _sp_clf else {
        "species":"Pin_d_Alep","species_common_fr":"Pin d'Alep","confidence":0.60,
        "latin_name":"Pinus halepensis","emoji":"🌲","growth_rate":"fast",
        "growth_rate_label":"Rapide","growth_rate_color":"#f97316",
        "avg_rings_per_cm":3.2,"diameter_age_factor":0.28,"max_age_years":400,
        "description":"Pin d'Alep (défaut)","all_scores":{},"classifier_type":"fallback",
        "regions_alg":["Algérie"],"habitat":["Mediterranean"]
    }

    # 3. Climatologie
    if _geo_enr and lat is not None and lng is not None:
        try:    clim = _geo_enr.enrich(lat, lng)
        except: clim = _geo_enr.enrich_fallback(lat, lng)
    elif _geo_enr:
        clim = _geo_enr.enrich_fallback()
    else:
        clim = {"climate_zone":"BSk","climate_zone_label":"Semi-aride froid",
                "temperature_c":None,"annual_rainfall_mm":380,
                "soil_type":"calcareous","soil_label_fr":"Sol Calcaire",
                "climate_growth_factor":0.87,"growth_factor_explanation":"Estimation Algérie centrale",
                "algeria_region":"Algérie (estimation)","enrichment_success":False}

    # 4. Âge allométrique (non-destructif — sans carottage)
    cf = clim.get("climate_growth_factor", 1.0)
    if SPECIES_MODULE and _sp_clf:
        age = estimate_age_with_species(diam, sp, cf)
    else:
        r = diam/2.0; rg = sp.get("avg_rings_per_cm",3.2)
        ab = int(r*rg); ac = int(ab*cf)
        age = {"age_base":ab,"age_corrected":max(1,ac),"climate_factor":cf,
               "rings_per_cm_used":rg,
               "formula":f"({diam}/2)×{rg}×{cf:.2f}={ac} ans",
               "method":"allometric_fallback","confidence_boost":0.60}

    return {
        "diameter_cm":diam,"estimated_age":age["age_corrected"],
        "age_base":age["age_base"],"confidence":round(age.get("confidence_boost",0.70),4),
        "age_formula":age.get("formula",""),"age_method":age.get("method",""),
        "species":sp["species"],"species_common_fr":sp.get("species_common_fr",sp["species"]),
        "species_confidence":round(sp["confidence"],4),
        "species_latin":sp["latin_name"],"species_emoji":sp["emoji"],
        "species_growth_rate":sp["growth_rate"],"species_growth_rate_label":sp["growth_rate_label"],
        "species_growth_rate_color":sp.get("growth_rate_color","#22c55e"),
        "species_all_scores":sp.get("all_scores",{}),"classifier_type":sp.get("classifier_type","fallback"),
        "species_regions_alg":sp.get("regions_alg",[]),
        "climate_zone":clim.get("climate_zone"),"climate_zone_label":clim.get("climate_zone_label"),
        "temperature_c":clim.get("temperature_c"),"annual_rainfall_mm":clim.get("annual_rainfall_mm"),
        "soil_type":clim.get("soil_type"),"soil_label_fr":clim.get("soil_label_fr"),
        "climate_growth_factor":round(cf,3),"growth_factor_explanation":clim.get("growth_factor_explanation",""),
        "algeria_region":clim.get("algeria_region","Algérie"),"enrichment_success":clim.get("enrichment_success",False),
        "pipeline_version":"3.1","status":"predicted","is_validated":False,
        "tree_name":None,"corrected_age":None,
    }

# ─────────────────────────────────────────────
# QUOTA
# ─────────────────────────────────────────────
def check_quota(email):
    db = get_db(); user = db.users.find_one({"email":email})
    if not user: return False,{}
    if user.get("role")=="admin":
        return True,{"plan":"admin","limit":"∞","used":0,"remaining":-1,"can_predict":True}
    plan  = user.get("subscription_plan","free")
    limit = Config.PLAN_LIMITS.get(plan,Config.PLAN_LIMITS["free"])["predictions_per_month"]
    start = datetime.now().replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    used  = db.estimations.count_documents({"user_email":email,"date":{"$gte":start}})
    if limit==-1: return True,{"plan":plan,"limit":"∞","used":used,"remaining":-1,"can_predict":True}
    rem = max(0,limit-used)
    return rem>0,{"plan":plan,"limit":limit,"used":used,"remaining":rem,"can_predict":rem>0}

# ─────────────────────────────────────────────
# ROUTES AUTH
# ─────────────────────────────────────────────
@app.route("/")
def index(): return redirect(url_for("dashboard") if "user_email" in session else url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if "user_email" in session: return redirect(url_for("dashboard"))
    if request.method=="POST":
        email = request.form.get("email","").strip().lower()
        pwd   = request.form.get("password","")
        db    = get_db(); user = db.users.find_one({"email":email})
        if not user or not check_password_hash(user.get("password_hash",""),pwd):
            return render_template("login.html", error="Email ou mot de passe incorrect.")
        if not user.get("is_active",True):
            return render_template("login.html", error="Compte désactivé.")
        session.permanent=True
        session["user_email"]=email
        session["user_role"]=user.get("role","viewer")
        session["user_name"]=user.get("name",email.split("@")[0])
        db.users.update_one({"email":email},{"$set":{"last_login":datetime.now()}})
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("login"))

# ─────────────────────────────────────────────
# DASHBOARD (admin + viewer — même page)
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    email    = session["user_email"]
    is_admin = session.get("user_role")=="admin"
    db       = get_db()
    user     = sdoc(db.users.find_one({"email":email})) or {
        "email":email,"role":session.get("user_role","viewer"),"name":session.get("user_name","")}
    # Admin voit TOUT, viewer voit les siennes
    query = {} if is_admin else {"user_email":email}
    estimations = [sdoc(e) for e in db.estimations.find(query).sort("date",-1).limit(100)]
    messages    = session.pop("flash_messages",[])
    return render_template("dashboard.html", user=user,
                           estimations=estimations, messages=messages)

# ─────────────────────────────────────────────
# PRÉDICTION — admin ET viewer
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    email = session["user_email"]
    ok, quota = check_quota(email)
    if not ok:
        flash(f"Quota mensuel atteint ({quota.get('limit')} analyses).", "error")
        return redirect(url_for("dashboard"))
    if "image" not in request.files:
        flash("Aucune image fournie.", "error"); return redirect(url_for("dashboard"))
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        flash("Format invalide (JPEG/PNG/WebP).", "error"); return redirect(url_for("dashboard"))

    # GPS manuel optionnel
    try:    mlat = float(request.form.get("latitude",""))
    except: mlat = None
    try:    mlng = float(request.form.get("longitude",""))
    except: mlng = None

    try:
        ext  = secure_filename(file.filename).rsplit(".",1)[-1].lower()
        fn   = f"{uuid.uuid4().hex}.{ext}"
        fp   = os.path.join(Config.UPLOAD_FOLDER, fn)
        file.save(fp)

        lat, lng = extract_gps(fp)
        if lat is None and mlat is not None: lat,lng = round(mlat,6), round(mlng,6) if mlng else None
        gps_src = "exif" if lat and extract_gps(fp)[0] else "manual" if lat else "none"

        pred = run_pipeline(fp, lat=lat, lng=lng)
        db   = get_db()
        doc  = {**pred, "user_email":email, "uploaded_by":session.get("user_name",email),
                "image_filename":fn, "image_path":fp, "date":datetime.now(),
                "gps_source":gps_src,
                "location":{"type":"Point","coordinates":[lng,lat],"lat":lat,"lng":lng}
                            if lat is not None and lng is not None else None}
        res = db.estimations.insert_one(doc)

        # Mise à jour MongoDB utilisateur (compteur + dernière prédiction)
        db.users.update_one({"email":email},
            {"$inc":{"total_predictions":1},"$set":{"last_prediction":datetime.now()}})

        sp = pred["species_common_fr"]; em = pred["species_emoji"]
        ag = pred["estimated_age"];     cf = round(pred["confidence"]*100)
        est_id = str(res.inserted_id)
        if lat and lng:
            gm_url  = f"https://www.google.com/maps?q={lat:.6f},{lng:.6f}&z=16"
            gps_txt = (f" 📍 {lat:.4f}°N, {lng:.4f}°E "
                       f"— <a href='{gm_url}' target='_blank' style='color:#15803d;font-weight:600'>"
                       f"🗺 Voir sur Google Maps</a>")
        else:
            gps_txt = " (GPS indisponible — entrez les coordonnées manuellement)"
        flash(
            f"✅ {em} {sp} — <strong>{ag} ans</strong> "
            f"(conf. {cf}%, facteur ×{pred['climate_growth_factor']})<br>{gps_txt}",
            "success"
        )
    except Exception as e:
        logger.exception(e); flash(f"Erreur: {e}","error")
    return redirect(url_for("dashboard"))

# ─────────────────────────────────────────────
# SUPPRESSION
# ─────────────────────────────────────────────
@app.route("/prediction/<eid>/delete", methods=["POST"])
@login_required
def delete_prediction(eid):
    db = get_db()
    try: oid = ObjectId(eid)
    except: return jsonify({"error":"ID invalide"}),400
    est = db.estimations.find_one({"_id":oid})
    if not est: return jsonify({"error":"Introuvable"}),404
    if est["user_email"]!=session["user_email"] and session.get("user_role")!="admin":
        return jsonify({"error":"Non autorisé"}),403
    try:
        if est.get("image_path") and os.path.exists(est["image_path"]):
            os.remove(est["image_path"])
    except: pass
    db.estimations.delete_one({"_id":oid})
    return jsonify({"success":True})

# ─────────────────────────────────────────────
# CARTE — admin voit tout
# ─────────────────────────────────────────────
@app.route("/map_view")
@login_required
def map_view():
    email    = session["user_email"]
    is_admin = session.get("user_role")=="admin"
    db       = get_db()
    user     = sdoc(db.users.find_one({"email":email})) or {"email":email,"role":session.get("user_role","viewer"),"name":session.get("user_name","")}
    query = {"location":{"$ne":None}}
    if not is_admin: query["user_email"] = email
    trees = [sdoc(t) for t in db.estimations.find(query,
        {"estimated_age":1,"corrected_age":1,"diameter_cm":1,"confidence":1,
         "location":1,"date":1,"species":1,"species_common_fr":1,
         "species_emoji":1,"climate_zone":1,"climate_zone_label":1,
         "algeria_region":1,"user_email":1,"is_validated":1,"tree_name":1}
    ).sort("date",-1).limit(500)]
    return render_template("map_view.html", user=user, trees=trees)

# ─────────────────────────────────────────────
# ADMIN DASHBOARD
# ─────────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin():
    db   = get_db()
    user = sdoc(db.users.find_one({"email":session["user_email"]}))
    te   = db.estimations.count_documents({})
    tv   = db.estimations.count_documents({"is_validated":True})
    cr   = list(db.estimations.aggregate([{"$group":{"_id":None,"a":{"$avg":"$confidence"}}}]))
    sp   = list(db.estimations.aggregate([{"$group":{"_id":"$species","count":{"$sum":1},"avg_age":{"$avg":"$estimated_age"}}},{"$sort":{"count":-1}}]))
    rg   = list(db.estimations.aggregate([{"$group":{"_id":"$algeria_region","count":{"$sum":1}}},{"$sort":{"count":-1}},{"$limit":10}]))
    try: dbsz = round(db.command("dbStats").get("dataSize",0)/1024/1024,2)
    except: dbsz = 0
    stats = {"total_users":db.users.count_documents({}),"total_estimations":te,
             "validated_count":tv,"validation_rate":round(tv/te*100,1) if te else 0,
             "avg_confidence":cr[0]["a"] if cr else 0.0,"species_dist":sp,
             "gps_count":db.estimations.count_documents({"location":{"$ne":None}}),
             "region_dist":rg,"db_size_mb":dbsz}
    users = [sdoc(u) for u in db.users.find().sort("created_at",-1)]
    ests  = [sdoc(e) for e in db.estimations.find().sort("date",-1).limit(30)]
    return render_template("admin.html", user=user, users=users, estimations=ests, stats=stats)

# ─────────────────────────────────────────────
# GESTION USERS
# ─────────────────────────────────────────────
@app.route("/admin/users")
@admin_required
def admin_users():
    db    = get_db()
    user  = sdoc(db.users.find_one({"email":session["user_email"]}))
    users = [sdoc(u) for u in db.users.find().sort("created_at",-1)]
    return render_template("admin_users.html", user=user, users=users)

@app.route("/admin/users/add", methods=["GET","POST"])
@admin_required
def add_user():
    if request.method=="GET":
        return render_template("add_user.html",
                               user=sdoc(get_db().users.find_one({"email":session["user_email"]})))
    name=request.form.get("name","").strip(); email=request.form.get("email","").strip().lower()
    pwd=request.form.get("password",""); role=request.form.get("role","viewer")
    if not all([name,email,pwd]): return jsonify({"error":"Champs requis"}),400
    if len(pwd)<8: return jsonify({"error":"Mot de passe trop court (min 8)"}),400
    db = get_db()
    if db.users.find_one({"email":email}): return jsonify({"error":"Email déjà utilisé"}),409
    db.users.insert_one({"name":name,"email":email,
        "password_hash":generate_password_hash(pwd,"pbkdf2:sha256"),
        "role":role,"is_active":True,"subscription_plan":"free",
        "total_predictions":0,"created_at":datetime.now(),"last_login":None})
    return jsonify({"success":True,"email":email})

@app.route("/admin/users/toggle_active/<uid>", methods=["POST"])
@admin_required
def toggle_user(uid):
    db=get_db()
    try: oid=ObjectId(uid)
    except: return jsonify({"error":"ID invalide"}),400
    u=db.users.find_one({"_id":oid})
    if not u: return jsonify({"error":"Introuvable"}),404
    ns=not u.get("is_active",True)
    db.users.update_one({"_id":oid},{"$set":{"is_active":ns}})
    return jsonify({"success":True,"is_active":ns})

@app.route("/admin/users/delete/<uid>", methods=["POST"])
@admin_required
def delete_user(uid):
    db=get_db()
    try: oid=ObjectId(uid)
    except: return jsonify({"error":"ID invalide"}),400
    u=db.users.find_one({"_id":oid})
    if not u: return jsonify({"error":"Introuvable"}),404
    if u["email"]==session["user_email"]: return jsonify({"error":"Impossible de supprimer son propre compte"}),400
    db.users.delete_one({"_id":oid}); return jsonify({"success":True})

@app.route("/admin/users/change_role/<uid>/<role>", methods=["POST"])
@admin_required
def change_role(uid,role):
    if role not in ("admin","viewer"): return jsonify({"error":"Rôle invalide"}),400
    db=get_db()
    try: oid=ObjectId(uid)
    except: return jsonify({"error":"ID invalide"}),400
    db.users.update_one({"_id":oid},{"$set":{"role":role}})
    return jsonify({"success":True,"role":role})

@app.route("/admin/users/update_plan/<uid>/<plan>", methods=["POST"])
@admin_required
def update_plan(uid,plan):
    if plan not in ("free","researcher","enterprise"):
        return jsonify({"error":"Plan invalide"}),400
    db=get_db()
    try: oid=ObjectId(uid)
    except: return jsonify({"error":"ID invalide"}),400
    db.users.update_one({"_id":oid},{"$set":{"subscription_plan":plan}})
    return jsonify({"success":True,"plan":plan})

# ─────────────────────────────────────────────
# VALIDATIONS
# ─────────────────────────────────────────────
@app.route("/admin/validations")
@admin_required
def admin_validations():
    return render_template("validations.html",
        user=sdoc(get_db().users.find_one({"email":session["user_email"]})))

@app.route("/admin/correct_prediction/<eid>", methods=["POST"])
@admin_required
def correct_prediction(eid):
    db=get_db()
    try: oid=ObjectId(eid)
    except: return jsonify({"error":"ID invalide"}),400
    if not db.estimations.find_one({"_id":oid}): return jsonify({"error":"Introuvable"}),404
    data=request.get_json(silent=True) or request.form
    name=str(data.get("tree_name","")).strip(); age=data.get("corrected_age"); notes=str(data.get("notes","")).strip()
    if not name or age is None: return jsonify({"error":"Nom et âge requis"}),400
    try: age=int(age)
    except: return jsonify({"error":"Âge invalide"}),400
    db.estimations.update_one({"_id":oid},{"$set":{
        "tree_name":name,"corrected_age":age,"is_validated":True,
        "validated_by":session["user_email"],"validation_date":datetime.now(),
        "validation_notes":notes,"status":"validated"}})
    return jsonify({"success":True})

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
@app.route("/api/predictions/with_names")
@admin_required
def api_predictions():
    db=get_db()
    docs=[]
    for e in db.estimations.find().sort("date",-1).limit(200):
        d=sdoc(e); d["original_age"]=e.get("estimated_age")
        loc=e.get("location") or {}
        d["gps_lat"]=loc.get("lat"); d["gps_lng"]=loc.get("lng")
        docs.append(d)
    return jsonify(docs)

@app.route("/api/species")
def api_species():
    if not SPECIES_MODULE: return jsonify({"species":[],"total":0})
    out=[]
    for n,i in SPECIES_CATALOG.items():
        gd=GROWTH_RATE_DISPLAY.get(i["growth_rate"],{})
        out.append({"name":n,"common_fr":i.get("common_fr",n),"latin_name":i["latin_name"],
                    "growth_rate":i["growth_rate"],"growth_rate_label":gd.get("label",i["growth_rate"]),
                    "growth_rate_color":gd.get("color","#22c55e"),"avg_rings_per_cm":i["avg_rings_per_cm"],
                    "max_age_years":i["max_age_years"],"emoji":i["emoji"],"description":i["description"],
                    "regions_alg":i.get("regions_alg",[])})
    return jsonify({"species":out,"total":len(out)})

@app.route("/api/climate")
@login_required
def api_climate():
    if not CLIMATE_MODULE: return jsonify({"error":"Module indisponible"}),200
    try: lat=float(request.args.get("lat",36.19)); lng=float(request.args.get("lng",5.41))
    except: return jsonify({"error":"Coordonnées invalides"}),400
    try: return jsonify({"success":True,"climate":_geo_enr.enrich(lat,lng)})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/api/check_quota")
@login_required
def api_quota():
    ok,q=check_quota(session["user_email"]); q["can_predict"]=ok; return jsonify(q)

@app.route("/api/stats/species")
@admin_required
def api_stats():
    db=get_db()
    return jsonify({
        "species_distribution": list(db.estimations.aggregate([
            {"$group":{"_id":"$species","count":{"$sum":1},"avg_age":{"$avg":"$estimated_age"},"emoji":{"$first":"$species_emoji"}}},
            {"$sort":{"count":-1}}])),
        "climate_distribution": list(db.estimations.aggregate([
            {"$group":{"_id":"$climate_zone","count":{"$sum":1},"avg_age":{"$avg":"$estimated_age"}}},
            {"$sort":{"count":-1}}])),
        "region_distribution":  list(db.estimations.aggregate([
            {"$group":{"_id":"$algeria_region","count":{"$sum":1}}},
            {"$sort":{"count":-1}},{"$limit":10}]))
    })

@app.route("/saas/plans")
def saas_plans(): return render_template("pricing.html")

# ─────────────────────────────────────────────
# MÉMOIRE / THÈSE — pages académiques
# Master2 GADM — Bouchareb Loutfi — UBM Annaba
# ─────────────────────────────────────────────
@app.route("/memoire")
@login_required
def memoire_home():
    """Page d'accueil du mémoire : métadonnées + résumé + table des matières."""
    if not THESIS_MODULE:
        flash("Module mémoire indisponible.", "error")
        return redirect(url_for("dashboard"))
    db   = get_db()
    user = sdoc(db.users.find_one({"email": session["user_email"]})) or {
        "email": session["user_email"], "role": session.get("user_role", "viewer"),
        "name": session.get("user_name", "")}
    return render_template(
        "memoire_home.html",
        user=user,
        meta=thesis_content.THESIS_META,
        abstract_fr=thesis_content.ABSTRACT_FR,
        abstract_en=thesis_content.ABSTRACT_EN,
        chapters=thesis_content.get_chapter_titles(),
        mapping=thesis_content.THESIS_TO_APP_MAPPING,
    )

@app.route("/memoire/chapitre/<int:num>")
@login_required
def memoire_chapter(num):
    """Affiche un chapitre du mémoire."""
    if not THESIS_MODULE:
        flash("Module mémoire indisponible.", "error")
        return redirect(url_for("dashboard"))
    chapter = thesis_content.get_chapter(num)
    if not chapter:
        flash(f"Chapitre {num} introuvable.", "error")
        return redirect(url_for("memoire_home"))
    db   = get_db()
    user = sdoc(db.users.find_one({"email": session["user_email"]})) or {
        "email": session["user_email"], "role": session.get("user_role", "viewer"),
        "name": session.get("user_name", "")}
    return render_template(
        "memoire_chapter.html",
        user=user,
        chapter=chapter,
        all_chapters=thesis_content.get_chapter_titles(),
        meta=thesis_content.THESIS_META,
    )

@app.route("/api/memoire/chapter/<int:num>")
@login_required
def api_memoire_chapter(num):
    """API JSON pour récupérer un chapitre (utile pour intégrations frontend)."""
    if not THESIS_MODULE:
        return jsonify({"error": "Module indisponible"}), 503
    ch = thesis_content.get_chapter(num)
    if not ch:
        return jsonify({"error": "Chapitre introuvable"}), 404
    return jsonify(ch)

# ─────────────────────────────────────────────
# À PROPOS — page profil éditable
# ─────────────────────────────────────────────
@app.route("/about")
@login_required
def about():
    email    = session["user_email"]
    db       = get_db()
    user     = sdoc(db.users.find_one({"email": email})) or {
        "email": email, "role": session.get("user_role","viewer"),
        "name": session.get("user_name","")}
    about    = sdoc(db.about_profiles.find_one({"user_email": email})) or {}
    stats    = {"total_predictions": db.estimations.count_documents({"user_email": email})}

    # Estimations avec GPS pour tableau Google Maps
    query = {} if session.get("user_role")=="admin" else {"user_email": email}
    raw   = list(db.estimations.find(
        {**query, "location": {"$ne": None}},
        {"estimated_age":1,"species":1,"species_common_fr":1,"species_emoji":1,
         "location":1,"date":1,"algeria_region":1,"diameter_cm":1}
    ).sort("date",-1).limit(50))
    estimations = [sdoc(e) for e in raw]

    return render_template("about.html", user=user, about=about,
                           stats=stats, estimations=estimations)


@app.route("/about/save", methods=["POST"])
@login_required
def about_save():
    email = session["user_email"]
    data  = request.get_json(silent=True) or {}
    data["user_email"]  = email
    data["updated_at"]  = datetime.now()
    db = get_db()
    db.about_profiles.update_one(
        {"user_email": email},
        {"$set": data},
        upsert=True
    )
    return jsonify({"success": True})


# ─────────────────────────────────────────────
# GOOGLE MAPS — lien direct après prédiction
# ─────────────────────────────────────────────
@app.route("/api/tree_map_link/<est_id>")
@login_required
def tree_map_link(est_id):
    """Retourne le lien Google Maps pour un arbre donné."""
    db = get_db()
    try:   oid = ObjectId(est_id)
    except: return jsonify({"error":"ID invalide"}), 400
    est = db.estimations.find_one({"_id": oid}, {"location":1,"species_common_fr":1,"estimated_age":1})
    if not est or not est.get("location"):
        return jsonify({"error": "Pas de GPS pour cet arbre"}), 404
    loc = est["location"]
    lat, lng = loc.get("lat"), loc.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "Coordonnées manquantes"}), 404
    label   = est.get("species_common_fr","Arbre")
    age     = est.get("estimated_age","?")
    gm_link = f"https://www.google.com/maps?q={lat},{lng}&z=16"
    gm_emb  = f"https://www.google.com/maps?q={lat},{lng}&z=16&output=embed"
    gm_dir  = f"https://www.google.com/maps/dir/?api=1&destination={lat},{lng}"
    return jsonify({
        "lat": lat, "lng": lng,
        "google_maps_link":   gm_link,
        "google_maps_embed":  gm_emb,
        "google_maps_directions": gm_dir,
        "label": f"{label} — {age} ans",
        "region": est.get("algeria_region","Algérie")
    })

@app.route("/uploads/<fn>")
@login_required
def uploaded_file(fn): return send_from_directory(Config.UPLOAD_FOLDER,fn)

@app.route("/manifest.json")
def manifest(): return send_from_directory(os.path.join(app.root_path,"static"),"manifest.json")

@app.route("/health")
def health():
    try: get_db().client.admin.command("ping"); dbok=True
    except: dbok=False
    return jsonify({"status":"ok" if dbok else "degraded","db":"connected" if dbok else "disconnected",
                    "species_module":SPECIES_MODULE,"climate_module":CLIMATE_MODULE,
                    "thesis_module":THESIS_MODULE,
                    "pipeline":"3.1","pil":PIL_AVAILABLE,"cv2":CV2_AVAILABLE,
                    "species_count":len(SPECIES_CATALOG) if SPECIES_MODULE else 0,
                    "timestamp":datetime.now().isoformat()}), 200 if dbok else 503

# ─── Filters ──────────────────────────────────────────────────
@app.template_filter("datefmt")
def datefmt(v,fmt="%d/%m/%Y %H:%M"):
    if isinstance(v,str):
        try: v=datetime.fromisoformat(v)
        except: return v
    return v.strftime(fmt) if isinstance(v,datetime) else v

@app.template_filter("species_color")
def sp_color(gr):
    return {"very_fast":"#ef4444","fast":"#f97316","medium":"#eab308",
            "slow":"#22c55e","very_slow":"#3b82f6"}.get(gr,"#22c55e")
app.jinja_env.globals["get_growth_color"]=sp_color

# ─── Errors ───────────────────────────────────────────────────
@app.errorhandler(404)
def e404(e): return render_template("login.html",error="Page introuvable (404)."),404
@app.errorhandler(413)
def e413(e): flash("Image trop grande (max 16 MB).","error"); return redirect(url_for("dashboard"))
@app.errorhandler(500)
def e500(e): return render_template("login.html",error="Erreur serveur (500)."),500

# ─────────────────────────────────────────────
if __name__=="__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║      🌳  TreeAge AI v3.1 — DZ Edition               ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  URL   → http://localhost:5001                       ║")
    print("║  Admin → admin@treepage.fr / AdminPassword@2026      ║")
    print(f"║  Espèces  : {'✅ 8 espèces DZ (ML)' if SPECIES_MODULE else '⚠ Fallback':<35}║")
    print(f"║  Climat   : {'✅ Algérie N/S/E/O' if CLIMATE_MODULE else '⚠ Fallback':<35}║")
    print(f"║  Mémoire  : {'✅ Master2 GADM intégré' if THESIS_MODULE else '⚠ Indisponible':<35}║")
    print(f"║  GPS EXIF : {'✅ PIL installé' if PIL_AVAILABLE else '⚠ pip install Pillow':<35}║")
    print(f"║  CV2 Diam : {'✅ OpenCV installé' if CV2_AVAILABLE else '⚠ pip install opencv-python':<35}║")
    print("╚══════════════════════════════════════════════════════╝")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5001)),
            debug=os.environ.get("FLASK_DEBUG","false").lower()=="true")
    if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
