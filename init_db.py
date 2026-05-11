"""init_db.py — Initialise MongoDB pour TreeAge AI v3.1"""
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
from datetime import datetime
import os

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME   = "tree_deep_db"

def init():
    print("="*55)
    print("🗄️  Initialisation MongoDB — TreeAge AI v3.1")
    print("="*55)
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    db = client[DB_NAME]
    print(f"✅ Connecté: {DB_NAME}")

    # Collections
    for col in ["users","estimations","forests","models_ia","about_profiles"]:
        if col not in db.list_collection_names():
            db.create_collection(col)
    
    # Index
    db.users.create_index("email", unique=True)
    db.estimations.create_index("user_email")
    db.estimations.create_index([("date",-1)])
    db.estimations.create_index([("location","2dsphere")], sparse=True)
    db.estimations.create_index("species")
    db.about_profiles.create_index("user_email", unique=True)
    print("✅ Index créés")

    # Admin
    if not db.users.find_one({"email":"admin@treepage.fr"}):
        db.users.insert_one({
            "name":"Admin","email":"admin@treepage.fr",
            "password_hash": generate_password_hash("AdminPassword@2026","pbkdf2:sha256"),
            "role":"admin","is_active":True,"subscription_plan":"enterprise",
            "total_predictions":0,"created_at":datetime.now(),"last_login":None
        })
        print("✅ Admin créé: admin@treepage.fr / AdminPassword@2026")

    # Viewer
    if not db.users.find_one({"email":"lotfi@email.com"}):
        db.users.insert_one({
            "name":"Lotfi","email":"lotfi@email.com",
            "password_hash": generate_password_hash("123","pbkdf2:sha256"),
            "role":"viewer","is_active":True,"subscription_plan":"researcher",
            "total_predictions":0,"created_at":datetime.now(),"last_login":None
        })
        print("✅ Viewer créé: lotfi@email.com / 123")

    print(f"\n📊 Users: {db.users.count_documents({})}")
    print(f"📊 Estimations: {db.estimations.count_documents({})}")
    print("="*55)
    client.close()

if __name__ == "__main__":
    init()
