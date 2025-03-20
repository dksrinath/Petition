# check_public_petitions.py
from app import app, db
from models import Petition

with app.app_context():
    public_petitions = Petition.query.filter_by(is_public=True, verified=True).all()
    if public_petitions:
        for p in public_petitions:
            print(f"ID: {p.id}, Title: {p.title}, Public: {p.is_public}, Verified: {p.verified}, Dept ID: {p.department_id}")
    else:
        print("No public, verified petitions found.")