from app import app, db


from models import Petition



with app.app_context():

    petitions = Petition.query.all()

    for p in petitions:

        print(f"Petition ID: {p.id}, Title: {p.title}, Status ID: {p.status_id}")