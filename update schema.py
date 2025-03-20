from app import app, db

from models import User, Department, PetitionStatus, Petition, Like, Comment, Notification, StatusUpdate



def clear_data_except_verification():

    with app.app_context():

        

        db.session.query(StatusUpdate).delete()

        db.session.query(Notification).delete()

        db.session.query(Comment).delete()

        db.session.query(Like).delete()

        db.session.query(Petition).delete()

        db.session.query(PetitionStatus).delete()

        db.session.query(Department).delete()

        db.session.query(User).delete()

        

        

        db.session.commit()

        print("Data cleared from all tables except 'verification'.")



if __name__ == "__main__":

    clear_data_except_verification()