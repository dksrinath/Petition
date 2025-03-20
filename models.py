from extensions import db
from datetime import datetime
from flask_login import UserMixin 
from sqlalchemy import DateTime
from sqlalchemy.sql import func # Added this import

class User(db.Model, UserMixin):  # Inherit from UserMixin
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    name = db.Column(db.String(120))
    role = db.Column(db.String(20), default="user")  # user, official, admin
    department = db.Column(db.String(50))  # For officials

    def get_id(self):  # Optional: Explicitly define get_id, though UserMixin provides a default
        return str(self.id)

class Department(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.String(200))

class PetitionStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200))

class Petition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_name = db.Column(db.String(200))
    content_text = db.Column(db.Text)
    is_public = db.Column(db.Boolean, default=False)
    priority = db.Column(db.String(20))
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'))
    status_id = db.Column(db.Integer, db.ForeignKey('petition_status.id'))
    upload_time = db.Column(DateTime(timezone=True), nullable=False, default=func.now())
    verified = db.Column(db.Boolean, default=False)
    resolution_notes = db.Column(db.Text)
    resolution_time = db.Column(db.DateTime)
    similar_petitions = db.Column(db.String(500))
    tags = db.Column(db.String(500))
    last_reminder_sent = db.Column(db.DateTime)
    cluster_id = db.Column(db.Integer)
    cluster_distance = db.Column(db.Float)
    department = db.relationship('Department', backref='petitions')

class Verification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    aadhar_no = db.Column(db.String(12), unique=True, nullable=False)
    dob = db.Column(db.String(10), nullable=False)
    location = db.Column(db.String(120), nullable=False)

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    petition_id = db.Column(db.Integer, db.ForeignKey('petition.id'), nullable=False)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    petition_id = db.Column(db.Integer, db.ForeignKey('petition.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    petition_id = db.Column(db.Integer, db.ForeignKey('petition.id'))
    message = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)

class StatusUpdate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    petition_id = db.Column(db.Integer, db.ForeignKey('petition.id'), nullable=False)
    old_status_id = db.Column(db.Integer, db.ForeignKey('petition_status.id'))
    new_status_id = db.Column(db.Integer, db.ForeignKey('petition_status.id'), nullable=False)
    notes = db.Column(db.Text)
    updated_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)