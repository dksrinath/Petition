from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, abort
from extensions import db, bcrypt, login_manager, mail, init_extensions
from flask_login import login_user, login_required, logout_user, current_user
from flask_mail import Message
from datetime import datetime, timedelta, timezone
from utils import handle_errors, get_current_utc, PetitionError, AuthenticationError
import logging
from logging.handlers import RotatingFileHandler
import os
import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from transformers import pipeline
import re
import unittest
from markupsafe import Markup  # Instead of from flask import Markup

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail configuration (disabled for development)
app.config['MAIL_SUPPRESS_SEND'] = True
app.config['MAIL_DEFAULT_SENDER'] = ('Petition System', 'noreply@petitionsystem.com')
app.config['MAIL_SERVER'] = 'localhost'
app.config['MAIL_PORT'] = 25

# Initialize extensions
init_extensions(app)

# Setup logging
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

file_handler = RotatingFileHandler('logs/petition.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Petition system startup')

# Import models after extensions
from models import User, Department, PetitionStatus, Petition, Verification, Like, Comment, Notification, StatusUpdate

# Load NLP and ML models
try:
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
except:
    nlp = spacy.load("en_core_web_sm", disable=["ner"]) if "en_core_web_sm" in spacy.util.get_installed_models() else None
    app.logger.warning("Using limited NLP capabilities due to missing spaCy model.")

try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
except:
    sentiment_analyzer = None
    app.logger.warning("Sentiment analysis disabled due to missing model.")

vectorizer = TfidfVectorizer()
classifier = MultinomialNB()

# AI Functions
def extract_text_from_file(file_path):
    if not os.path.exists(file_path):
        return None
    file_ext = file_path.rsplit(".", 1)[-1].lower()
    try:
        if file_ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_ext == "pdf":
            doc = fitz.open(file_path)
            return "\n".join([page.get_text() for page in doc])
        return None
    except Exception as e:
        app.logger.error(f"Error reading file {file_path}: {e}")
        return None

def find_similar_petitions(new_petition_content, limit=5):
    """Find petitions similar to the provided content using cosine similarity."""
    if not new_petition_content:
        return []
    
    existing_petitions = Petition.query.filter(Petition.content_text.isnot(None)).all()
    if not existing_petitions:
        return []

    petition_texts = [p.content_text for p in existing_petitions]
    petition_ids = [p.id for p in existing_petitions]
    
    # Combine existing texts with the new petition content
    all_texts = petition_texts + [new_petition_content]
    
    # Compute TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Get the indices of the most similar petitions
    similar_indices = similarities.argsort()[::-1][:limit]
    
    # Return a list of similar petitions with their similarity scores
    return [{"id": petition_ids[idx], "similarity": similarities[idx]} for idx in similar_indices if similarities[idx] > 0.2]

def schedule_clustering():
    """Schedule clustering of petitions."""
    from clustering import cluster_petitions  # Import your clustering function
    clusters = cluster_petitions()  # Call the clustering function
    app.logger.info(f"Clustered petitions into {len(clusters)} groups")


def analyze_petition_content(content):
    if not content or not content.strip():
        return {"priority": "Normal", "department": "General", "tags": []}

    # Define department keywords
    department_keywords = {
    "Infrastructure": {
        "primary": ["road", "traffic", "bridge", "pavement", "infrastructure", "construction"],
        "secondary": ["safety", "public transport", "maintenance", "repair", "accessibility"]
    },
    "Public Safety": {
        "primary": ["police", "crime", "fire", "safety", "emergency", "security", "disaster"],
        "secondary": ["protection", "community watch", "crime prevention", "response"]
    },
    "Environment": {
        "primary": ["pollution", "waste", "climate", "environment", "conservation", "recycling"],
        "secondary": ["sustainable", "clean energy", "wildlife", "nature", "habitat", "ecosystem"]
    },
    "Education": {
        "primary": ["school", "education", "student", "teacher", "curriculum", "tuition", "classroom"],
        "secondary": ["academic", "training", "resources", "funding", "scholarship", "after-school"]
    },
    "Health": {
        "primary": ["health", "hospital", "doctor", "healthcare", "medical", "treatment"],
        "secondary": ["care", "disease", "mental health", "insurance", "wellness", "support"]
    },
    "Housing": {
        "primary": ["housing", "home", "rent", "affordable housing", "shelter", "eviction"],
        "secondary": ["tenant", "landlord", "subsidy", "housing assistance", "homelessness"]
    },
    "Social Welfare": {
        "primary": ["welfare", "benefit", "aid", "support", "community service", "assistance"],
        "secondary": ["food security", "employment", "training programs", "social services"]
    },
    "Transportation": {
        "primary": ["public transport", "bus", "train", "subway", "commuting", "traffic"],
        "secondary": ["bike lanes", "carpool", "ride-sharing", "transportation policy", "access"]
    },
    "Employment": {
        "primary": ["job", "employment", "work", "salary", "benefits", "unemployment"],
        "secondary": ["training", "workplace safety", "labor rights", "job security"]
    },
    "Community Development": {
        "primary": ["community", "development", "revitalization", "programs", "resources"],
        "secondary": ["neighborhood", "engagement", "outreach", "partnerships", "involvement"]
    }
}


    text_lower = content.lower()
    department_scores = {dept: 0 for dept in department_keywords}
    
    # Calculate department scores based on keywords
    for dept, keywords in department_keywords.items():
        for keyword in keywords["primary"]:
            department_scores[dept] += text_lower.count(keyword) * 2
        for keyword in keywords["secondary"]:
            department_scores[dept] += text_lower.count(keyword)
    
    max_score = max(department_scores.values())
    department = max(department_scores.items(), key=lambda x: x[1])[0] if max_score > 1 else "General"

    priority = "Normal"
    tags = []
    if nlp:
        doc = nlp(text_lower)
        tags = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'GPE', 'LOC']] + \
               [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2][:10]
        
        priority_keywords = {
            "Urgent": ["urgent", "emergency"], "High": ["important", "priority"],
            "Normal": ["request", "general"], "Low": ["minor", "small"]
        }
        priority_scores = {p: sum(text_lower.count(kw) for kw in kws) for p, kws in priority_keywords.items()}
        priority = max(priority_scores.items(), key=lambda x: x[1])[0] if max(priority_scores.values()) > 0 else "Normal"

    if sentiment_analyzer:
        try:
            sentiment = sentiment_analyzer(content[:512])[0]
            if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
                priority = "High" if priority == "Normal" else "Urgent" if priority == "High" else priority
        except Exception as e:
            app.logger.error(f"Sentiment analysis error: {e}")

    return {"priority": priority, "department": department, "tags": tags}


@app.context_processor
def utility_processor():
    def convert_to_ist(utc_time):
        return utc_time + timedelta(hours=5, minutes=30) if utc_time else None
    return dict(convert_to_ist=convert_to_ist)

# Routes
@app.route('/')
def home():
    # Automatically log out the user when they access the home page
    if current_user.is_authenticated:
        flash("You have been logged out.", "info")
        logout_user()
    return redirect(url_for('login'))  # Redirect to the login page

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name', '')
        
        # Simple validation
        if not email or not password:
            flash("Email and password are required", "danger")
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return render_template('register.html')
        
        # Create new user
        try:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(email=email, password=hashed_password, name=name)
            db.session.add(user)
            db.session.commit()
            
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            flash("Registration failed. Please try again.", "danger")
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Simple validation for empty fields
        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template('login.html', email=email)  # Retain the email input

        # Attempt to find the user by email
        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Invalid email or password.", "danger")
            return render_template('login.html', email=email)  # Retain the email input

        # Check if the password is correct
        if not bcrypt.check_password_hash(user.password, password):
            flash("Invalid email or password.", "danger")
            return render_template('login.html', email=email)  # Retain the email input

        # Log the user in
        login_user(user)
        app.logger.info(f"User {email} logged in successfully.")
        return redirect(url_for('dashboard'))  # Redirect to the dashboard or home page

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

@app.route("/dashboard")
@login_required
def dashboard():
    pending_count = Petition.query.filter_by(status_id=1).count()
    in_progress_count = Petition.query.filter_by(status_id=2).count()
    resolved_count = Petition.query.filter_by(status_id=5).count()
    petition_statuses = {s.id: s.name for s in PetitionStatus.query.all()}
    if current_user.role == 'official':
        assigned_petitions = Petition.query.filter(Petition.department_id == Department.query.filter_by(name=current_user.department).first().id, Petition.status_id != 5).order_by(Petition.upload_time.desc()).all()
    elif current_user.role == 'admin':
        assigned_petitions = Petition.query.order_by(Petition.upload_time.desc()).all()
    else:
        assigned_petitions = Petition.query.filter_by(user_id=current_user.id).order_by(Petition.upload_time.desc()).all()
    return render_template('dashboard.html', pending_count=pending_count, in_progress_count=in_progress_count,
                           resolved_count=resolved_count, assigned_petitions=assigned_petitions, petition_statuses=petition_statuses)

@app.route('/verify', methods=['GET', 'POST'])
@login_required
def verify():
    if request.method == 'POST':
        verification = Verification.query.filter_by(
            name=request.form['name'],
            aadhar_no=request.form['aadhar_no'],
            dob=request.form['dob'],
            location=request.form['location']
        ).first()
        if verification and (petition_data := session.get("petition_data")):
            department = Department.query.filter_by(name=petition_data["department"]).first() or \
                         Department(name=petition_data["department"], description=f"Department of {petition_data['department']}")
            if not department.id:
                db.session.add(department)
                db.session.commit()
            initial_status = db.session.get(PetitionStatus, 1)  # Use db.session.get() for consistency
            new_petition = Petition(
                title=petition_data["title"],
                file_name=petition_data["file_name"],
                content_text=petition_data["content_text"],
                is_public=petition_data["is_public"],
                priority=petition_data["priority"],
                department_id=department.id,
                status_id=initial_status.id,
                upload_time=get_current_utc(),
                user_id=current_user.id,
                verified=True,
                tags=','.join(petition_data["tags"])
            )
            db.session.add(new_petition)
            db.session.commit()
            app.logger.info(f"Petition saved: ID={new_petition.id}, is_public={new_petition.is_public}, verified={new_petition.verified}")
            similar_petitions = find_similar_petitions(new_petition.content_text)
            if similar_petitions:
                new_petition.similar_petitions = ','.join([str(p['id']) for p in similar_petitions])
                db.session.commit()
            for official in User.query.filter_by(department=department.name, role='official').all():
                db.session.add(Notification(
                    user_id=official.id,
                    petition_id=new_petition.id,
                    message=f"New {new_petition.priority} petition: {new_petition.title}"
                ))
            db.session.commit()
            session.pop("petition_data", None)
            flash("Petition Verified & Submitted Successfully!", "success")
            return redirect(url_for('my_petitions'))
        flash("Verification Failed! Check Your Details.", "danger")
    return render_template('verify.html')

@app.route("/upload_petition", methods=["GET", "POST"])
@login_required
def upload_petition():
    if request.method == "POST":
        title = request.form.get("title")
        is_public = request.form.get("is_public") == "on"
        content_type = request.form.get("content_type")
        
        if not title:
            flash("Title is required!", "danger")
            return redirect(url_for("upload_petition"))
        
        content_text = None
        file_name = None
        
        if content_type == "file":
            file = request.files.get("file")
            if not file:
                flash("Please upload a file or choose to write content directly.", "danger")
                return redirect(url_for("upload_petition"))
                
            file_ext = file.filename.rsplit(".", 1)[-1].lower()
            if file_ext not in ['txt', 'pdf']:
                flash("Only TXT and PDF files are allowed!", "danger")
                return redirect(url_for("upload_petition"))
                
            unique_filename = f"{get_current_utc().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(file_path)
            
            content_text = extract_text_from_file(file_path)
            file_name = unique_filename
            
            if not content_text:
                flash("Uploaded file is empty or unreadable!", "danger")
                return redirect(url_for("upload_petition"))
        else:  # content_type == "text"
            content_text = request.form.get("content_text")
            if not content_text:
                flash("Petition content is required!", "danger")
                return redirect(url_for("upload_petition"))
            
            # No file was uploaded, but we'll create a text file for consistency
            unique_filename = f"{get_current_utc().strftime('%Y%m%d%H%M%S')}_direct_input.txt"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_text)
            file_name = unique_filename
        
        analysis_results = analyze_petition_content(content_text)
        
        session["petition_data"] = {
            "title": title, 
            "file_name": file_name, 
            "content_text": content_text, 
            "is_public": is_public,
            "priority": analysis_results["priority"], 
            "department": analysis_results["department"], 
            "tags": analysis_results["tags"]
        }
        
        flash("Petition content processed successfully! Proceed to verification.", "info")
        return redirect(url_for("verify"))
        
    return render_template("upload_petition.html")

@app.route("/petitions")
def view_petitions():
    petitions = Petition.query.filter_by(is_public=True, verified=True).order_by(Petition.upload_time.desc()).all()
    app.logger.info(f"Found {len(petitions)} public, verified petitions.")
    petitions_by_department = {dept.name: [p for p in petitions if p.department_id == dept.id] for dept in Department.query.all() if any(p.department_id == dept.id for p in petitions)}
    no_dept_petitions = [p for p in petitions if not p.department_id]
    if no_dept_petitions:
        petitions_by_department["Other"] = no_dept_petitions
    return render_template('petitions.html', petitions_by_department=petitions_by_department)

@app.route("/my_petitions")
@login_required
def my_petitions():
    petitions = Petition.query.filter_by(user_id=current_user.id).order_by(Petition.upload_time.desc()).all()
    status_dict = {s.id: s.name for s in PetitionStatus.query.all()}
    return render_template('my_petitions.html', petitions=petitions, statuses=status_dict)

@app.template_filter('nl2br')
def nl2br_filter(text):
    """Convert newlines to HTML line breaks."""
    if not text:
        return ""
    return Markup(text.replace('\n', '<br>'))


@app.route("/petition/<int:petition_id>")
@login_required
def view_petition(petition_id):
    try:
        # Get the petition with proper error handling
        petition = db.session.get(Petition, petition_id)
        if not petition:
            flash("Petition not found.", "danger")
            return redirect(url_for("dashboard"))  # Redirect to dashboard instead of home
        
        # Check permissions
        if not petition.is_public and petition.user_id != current_user.id and current_user.role not in ['official', 'admin']:
            flash("You don't have permission to view this petition.", "danger")
            return redirect(url_for("dashboard"))  # Redirect to dashboard instead of home
        
        # Get related objects
        department = db.session.get(Department, petition.department_id) if petition.department_id else None
        status = db.session.get(PetitionStatus, petition.status_id) if petition.status_id else None
        
        # Get similar petitions
        similar_petitions = []
        if petition.similar_petitions:
            try:
                similar_ids = [int(id) for id in petition.similar_petitions.split(',')]
                similar_petitions = Petition.query.filter(Petition.id.in_(similar_ids)).all()
            except Exception as e:
                app.logger.error(f"Error processing similar petitions: {str(e)}")
        
        # Get comments, status updates, etc.
        comments = Comment.query.filter_by(petition_id=petition.id).order_by(Comment.timestamp.desc()).all()
        status_updates = StatusUpdate.query.filter_by(petition_id=petition.id).order_by(StatusUpdate.timestamp.desc()).all()
        departments = Department.query.all() if current_user.role == 'admin' else []
        petition_statuses = {s.id: s.name for s in PetitionStatus.query.all()}
        likes_count = Like.query.filter_by(petition_id=petition.id).count()
        user_liked = bool(Like.query.filter_by(user_id=current_user.id, petition_id=petition.id).first())
        
        # Render the template with all necessary data
        return render_template('petition_detail.html', 
                              petition=petition, 
                              department=department, 
                              status=status,
                              similar_petitions=similar_petitions, 
                              comments=comments, 
                              status_updates=status_updates,
                              departments=departments, 
                              petition_statuses=petition_statuses, 
                              likes_count=likes_count, 
                              user_liked=user_liked)
    
    except Exception as e:
        app.logger.error(f"Error viewing petition {petition_id}: {str(e)}")
        flash("An error occurred while trying to view this petition.", "danger")
        return redirect(url_for("dashboard"))  # Redirect to dashboard instead of home


@app.route("/like_petition/<int:petition_id>", methods=["POST"])
@login_required
def like_petition(petition_id):
    petition = Petition.query.get_or_404(petition_id)
    existing_like = Like.query.filter_by(user_id=current_user.id, petition_id=petition_id).first()
    if existing_like:
        db.session.delete(existing_like)
        flash("Like removed", "info")
    else:
        db.session.add(Like(user_id=current_user.id, petition_id=petition_id))
        flash("Petition liked", "success")
    db.session.commit()
    return redirect(url_for("view_petition", petition_id=petition_id))

@app.route("/add_comment/<int:petition_id>", methods=["POST"])
@login_required
def add_comment(petition_id):
    petition = Petition.query.get_or_404(petition_id)
    comment_text = request.form.get("comment_text")
    if not comment_text:
        flash("Comment cannot be empty.", "danger")
    else:
        new_comment = Comment(petition_id=petition_id, user_id=current_user.id, text=comment_text)
        db.session.add(new_comment)
        if petition.user_id != current_user.id:
            db.session.add(Notification(user_id=petition.user_id, petition_id=petition_id,
                                        message=f"New comment on your petition: {petition.title}"))
        db.session.commit()
        flash("Comment added successfully.", "success")
    return redirect(url_for("view_petition", petition_id=petition_id))

@app.route("/update_status/<int:petition_id>", methods=["POST"])
@login_required
def update_status(petition_id):
    try:
        # Check permissions
        if current_user.role not in ['official', 'admin']:
            flash("Permission denied.", "danger")
            return redirect(url_for("view_petition", petition_id=petition_id))
        
        # Get the petition
        petition = db.session.get(Petition, petition_id)
        if not petition:
            flash("Petition not found.", "danger")
            return redirect(url_for("home"))
        
        # Check department permission for officials
        if current_user.role == 'official':
            petition_department = db.session.get(Department, petition.department_id)
            if petition_department and current_user.department != petition_department.name:
                flash("You can only update your department's petitions.", "danger")
                return redirect(url_for("view_petition", petition_id=petition_id))
        
        # Get form data
        new_status_id = int(request.form.get("status_id"))
        notes = request.form.get("notes", "")
        old_status_id = petition.status_id
        
        # Update petition status
        petition.status_id = new_status_id
        
        # Handle resolution
        if new_status_id == 5:  # Resolved
            petition.resolution_time = datetime.now(timezone.utc)
            petition.resolution_notes = notes
        
        # Add status update
        status_update = StatusUpdate(
            petition_id=petition_id,
            old_status_id=old_status_id,
            new_status_id=new_status_id,
            notes=notes,
            updated_by=current_user.id
        )
        db.session.add(status_update)
        
        # Add notification
        new_status = db.session.get(PetitionStatus, new_status_id)
        notification = Notification(
            user_id=petition.user_id,
            petition_id=petition_id,
            message=f"Your petition '{petition.title}' is now {new_status.name if new_status else 'updated'}"
        )
        db.session.add(notification)
        
        # Commit changes
        db.session.commit()
        
        flash("Status updated successfully.", "success")
        return redirect(url_for("view_petition", petition_id=petition_id))
    
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error updating petition status: {str(e)}")
        flash("An error occurred while updating the petition status.", "danger")
        return redirect(url_for("view_petition", petition_id=petition_id))

@app.route("/notifications")
@login_required
def view_notifications():
    notifications = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.timestamp.desc()).all()
    for n in notifications:
        n.is_read = True
    db.session.commit()
    return render_template("notifications.html", notifications=notifications)

@app.route("/assign_department/<int:petition_id>", methods=["POST"])
@login_required
def assign_department(petition_id):
    if current_user.role != 'admin':
        flash("Only admins can reassign departments.", "danger")
        return redirect(url_for("view_petition", petition_id=petition_id))
    petition = Petition.query.get_or_404(petition_id)
    department_id = int(request.form.get("department_id"))
    department = Department.query.get(department_id)
    if department:
        petition.department_id = department_id
        for official in User.query.filter_by(department=department.name, role='official').all():
            db.session.add(Notification(user_id=official.id, petition_id=petition_id,
                                        message=f"New {petition.priority} petition: {petition.title}"))
        db.session.commit()
        flash(f"Petition reassigned to {department.name}.", "success")
    else:
        flash("Invalid department.", "danger")
    return redirect(url_for("view_petition", petition_id=petition_id))

@app.route("/admin/departments", methods=["GET", "POST"])
@login_required
def manage_departments():
    if current_user.role != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for("home"))
    if request.method == "POST":
        name = request.form.get("name")
        if name and not Department.query.filter_by(name=name).first():
            db.session.add(Department(name=name, description=request.form.get("description", "")))
            db.session.commit()
            flash(f"Department '{name}' added.", "success")
        else:
            flash("Department name required or already exists.", "danger")
    departments = Department.query.order_by(Department.name).all()
    return render_template("manage_departments.html", departments=departments)

@app.route("/admin/officials", methods=["GET", "POST"])
@login_required
def manage_officials():
    if current_user.role != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form.get("email")
        if email and not User.query.filter_by(email=email).first():
            db.session.add(User(email=email, name=request.form.get("name", ""),
                                password=bcrypt.generate_password_hash(request.form.get("password")).decode('utf-8'),
                                role="official", department=request.form.get("department")))
            db.session.commit()
            flash(f"Official '{email}' added.", "success")
        else:
            flash("Email required or already exists.", "danger")
    officials = User.query.filter_by(role="official").all()
    departments = Department.query.all()
    return render_template("manage_officials.html", officials=officials, departments=departments)

@app.route("/stats")
@login_required
def petition_statistics():
    # Get department statistics
    dept_stats = db.session.query(
        Department.name,
        db.func.count(Petition.id).label('count')
    ).outerjoin(Petition).group_by(Department.name).all()

    # Get priority statistics
    priority_stats = db.session.query(
        Petition.priority,
        db.func.count(Petition.id).label('count')
    ).group_by(Petition.priority).all()

    # Get status statistics
    status_stats = db.session.query(
        PetitionStatus.name,
        db.func.count(Petition.id).label('count')
    ).outerjoin(Petition).group_by(PetitionStatus.name).all()

    # Calculate average resolution time
    resolved_petitions = Petition.query.filter(
        Petition.resolution_time.isnot(None)
    ).all()
    
    avg_resolution_time = None
    if resolved_petitions:
        total_time = sum(
            (p.resolution_time - p.upload_time).total_seconds() 
            for p in resolved_petitions
        )
        avg_resolution_time = total_time / len(resolved_petitions)

    # Get clusters
    clusters = {}
    for p in Petition.query.filter(Petition.cluster_id.isnot(None)).order_by(Petition.cluster_distance).all():
        cluster_id = p.cluster_id  # Assuming cluster_id is already an integer
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(p)

    return render_template(
        "statistics.html",
        dept_stats=dept_stats,
        priority_stats=priority_stats,
        status_stats=status_stats,
        avg_resolution_time=avg_resolution_time,
        clusters=clusters
    )

def initialize_app_data():
    with app.app_context():
        db.create_all()
        statuses = [
            (1, "Pending", "Petition received"),
            (2, "In Progress", "Being processed"),
            (3, "Under Review", "Additional review needed"),
            (4, "Awaiting Response", "Waiting for petitioner"),
            (5, "Resolved", "Petition resolved"),
            (6, "Rejected", "Petition rejected")
        ]
        for id, name, desc in statuses:
            if not PetitionStatus.query.get(id):
                db.session.add(PetitionStatus(id=id, name=name, description=desc))
        departments = ["General", "Health", "Education", "Infrastructure", "Environment", "Public Safety", "Housing", "Social Welfare"]
        for name in departments:
            if not Department.query.filter_by(name=name).first():
                db.session.add(Department(name=name, description=f"{name} Department"))
        if not User.query.filter_by(email="admin@petition-system.com").first():
            db.session.add(User(email="admin@petition-system.com", password=bcrypt.generate_password_hash("admin123").decode('utf-8'),
                                name="Admin", role="admin"))
        db.session.commit()

def send_reminders():
    now = get_current_utc()
    reminder_intervals = {"Urgent": 1, "High": 2, "Normal": 7, "Low": 14}
    petitions = Petition.query.filter(Petition.status_id != 5).all()
    for petition in petitions:
        interval = reminder_intervals.get(petition.priority, 7)
        if not petition.last_reminder_sent or (now - petition.last_reminder_sent).days >= interval:
            department = Department.query.get(petition.department_id)
            if department:
                for official in User.query.filter_by(department=department.name, role='official').all():
                    db.session.add(Notification(user_id=official.id, petition_id=petition.id,
                                                message=f"REMINDER ({petition.priority}): Petition '{petition.title}'"))
            petition.last_reminder_sent = now
    db.session.commit()
    app.logger.info("Reminders sent.")

if __name__ == '__main__':
    # Initialize the database and start the app
    initialize_app_data()
    scheduler = BackgroundScheduler()
    scheduler.add_job(send_reminders, 'interval', hours=24)
    scheduler.add_job(schedule_clustering, 'interval', hours=1)
    scheduler.start()
    
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
