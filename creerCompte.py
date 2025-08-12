from flask import Flask, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_session import Session
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt

from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask_login import (
    LoginManager,
    login_user,
    current_user,
    logout_user,
    login_required,
    UserMixin,
)
import os
import pymysql
from werkzeug.utils import secure_filename
import pandas as pd
import logging
import shutil
from datetime import datetime
from flask import request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from collections import OrderedDict  # Assurez-vous que cet import est présent
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDRegressor, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import datetime
from datetime import datetime, date
import numpy as np
from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
)
from flask import send_from_directory
from datetime import timedelta, datetime
from datetime import timezone
from sqlalchemy import func
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

import json
import gzip
import traceback
import re  # Add this import for regular expressions


logging.basicConfig(level=logging.DEBUG)
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql://root:@localhost/prj"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "supersecretkey"
app.config["SESSION_TYPE"] = "filesystem"
app.config["UPLOAD_FOLDER"] = "uploads/"  # Dossier de téléchargement des fichiers
app.config["MAX_CONTENT_LENGTH"] = 900 * 1024 * 1024  # 50 MB
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 30 * 60
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = 7 * 24 * 60 * 60  # 7 jours
mod_FOLDER = "mod/"


from flask_mail import Mail, Message

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = "ibtihelomri12@gmail.com"
app.config["MAIL_PASSWORD"] = "aflj rmlb iboh opdp"

mail = Mail(app)


db = SQLAlchemy(app)
CORS(app, supports_credentials=True)
Session(app)
jwt = JWTManager(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
refresh_tokens = {}


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(500), nullable=False)
    phone_number = db.Column(db.String(20))
    speciality = db.Column(db.String(100))
    hospital = db.Column(db.String(100))
    profile_image = db.Column(db.String(200))  # chemin vers le fichier
    remember_me = db.Column(db.Boolean, default=False)
    is_approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trial_count = db.Column(db.Integer, default=0)
    is_admin = db.Column(db.Boolean, default=False)
    projects = db.relationship(
        "Project", backref="user", cascade="all, delete-orphan", lazy=True
    )
    logins = db.relationship(
        "LoginLog",
        backref="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy=True
    )



class LoginLog(db.Model):
    __tablename__ = "login_log"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False
    )
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    imported_files = db.relationship(
        "ImportedFile", backref="project", cascade="all, delete-orphan", lazy=True
    )
    targetfeature = db.relationship(
        "TargetFeature", backref="project", cascade="all, delete-orphan", lazy=True
    )
    modifiedfile = db.relationship(
        "ModifiedFile", backref="project", cascade="all, delete-orphan", lazy=True
    )
    models = db.relationship(
        "Models", backref="project", cascade="all, delete-orphan", lazy=True
    )


class ImportedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False)
    filepath = db.Column(
        db.String(300), nullable=False
    )  # Nouveau champ pour le chemin complet du fichier
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    modified_files = db.relationship(
        "ModifiedFile",
        backref="imported_file",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys="[ModifiedFile.parent_id]"
    )


class TargetFeature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(300), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)


class ModifiedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(300), nullable=False)
    filepath = db.Column(
        db.String(300), nullable=False
    )  # Nouveau champ pour le chemin complet du fichier
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    target_feature = db.Column(db.String(300))  # Nouveau champ

    modification = db.Column(db.String(255))  #  Nouveau

    parent_id = db.Column(db.Integer, db.ForeignKey("imported_file.id"))
    parent = db.relationship("ImportedFile", foreign_keys=[parent_id])
    preprocessing_steps = db.Column(db.JSON)
    is_for_prediction = db.Column(db.Boolean, default=False)



class Models(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    modelname = db.Column(db.String(300), nullable=False)
    modelpath = db.Column(db.String(300), nullable=False)
    validpath = db.Column(db.String(300), nullable=False)
    target_feature = db.Column(db.String(255))
    Accuracy = db.Column(db.Float, nullable=False)
    Precisionn = db.Column(db.Float, nullable=False)
    Recall = db.Column(db.Float, nullable=False)
    F1_Score = db.Column(db.Float, nullable=False)
    ROC_AUC = db.Column(db.Float, nullable=False)
    MeanAbsoluteError = db.Column(db.Float, nullable=False)
    MeanSquaredError = db.Column(db.Float, nullable=False)
    RScore = db.Column(db.Float, nullable=False)
    featureimportance = db.Column(db.String(10000), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey("project.id"), nullable=False)
    trainingset = db.Column(db.Integer, nullable=True)
    testset = db.Column(db.Integer, nullable=True)
    k = db.Column(db.Integer, nullable=True)
    confusion_matrix = db.Column(db.Text)  # Stockage JSON stringifié
    confusion_labels = db.Column(db.Text)  # Stockage JSON stringifié
    file_id = db.Column(db.Integer, db.ForeignKey("modified_file.id"), nullable=True)
    file = db.relationship("ModifiedFile", backref="models")


@app.route("/admin/pending-users", methods=["GET"])
@jwt_required()
def get_pending_users():
    current_user_id = get_jwt_identity()
    current_user = User.query.get(current_user_id)

    # Vérifie si l'utilisateur est un administrateur
    if current_user.email != "admin@admin.com":  # ou ajouter un champ `is_admin`
        return jsonify({"message": "Unauthorized"}), 403

    pending_users = User.query.filter_by(is_approved=False).all()
    user_list = [
        {
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "speciality": u.speciality,
            "created_at": u.created_at.isoformat(),
        }
        for u in pending_users
    ]

    return jsonify(user_list), 200


def sanitize_string(s):
    """Remove newlines and control characters from a string."""
    if s:
        return re.sub(r"[\n\r\t]", "", s).strip()
    return s


def is_valid_email(email):
    """Validate email format."""
    return email and re.match(r"[^@]+@[^@]+\.[^@]+", email)


@app.route("/admin/user/<int:user_id>/approve", methods=["POST"])
@jwt_required()
def approve_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        # Sanitize and validate email
        user_email = sanitize_string(user.email)
        if not is_valid_email(user_email):
            return jsonify({"message": f"Invalid email address: {user_email}"}), 400

        # Update user approval status
        user.is_approved = True
        db.session.commit()

        # Prepare email
        subject = sanitize_string("Votre compte a été approuvé")
        msg = Message(
            subject=subject,
            sender=app.config.get("MAIL_DEFAULT_SENDER", "no-reply@medical-vision.com"),
            recipients=[user_email],
        )
        msg.html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 30px;">
            <div style="max-width: 600px; margin: auto; background-color: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h2 style="color: #1a7bd3;">Bienvenue chez Medical Vision !</h2>
            <p>Bonjour {sanitize_string(user.name)},</p>
            <p>Nous avons le plaisir de vous annoncer que votre demande d’inscription à <strong>Medical Vision</strong> a été <span style="color: green;"><strong>acceptée</strong></span>.</p>
            <p>Vous pouvez maintenant vous connecter à votre compte et découvrir nos outils d’aide à la décision médicale.</p>
            <div style="text-align: center;">
                <a href="http://medical-vision.com/login" style="margin-top: 20px; display: inline-block; padding: 12px 24px; background-color: #1a7bd3; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">Se connecter</a>
            </div>
            <p style="margin-top: 40px; font-size: 13px; color: #555;">Besoin d’aide ? Écrivez-nous à <a href="mailto:contact@medical-vision.com">contact@medical-vision.com</a></p>
            <hr style="margin-top: 30px; border: none; border-top: 1px solid #eee;">
            <p style="text-align: center; font-size: 12px; color: #aaa;">© {datetime.now().year} Medical Vision – Tous droits réservés.</p>
            </div>
        </body>
        </html>"""

        # Log email details for debugging
        print(
            f"Sending email - Subject: {msg.subject}, Sender: {msg.sender}, Recipients: {msg.recipients}"
        )
        mail.send(msg)

        return jsonify({"message": "User approved and email sent"}), 200
    except Exception as e:
        db.session.rollback()  # Roll back database changes on error
        print(f"Error approving user {user_id}: {str(e)}")
        return jsonify({"message": f"Failed to approve user: {str(e)}"}), 500


@app.route("/admin/user/<int:user_id>/reject", methods=["DELETE"])
@jwt_required()
def reject_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        # Sanitize and validate email
        user_email = sanitize_string(user.email)
        if not is_valid_email(user_email):
            return jsonify({"message": f"Invalid email address: {user_email}"}), 400

        # Prepare email
        subject = sanitize_string("Votre compte a été refusé")
        msg = Message(
            subject=subject,
            sender=app.config.get("MAIL_DEFAULT_SENDER", "no-reply@medical-vision.com"),
            recipients=[user_email],
        )
        msg.html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 30px;">
            <div style="max-width: 600px; margin: auto; background-color: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <h2 style="color: #e74c3c;">Demande refusée</h2>
            <p>Bonjour {sanitize_string(user.name)},</p>
            <p>Merci pour votre intérêt pour <strong>Medical Vision</strong>.</p>
            <p>Après analyse de votre profil, nous vous informons que votre demande d'inscription a été <span style="color: red;"><strong>refusée</strong></span>.</p>
            <p>Si vous avez des questions ou souhaitez contester cette décision, vous pouvez nous contacter.</p>
            <p style="margin-top: 20px; font-size: 13px; color: #555;">Écrivez-nous à <a href="mailto:contact@medical-vision.com">contact@medical-vision.com</a></p>
            <hr style="margin-top: 30px; border: none; border-top: 1px solid #eee;">
            <p style="text-align: center; font-size: 12px; color: #aaa;">© {datetime.now().year} Medical Vision – Tous droits réservés.</p>
            </div>
        </body>
        </html>"""

        print(
            f"Sending email - Subject: {msg.subject}, Sender: {msg.sender}, Recipients: {msg.recipients}"
        )
        mail.send(msg)
        
        # Supprimer les logs (si la cascade n'est pas active côté SQL)
        LoginLog.query.filter_by(user_id=user.id).delete()

        # Delete user
        db.session.delete(user)
        db.session.commit()

        return jsonify({"message": "User rejected and email sent"}), 200
    except Exception as e:
        db.session.rollback()
        print(f"Error rejecting user {user_id}: {str(e)}")
        return jsonify({"message": f"Failed to reject user: {str(e)}"}), 500


@app.route("/admin/stats", methods=["GET"])
@jwt_required()
def admin_stats():
    now = datetime.utcnow()

    # 1. Inscriptions par semaine (6 dernières semaines)
    six_weeks_ago = now - timedelta(weeks=6)
    weekly_counts = (
        db.session.query(
            func.date_format(User.created_at, "%x-%v").label(
                "week"
            ),  # Année-semaine ISO
            func.count(User.id),
        )
        .filter(User.created_at >= six_weeks_ago)
        .group_by("week")
        .order_by("week")
        .all()
    )

    signups_labels = [week for week, _ in weekly_counts]
    signups_values = [count for _, count in weekly_counts]

    # 2. Répartition des spécialités (camembert)
    speciality_counts = (
        db.session.query(User.speciality, func.count(User.id))
        .group_by(User.speciality)
        .all()
    )

    speciality_labels = [spec if spec else "Inconnue" for spec, _ in speciality_counts]
    speciality_values = [count for _, count in speciality_counts]

    # 3. Approuvés vs en attente
    approved = (
        db.session.query(func.count(User.id)).filter(User.is_approved == True).scalar()
        or 0
    )
    pending = (
        db.session.query(func.count(User.id)).filter(User.is_approved == False).scalar()
        or 0
    )

    # 4. Connexions par jour (7 derniers jours)
    seven_days_ago = now - timedelta(days=7)
    logins = (
        db.session.query(
            func.date(LoginLog.timestamp).label("day"), func.count(LoginLog.id)
        )
        .filter(LoginLog.timestamp >= seven_days_ago)
        .group_by("day")
        .order_by("day")
        .all()
    )

    logins_labels = [day.strftime("%Y-%m-%d") for day, _ in logins]
    logins_values = [count for _, count in logins]

    return jsonify(
        {
            "signupsPerWeek": {
                "labels": signups_labels,
                "datasets": [
                    {
                        "label": "Inscriptions",
                        "data": signups_values,
                        "backgroundColor": "rgba(54, 162, 235, 0.5)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "borderWidth": 1,
                    }
                ],
            },
            "specialities": {
                "labels": speciality_labels,
                "datasets": [
                    {
                        "data": speciality_values,
                        "backgroundColor": [
                            "#FF6384",
                            "#36A2EB",
                            "#FFCE56",
                            "#4BC0C0",
                            "#9966FF",
                            "#FF9F40",
                            "#B2FF66",
                        ],
                    }
                ],
            },
            "approvalStatus": {
                "labels": ["Approuvés", "En attente"],
                "datasets": [
                    {
                        "data": [approved, pending],
                        "backgroundColor": ["#2ecc71", "#e74c3c"],
                    }
                ],
            },
            "loginsPerDay": {
                "labels": logins_labels,
                "datasets": [
                    {
                        "label": "Connexions",
                        "data": logins_values,
                        "fill": False,
                        "borderColor": "rgba(255, 99, 132, 1)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    }
                ],
            },
        }
    )


@app.route("/refresh-token", methods=["POST"])
@jwt_required(refresh=True)
def refresh_token():
    current_user = get_jwt_identity()  # Récupère l'identité de l'utilisateur
    new_access_token = create_access_token(
        identity=current_user
    )  # Crée un nouveau jeton d'accès
    return jsonify(access_token=new_access_token)


@app.route("/projects", methods=["POST"])
@jwt_required()
def add_project():
    data = request.get_json()
    new_project = Project(name=data["name"], user_id=get_jwt_identity())
    db.session.add(new_project)
    db.session.commit()
    return (
        jsonify(message="Project created successfully", project_id=new_project.id),
        201,
    )


@app.route("/projects/<int:user_id>", methods=["GET"])
@jwt_required()
def get_projects(user_id):
    # if user_id != get_jwt_identity():
    # return jsonify(message='Unauthorized access'), 401
    projects = Project.query.filter_by(user_id=user_id).all()
    return jsonify([{"id": project.id, "name": project.name} for project in projects])


@app.route("/signup", methods=["POST"])
def signup():
    data = request.form
    file = request.files.get("profileImage")

    if not data:
        return jsonify(message="No form data provided"), 400

    if User.query.filter_by(email=data["email"]).first():
        return jsonify(message="cet email existe déjà"), 400

    hashed_password = generate_password_hash(
        data["password"], method="pbkdf2:sha256", salt_length=8
    )

    # Sauvegarder l’image si présente
    filename = None
    if file:
        filename = secure_filename(file.filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
    else:
        filepath = None

    new_user = User(
        name=data["name"],
        email=data["email"],
        password=hashed_password,
        phone_number=data.get("phoneNumber"),
        speciality=data.get("speciality"),
        hospital=data.get("hospital"),
        profile_image=filepath,
        remember_me=(data.get("rememberMe") == "true"),
        created_at=datetime.utcnow(),
        is_approved=False,
        trial_count=0,
    )

    db.session.add(new_user)
    db.session.commit()
    return jsonify(message="User created successfully"), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify(message="E-mail ou mot de passe invalide!"), 401

    # Cas non approuvé
    if not user.is_approved:
        now = datetime.now(timezone.utc)
        expired = now > user.created_at.replace(tzinfo=timezone.utc) + timedelta(
            weeks=1
        )

        if user.trial_count >= 2 or expired:
            return (
                jsonify(
                    message="La durée d'essai est dépassée ou la période a expiré. Veuillez patienter pour obtenir l'approbation."
                ),
                403,
            )

        # Sinon, on incrémente le nombre d’essais
        user.trial_count += 1
        log = LoginLog(user_id=user.id)
        db.session.add(log)
        db.session.commit()

    access_token = create_access_token(
        identity=str(user.id), additional_claims={"is_admin": user.is_admin}
    )

    return jsonify(access_token=access_token), 200


@app.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()  # c’est juste l’id sous forme de string
    claims = get_jwt()  # dict avec tous les claims, dont "is_admin"

    # Optionnel : récupérer is_admin si besoin
    is_admin = claims.get("is_admin", False)

    user = User.query.get_or_404(int(user_id))  # convertir en int pour la query

    image_url = (
        f"/uploads/{os.path.basename(user.profile_image)}"
        if user.profile_image
        else None
    )

    return jsonify(
        id=user.id,
        name=user.name,
        email=user.email,
        phone=user.phone_number,
        speciality=user.speciality,
        hospital=user.hospital,
        imageUrl=image_url,
        isAdmin=is_admin,
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/projects/<int:project_id>", methods=["DELETE"])
@jwt_required()
def delete_project(project_id):
    try:
        project = Project.query.get_or_404(project_id)

        for imported_file in project.imported_files:
            for modified_file in imported_file.modified_files:
                db.session.delete(modified_file)
            db.session.delete(imported_file)

        db.session.delete(project)
        db.session.commit()

        return jsonify(message="Project deleted successfully"), 200

    except Exception as e:
        print(f"Error deleting project: {e}")
        return jsonify(message="Error deleting project"), 500



@app.route("/import-database/<int:project_id>", methods=["POST"])
@jwt_required()
def import_database(project_id):
    if "database" not in request.files:
        return jsonify(message="No file part"), 400

    file = request.files["database"]

    if file.filename == "":
        return jsonify(message="No selected file"), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        imported_file = ImportedFile(
            filename=filename, filepath=filepath, project_id=project_id
        )
        db.session.add(imported_file)
        db.session.commit()

        # Récupérer l'imported_file pour s'assurer que filepath est bien enregistré
        imported_file = ImportedFile.query.filter_by(
            filename=filename, project_id=project_id
        ).first()

        if imported_file:
            return (
                jsonify(
                    message="Database imported successfully",
                    filepath=imported_file.filepath,
                ),
                200,
            )
        else:
            return jsonify(message="Failed to save database"), 500

    return jsonify(message="File not allowed"), 400


@app.route("/projects/<int:project_id>/imported-files", methods=["GET"])
@jwt_required()
def get_imported_files(project_id):
    # Récupérer le projet en fonction de l'ID
    project = Project.query.get_or_404(project_id)

    # Vérification de l'utilisateur
    # if project.user_id != get_jwt_identity():
    # return jsonify(message='Unauthorized access'), 401

    # Récupérer tous les fichiers importés associés au projet
    imported_files = ImportedFile.query.filter_by(project_id=project_id).all()
    if not imported_files:
        logging.warning(f"No files found for project {project_id}")
        return jsonify(message="No files found"), 404

    all_data = []

    def read_file(file_path):
        try:
            # Lire le fichier en fonction de son extension
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                try:
                    df = pd.read_excel(file_path)
                except ImportError:
                    logging.error(
                        "Missing optional dependency 'openpyxl'. Use pip or conda to install it."
                    )
                    return {"data": [], "last_column_name": None}
            else:
                logging.warning(f"Unsupported file type: {file_path}")
                return []

            # Journaux pour vérifier les données lues
            logging.info(f"Columns: {df.columns.tolist()}")
            logging.info(f"Data preview: \n{df.head()}")

            # Conserver l'ordre des colonnes et transformer en dictionnaires ordonnés
            ordered_columns = df.columns.tolist()
            data = [
                OrderedDict((col, row[col]) for col in ordered_columns)
                for row in df.to_dict(orient="records")
            ]
            # logging.info(f"Data after conversion: {data[:5]}")  # Affiche les 5 premiers enregistrements

            # Retourner les données ainsi que le nom de la dernière colonne
            last_column_name = ordered_columns[-1] if ordered_columns else None
            return {"data": data, "last_column_name": last_column_name}
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {str(e)}")
            return {"data": [], "last_column_name": None}

    # Parcourir tous les fichiers importés
    for file in imported_files:
        logging.info(f"Processing file: {file.filepath}")
        file_data = read_file(file.filepath)
        if file_data["data"]:
            all_data.extend(file_data["data"])
        else:
            logging.warning(f"No data found in file: {file.filepath}")

    if not all_data:
        logging.warning("No data was successfully loaded from any files.")
        return jsonify(message="No data available"), 404

    # Inclure le nom de la dernière colonne dans la réponse JSON
    last_column_name = (
        file_data["last_column_name"] if "last_column_name" in file_data else None
    )
    response_data = {"data": all_data, "last_column_name": last_column_name}
    return jsonify(response_data)


@app.route("/projects/<int:project_id>/set-target", methods=["POST"])
# @cross_origin()  # Si besoin de CORS spécifique ici
def set_target_feature(project_id):
    try:
        # Vérifie que le corps de la requête est en JSON
        if not request.is_json:
            print("❌ Requête sans JSON")
            return jsonify({"error": "Missing JSON in request"}), 400

        data = request.get_json()
        print("✅ JSON reçu:", data)

        # Vérifie la présence de "targetFeature"
        if "targetFeature" not in data:
            return jsonify({"error": "targetFeature field is required"}), 400

        target_feature = data["targetFeature"]
        print(f"🎯 Target feature reçu: {target_feature}")

        # Récupère le fichier associé au projet
        file = ImportedFile.query.filter_by(project_id=project_id).first()
        if not file:
            print("❌ Aucun fichier trouvé pour ce projet.")
            return jsonify({"error": "No data file found for project"}), 404

        print(f"📄 Fichier trouvé: {file.filepath}")

        # Récupère les colonnes disponibles dans le fichier
        columns = get_columns_from_file(file)
        print("📊 Colonnes extraites:", columns)

        if not columns or target_feature not in columns:
            return (
                jsonify(
                    {
                        "error": "Target feature not found in data",
                        "availableFeatures": columns or [],
                    }
                ),
                400,
            )

        # Supprime tout target précédent pour ce projet
        TargetFeature.query.filter_by(project_id=project_id).delete()

        # Crée et sauvegarde le nouveau target
        new_target = TargetFeature(name=target_feature, project_id=project_id)
        db.session.add(new_target)
        db.session.commit()

        print("✅ Target enregistré avec succès")

        return (
            jsonify(
                {
                    "message": "Target feature set successfully",
                    "targetFeature": target_feature,
                }
            ),
            200,
        )

    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        app.logger.error(f"Error setting target feature: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/projects/<int:project_id>/target-feature", methods=["GET"])
def get_target_feature(project_id):
    try:
        # Check if target feature was previously set
        target = TargetFeature.query.filter_by(project_id=project_id).first()

        # Get available columns from file
        file = ModifiedFile.query.filter_by(project_id=project_id).first()
        if not file:
            return jsonify({"error": "No files found"}), 404

        columns = get_columns_from_file(file)

        # Return both stored target and available columns
        return jsonify(
            {
                "columns": columns,
                "currentTarget": target.name if target else None,
                "defaultTarget": columns[0] if columns else None,
            }
        )

    except Exception as e:
        app.logger.error(f"Error getting target feature: {str(e)}")
        return jsonify({"error": str(e)}), 500


def get_columns_from_file(file):
    """Helper function to extract columns from different file types"""
    if file.filepath.endswith(".csv"):
        df = pd.read_csv(file.filepath, encoding="utf-8")
        return df.columns.tolist()
    elif file.filepath.endswith(".json"):
        with open(file.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return list(data[0].keys()) if data and isinstance(data, list) else []
    return []


mod_FOLDER = "./modified_files"


def save_to_csv(data, filepath):
    try:
        df = pd.DataFrame(data)

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].apply(
                lambda x: (
                    str(x).encode("utf-8", errors="ignore").decode("utf-8")
                    if x is not None
                    else ""
                )
            )

        df.to_csv(filepath, index=False, encoding="utf-8-sig")
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde du fichier CSV: {e}")
        raise


import json
import os


def save_file(data, filename):
    if not os.path.exists(mod_FOLDER):
        os.makedirs(mod_FOLDER)

    # Forcer l’extension en .json si ce n’est pas déjà fait
    if not filename.endswith(".json"):
        filename += ".json"

    filepath = os.path.join(mod_FOLDER, filename)

    # Sauvegarde au format JSON (brut, indenté, unicode supporté)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return filepath


def generate_timestamped_filename(extension="csv"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"data_{timestamp}.{extension}"


def generate_professional_filename(project_id, modification):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extraction des mots clés
    if modification:
        keywords = []
        for word in [
            "cleaning",
            "encoding",
            "normalization",
            "scaling",
            "imputation",
            "filtering",
            "selection",
        ]:
            if word in modification.lower():
                keywords.append(word)
    else:
        keywords = ["modification"]

    mod_part = "_".join(keywords)
    return f"modfile_{project_id}_{mod_part}_{timestamp}.csv"


@app.route("/save-data/<int:project_id>", methods=["POST"])
def saveMod(project_id):
    content_encoding = request.headers.get("Content-Encoding", "")
    raw_data = request.get_data()

    try:
        if content_encoding == "gzip":
            decompressed_data = gzip.decompress(raw_data)
            data = json.loads(decompressed_data.decode("utf-8"))
        else:
            data = request.get_json()

        if data is None:
            return jsonify({"error": "Aucune donnée reçue"}), 400

        modification = data.get("modification", "Traitement personnalisé")

        # Récupérer l'indicateur de prédiction
        is_for_prediction = data.get("isForPrediction", False)

        # 🎯 Récupération de la target feature
        target_feature = data.get("target_feature")
        if not target_feature:
            tf = TargetFeature.query.filter_by(project_id=project_id).first()
            target_feature = tf.name if tf else None

        # Fichier parent
        imported_file = ImportedFile.query.filter_by(project_id=project_id).first()
        if not imported_file:
            return jsonify({"error": "Aucun fichier importé trouvé pour ce projet"}), 400

        parent_id = imported_file.id

        # Sauvegarde du fichier
        # Récupérer le flag de prédiction
        is_for_prediction = data.get("isForPrediction", False)

        # Adapter le nom du fichier
        if is_for_prediction:
            filename = f"prediction_v{project_id}_{datetime.now().strftime('%d-%m-%Y_%Hh%M')}.csv"
        else:
            filename = generate_professional_filename(project_id, modification)
        filepath = save_file(data, filename)

        new_file = ModifiedFile(
            filename=filename,
            filepath=filepath,
            project_id=project_id,
            modification=modification,
            parent_id=parent_id,
            target_feature=target_feature,
            is_for_prediction=is_for_prediction,  # ✅ Ajout ici
        )
        db.session.add(new_file)
        db.session.commit()

        return (
            jsonify(
                {
                    "message": "Données sauvegardées avec succès",
                    "filepath": filepath,
                    "file_id": new_file.id,
                    "target_feature": target_feature,
                    "parent_id": parent_id,
                }
            ),
            200,
        )

    except Exception as e:
        app.logger.error(f"Error saving data: {str(e)}")
        return jsonify({"error": str(e)}), 400



@app.route("/historique/<int:project_id>", methods=["GET"])
def historique(project_id):
    app.logger.info(f"Fetching history for project ID: {project_id}")
    try:
        mod_files = (
            ModifiedFile.query.filter_by(project_id=project_id)
            .order_by(ModifiedFile.created_at.desc())
            .all()
        )
        return (
            jsonify(
                [
                    {
                        "id": mf.id,
                        "name": mf.filename,
                        "modification": mf.modification or "Non spécifiée",
                        "created_at": (
                            mf.created_at.strftime("%Y-%m-%d %H:%M:%S")
                            if mf.created_at
                            else "Unknown"
                        ),
                        "parent_id": mf.parent_id,
                        "is_for_prediction": mf.is_for_prediction,  # ✅ ici

                    }
                    for mf in mod_files
                ]
            ),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error fetching history: {str(e)}")
        return jsonify({"error": str(e)}), 500


app.logger.setLevel(logging.INFO)

from flask import send_from_directory

import tempfile
from flask import send_file


@app.route("/download-version/<int:file_id>", methods=["GET"])
@jwt_required()
def download_version(file_id):
    try:
        file = ModifiedFile.query.get_or_404(file_id)

        if not os.path.exists(file.filepath):
            return jsonify({"error": "Fichier introuvable"}), 404

        # Charger le contenu JSON
        with open(file.filepath, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        if "data" not in json_data:
            return jsonify({"error": "Format de fichier invalide"}), 400

        # Convertir en DataFrame
        df = pd.DataFrame(json_data["data"])

        # Sauvegarder dans un fichier temporaire CSV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(temp_file.name, index=False, encoding="utf-8-sig")

        return send_file(temp_file.name, as_attachment=True, download_name="export.csv")

    except Exception as e:
        app.logger.error(f"Erreur téléchargement version : {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/delete-version/<int:file_id>", methods=["DELETE"])
@jwt_required()
def delete_version(file_id):
    try:
        file = ModifiedFile.query.get_or_404(file_id)

        # Supprimer le fichier physique du système
        if os.path.exists(file.filepath):
            os.remove(file.filepath)

        db.session.delete(file)
        db.session.commit()

        return jsonify({"message": "Version supprimée avec succès"}), 200
    except Exception as e:
        app.logger.error(f"Erreur lors de la suppression de la version : {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/fichier/<int:file_id>", methods=["GET"])
def get_file_data(file_id):
    try:
        file = ModifiedFile.query.get_or_404(file_id)

        # Lire les données depuis le fichier
        if file.filepath.endswith(".csv"):
            df = pd.read_csv(file.filepath)
            data = df.replace({np.nan: None}).to_dict("records")
        elif file.filepath.endswith(".json"):
            with open(file.filepath, encoding="utf-8") as f:
                data = json.load(f)
                # Si c'est un objet avec une propriété data, utiliser cette propriété
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # S'assurer que data est une liste
        if not isinstance(data, list):
            data = [data]

        # Préparer la réponse
        response = {
            "data": data,  # Toujours un tableau
            "metadata": {
                "file_id": file.id,
                "target_feature": file.target_feature,
                "created_at": file.created_at.isoformat() if file.created_at else None,
                "modification": file.modification or "Non spécifiée",
            },
        }

        return jsonify(response), 200
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération du fichier {file_id} : {str(e)}")
        return jsonify({"error": str(e)}), 500


from datetime import datetime, timezone
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import random


def safe_json(obj):
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(i) for i in obj]
    return obj


from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline


def clean_dataset(dataset, target):
    """
    Nettoie le dataset avec une gestion stricte des labels.
    - Vérifie explicitement le format de la cible.
    - Supprime les colonnes corrélées entre elles pour réduire le bruit.
    """
    logging.info("Cleaning dataset...")

    # Vérification du format de la cible
    if dataset[target].dtype == object or dataset[target].dtype == str:
        le = LabelEncoder()
        y = le.fit_transform(dataset[target].astype(str))
        logging.info(
            f"Classes encodées : {dict(zip(le.classes_, le.transform(le.classes_)))}"
        )
    else:
        y = pd.to_numeric(dataset[target], errors="coerce")
        if y.isnull().any():
            logging.warning("Valeurs non numériques ou manquantes dans la cible")
            valid_indices = y.notnull()
            y = y[valid_indices]
            dataset = dataset[valid_indices]
        else:
            valid_indices = slice(None)  # Tous les indices valides

    X = dataset.drop(columns=[target])[valid_indices]

    # Vérification de la taille du dataset
    # if X.shape[0] < 50:
    # logging.warning("Dataset trop petit pour un entraînement fiable (< 50 échantillons)")
    # raise ValueError("Dataset trop petit (< 50 échantillons)")

    # Vérification de la distribution des classes
    class_counts = pd.Series(y).value_counts(normalize=True)
    logging.info(f"Distribution des classes : {class_counts.to_dict()}")
    if len(class_counts) < 2:
        raise ValueError("La cible doit contenir au moins deux classes")
    if any(class_counts < 0.1):
        logging.warning(
            "Classe(s) avec moins de 10% des données détectée(s), risque de biais"
        )

    # Gestion des colonnes catégoriques
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        X[col] = X[col].astype(str).replace("nan", np.nan)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].fillna("missing"))
        if len(le.classes_) > 50:
            logging.warning(
                f"Colonne {col} a trop de catégories ({len(le.classes_)}), risque de surapprentissage"
            )

    # Gestion des colonnes numériques
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Suppression des colonnes à faible variance
    selector = VarianceThreshold(
        threshold=0.05
    )  # Seuil augmenté pour plus de sélectivité
    try:
        X = pd.DataFrame(
            selector.fit_transform(X),
            columns=X.columns[selector.get_support()],
            index=X.index,
        )
    except Exception as e:
        logging.error(f"Erreur dans VarianceThreshold : {e}")
        raise ValueError(f"Erreur dans la sélection des caractéristiques : {e}")

    # Suppression des colonnes fortement corrélées entre elles
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    if to_drop:
        logging.info(f"Colonnes fortement corrélées supprimées : {to_drop}")
        X = X.drop(columns=to_drop)

    # Vérification des corrélations avec la cible
    correlations = X.corrwith(y).abs()
    high_corr_cols = correlations[correlations > 0.9].index.tolist()
    if high_corr_cols:
        logging.warning(
            f"Colonnes fortement corrélées avec la cible : {high_corr_cols}. Suppression pour éviter target leakage."
        )
        X = X.drop(columns=high_corr_cols)

    return X, y


def get_model(model_name, is_classification, use_smote=False):
    """
    Retourne un pipeline avec des hyperparamètres optimisés pour les performances attendues.
    - Ajout de class_weight='balanced' pour gérer le déséquilibre des classes.
    - Hyperparamètres ajustés pour SVM, KNN, etc.
    """

    def make_pipeline(model):
        steps = [
            (
                "imputer",
                KNNImputer(n_neighbors=3, weights="distance"),
            ),  # Réduction de n_neighbors
            ("scaler", StandardScaler()),
            ("selector", VarianceThreshold(threshold=0.05)),
        ]
        if is_classification and use_smote:
            steps.append(("smote", SMOTE(random_state=42, k_neighbors=3)))
        steps.append(("clf", model))
        return (
            ImbPipeline(steps) if is_classification and use_smote else Pipeline(steps)
        )

    CLASSIFIERS = {
        "Random Forest": (
            make_pipeline(
                RandomForestClassifier(
                    random_state=42, min_samples_leaf=5, class_weight="balanced"
                )
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10],
                "clf__min_samples_split": [5, 10],
            },
        ),
        "Gradient Boosting": (
            make_pipeline(GradientBoostingClassifier(random_state=42)),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.01, 0.1],
                "clf__max_depth": [3, 5],
            },
        ),
        "Logistic Regression": (
            make_pipeline(
                LogisticRegression(
                    max_iter=1000, random_state=42, class_weight="balanced"
                )
            ),
            {"clf__C": [0.1, 1.0, 10.0], "clf__solver": ["lbfgs", "liblinear"]},
        ),
        "LightGBM": (
            make_pipeline(lgb.LGBMClassifier(random_state=42, class_weight="balanced")),
            {
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [20, 31],
                "clf__learning_rate": [0.01, 0.1],
            },
        ),
        "XGBoost": (
            make_pipeline(xgb.XGBClassifier(eval_metric="logloss", random_state=42)),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.01, 0.1],
            },
        ),
        "Decision Tree": (
            make_pipeline(
                DecisionTreeClassifier(
                    random_state=42, min_samples_leaf=5, class_weight="balanced"
                )
            ),
            {"clf__max_depth": [3, 5], "clf__min_samples_split": [5, 10]},
        ),
        "Support Vector Machine": (
            make_pipeline(
                SVC(probability=True, random_state=42, class_weight="balanced")
            ),
            {
                "clf__C": [10.0, 100.0],  # Augmentation de C pour SVM
                "clf__kernel": ["rbf"],
                "clf__gamma": ["scale", 0.01],
            },
        ),
        "KNN": (
            make_pipeline(
                KNeighborsClassifier(n_neighbors=3)
            ),  # Réduction de n_neighbors
            {"clf__n_neighbors": [3, 5], "clf__weights": ["uniform", "distance"]},
        ),
        "Extra Random Trees": (
            make_pipeline(
                ExtraTreesClassifier(
                    random_state=42, min_samples_leaf=5, class_weight="balanced"
                )
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10],
                "clf__min_samples_split": [5, 10],
            },
        ),
        "Single Layer Perceptron": (
            make_pipeline(MLPClassifier(max_iter=1000, random_state=42)),
            {
                "clf__hidden_layer_sizes": [(50,), (100,)],
                "clf__alpha": [0.001, 0.01],
                "clf__learning_rate": ["constant"],
            },
        ),
    }

    REGRESSORS = {
        "Random Forest": (
            make_pipeline(RandomForestRegressor(random_state=42, min_samples_leaf=5)),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10],
                "clf__min_samples_split": [5, 10],
            },
        ),
        "Gradient Boosting": (
            make_pipeline(GradientBoostingRegressor(random_state=42)),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.01, 0.1],
                "clf__max_depth": [3, 5],
            },
        ),
        "LightGBM": (
            make_pipeline(lgb.LGBMRegressor(random_state=42)),
            {
                "clf__n_estimators": [100, 200],
                "clf__num_leaves": [20, 31],
                "clf__learning_rate": [0.01, 0.1],
            },
        ),
        "XGBoost": (
            make_pipeline(xgb.XGBRegressor(random_state=42)),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.01, 0.1],
            },
        ),
        "Decision Tree": (
            make_pipeline(DecisionTreeRegressor(random_state=42, min_samples_leaf=5)),
            {"clf__max_depth": [3, 5], "clf__min_samples_split": [5, 10]},
        ),
        "Support Vector Machine": (
            make_pipeline(SVR()),
            {
                "clf__C": [10.0, 100.0],
                "clf__kernel": ["rbf"],
                "clf__gamma": ["scale", 0.01],
            },
        ),
        "Stochastic Gradient Descent": (
            make_pipeline(SGDRegressor(random_state=42)),
            {"clf__alpha": [0.001, 0.01], "clf__penalty": ["l2", "l1"]},
        ),
        "KNN": (
            make_pipeline(KNeighborsRegressor(n_neighbors=3)),
            {"clf__n_neighbors": [3, 5], "clf__weights": ["uniform", "distance"]},
        ),
        "Extra Random Trees": (
            make_pipeline(ExtraTreesRegressor(random_state=42, min_samples_leaf=5)),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [5, 10],
                "clf__min_samples_split": [5, 10],
            },
        ),
        "Single Layer Perceptron": (
            make_pipeline(MLPRegressor(max_iter=1000, random_state=42)),
            {
                "clf__hidden_layer_sizes": [(50,), (100,)],
                "clf__alpha": [0.001, 0.01],
                "clf__learning_rate": ["constant"],
            },
        ),
        "Lasso Path": (
            make_pipeline(Lasso(random_state=42)),
            {"clf__alpha": [0.01, 0.1, 1.0]},
        ),
    }

    MODELS = CLASSIFIERS if is_classification else REGRESSORS
    if model_name not in MODELS:
        raise ValueError(
            f"Modèle '{model_name}' non reconnu parmi : {list(MODELS.keys())}"
        )
    return MODELS[model_name]


def safe_predict(model, X):
    """
    Effectue une prédiction sécurisée avec journalisation.
    """
    try:
        if hasattr(model, "predict_proba"):
            y_pred = model.predict(X)
            logging.info(f"Prédictions (premiers 5) : {y_pred[:5]}")
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            y_pred = (
                np.argmax(decision, axis=1)
                if len(decision.shape) > 1
                else (decision > 0).astype(int)
            )
            logging.info(f"Prédictions (decision_function, premiers 5) : {y_pred[:5]}")
        else:
            y_pred = model.predict(X)
            logging.info(f"Prédictions (premiers 5) : {y_pred[:5]}")
        return y_pred
    except Exception as e:
        logging.error(f"Erreur de prédiction : {e}")
        raise RuntimeError(f"Erreur de prédiction : {e}")


def calculate_metrics(y_true, y_pred, metrics, task_type, model=None, X=None):
    """
    Calcule les métriques avec une validation stricte.
    - Vérifie l’alignement des labels.
    - Ajoute une analyse des prédictions incorrectes.
    """
    results = {}
    try:
        if len(y_true) != len(y_pred):
            logging.error(
                f"Incohérence de taille : y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
            )
            raise ValueError("Les tailles de y_true et y_pred ne correspondent pas")

        labels = np.unique(y_true)
        pred_labels = np.unique(y_pred)
        logging.info(f"Classes dans y_true : {labels}")
        logging.info(f"Classes dans y_pred : {pred_labels}")
        if not np.all(np.isin(pred_labels, labels)):
            logging.warning("Certaines classes prédites ne sont pas dans y_true")

        if task_type == "classification":
            if len(labels) < 2:
                raise ValueError("Le target doit contenir au moins deux classes")
            class_counts = pd.Series(y_true).value_counts(normalize=True)
            logging.info(
                f"Distribution des classes dans y_true : {class_counts.to_dict()}"
            )
            if any(class_counts < 0.1):
                logging.warning(
                    "Classe(s) avec moins de 10% des données détectée(s), risque de biais"
                )

            if "Accuracy" in metrics:
                acc = accuracy_score(y_true, y_pred)
                results["Accuracy"] = acc
                if abs(acc - 0.375) < 0.01:
                    logging.warning(
                        "Accuracy = 0.375 détectée, vérifiez si le modèle prédit une classe dominante ou aléatoire"
                    )
                    incorrect = y_true != y_pred
                    logging.info(
                        f"Exemples mal classés (indices) : {np.where(incorrect)[0]}"
                    )
            if "Precision" in metrics:
                results["Precision"] = precision_score(
                    y_true, y_pred, average="weighted", labels=labels, zero_division=0
                )
            if "Recall" in metrics:
                results["Recall"] = recall_score(
                    y_true, y_pred, average="weighted", labels=labels, zero_division=0
                )
            if "F1 Score" in metrics:
                results["F1 Score"] = f1_score(
                    y_true, y_pred, average="weighted", labels=labels, zero_division=0
                )
            if "ROC AUC" in metrics and model is not None and X is not None:
                try:
                    if len(labels) == 2:
                        y_pred_proba = (
                            model.predict_proba(X)[:, 1]
                            if hasattr(model, "predict_proba")
                            else y_pred
                        )
                        results["ROC AUC"] = roc_auc_score(y_true, y_pred_proba)
                    else:
                        y_pred_proba = (
                            model.predict_proba(X)
                            if hasattr(model, "predict_proba")
                            else y_pred
                        )
                        results["ROC AUC"] = roc_auc_score(
                            y_true, y_pred_proba, multi_class="ovr"
                        )
                except Exception as e:
                    logging.error(f"Erreur calcul ROC AUC : {e}")
                    results["ROC AUC"] = 0.0
        elif task_type == "regression":
            if "Mean Absolute Error" in metrics:
                results["Mean Absolute Error"] = mean_absolute_error(y_true, y_pred)
            if "Mean Squared Error" in metrics:
                results["Mean Squared Error"] = mean_squared_error(y_true, y_pred)
            if "R² Score" in metrics:
                results["R² Score"] = r2_score(y_true, y_pred)
    except Exception as e:
        logging.error(f"Erreur calcul métriques : {e}")
        results["error"] = str(e)
    return results


def extract_feature_importance(model_or_pipeline, original_columns):
    """
    Retourne un dict {col: importance} avec importances normalisées (somme = 1).
    - model_or_pipeline : soit un pipeline complet (avec named_steps), soit l'estimateur final.
    - original_columns : liste des noms de colonnes avant pipeline.
    """
    try:
        # 1) Détecter si on a un pipeline
        pipeline = None
        if hasattr(model_or_pipeline, "named_steps"):
            pipeline = model_or_pipeline
            # estimator final (ex: 'clf')
            estimator = pipeline.named_steps.get("clf", None) or model_or_pipeline
        else:
            estimator = model_or_pipeline

        # 2) Récupérer les colonnes effectivement gardées par le selector (si présent)
        selected_columns = list(original_columns)
        selector = None
        if pipeline:
            # chercher un selector courant (VarianceThreshold, SelectKBest, etc.)
            for name, step in pipeline.named_steps.items():
                if hasattr(step, "get_support") and callable(step.get_support):
                    selector = step
                    break
        if selector is not None:
            try:
                support = selector.get_support()
                selected_columns = [c for c, s in zip(original_columns, support) if s]
            except Exception:
                # fallback : garder original_columns
                selected_columns = list(original_columns)

        # 3) Extraire l'importance brute
        importances = None
        if hasattr(estimator, "feature_importances_"):
            arr = np.array(estimator.feature_importances_, dtype=float)
            importances = arr
        elif hasattr(estimator, "coef_"):
            coefs = np.array(estimator.coef_, dtype=float)
            # si multi-classe -> moyenne des valeurs absolues par feature
            if coefs.ndim == 1:
                arr = np.abs(coefs)
            else:
                arr = np.mean(np.abs(coefs), axis=0)
            importances = arr
        else:
            return {}

        # 4) Si longueur ne correspond pas -> tenter d'ajuster (trim/pad) ou échouer proprement
        if len(importances) != len(selected_columns):
            # cas courant : l'estimateur a moins de features (selector interne) ou shape différente
            # On essaie de loguer et de normaliser en prenant la min-len (safe)
            logging.warning(
                f"Feature importance length mismatch: importances={len(importances)}, columns={len(selected_columns)}. "
                "Attempting safe alignment."
            )
            min_len = min(len(importances), len(selected_columns))
            importances = importances[:min_len]
            selected_columns = selected_columns[:min_len]

        # 5) Normaliser en relative importances (somme = 1)
        importances = np.array(importances, dtype=float)
        # assurer non-négatif
        importances = np.abs(importances)
        s = importances.sum()
        if s <= 0 or np.isnan(s):
            # fallback : répartir équitablement
            n = len(importances) if len(importances) > 0 else len(selected_columns)
            if n == 0:
                return {}
            normalized = np.ones(n) / n
        else:
            normalized = importances / s

        # 6) Retourner mapping col -> importance (float entre 0 et 1)
        return dict(zip(selected_columns, normalized.tolist()))

    except Exception as e:
        logging.error(f"Erreur extraction importances : {e}")
        return {}



@app.route("/train/<int:project_id>", methods=["POST"])
@jwt_required()
def train_model(project_id):
    """
    Entraîne un modèle avec des protections contre les biais et les erreurs de prédiction.
    - Utilise une séparation train/test plus robuste.
    - Vérifie la taille des folds pour éviter des ensembles de test trop petits.
    """
    try:
        data = request.get_json()
        version_id = data.get("versionId")
        method = data.get("method", "split")
        custom_code = data.get("custom_code", "").strip()
        is_classification = data.get("task") == "classification"
        use_grid_search = data.get("use_grid_search", True)
        use_smote = (
            is_classification
            if data.get("use_smote") is None
            else data.get("use_smote", False)
        )

        SEED = 42
        np.random.seed(SEED)

        # Validation des champs requis
        required_fields = ["data", "targ", "metrics", "task"]
        if not data.get("model") and not custom_code:
            return (
                jsonify(
                    {
                        "error": "Veuillez sélectionner au moins un modèle ou fournir du code personnalisé"
                    }
                ),
                400,
            )

        required_fields += ["k"] if method == "kfold" else ["trainset"]
        missing = [
            f for f in required_fields if f not in data or data[f] in [None, "", []]
        ]
        if missing:
            return jsonify({"error": f"Champs manquants : {', '.join(missing)}"}), 400

        dataset = pd.DataFrame(data["data"])
        target = data["targ"].strip()
        if target not in dataset.columns:
            return (
                jsonify({"error": f"Target '{target}' introuvable dans les données"}),
                400,
            )

        # Nettoyage initial des données
        X, y = clean_dataset(dataset, target)
        if is_classification and len(np.unique(y)) < 2:
            return (
                jsonify({"error": "Le target doit contenir au moins deux classes"}),
                400,
            )

        project = Project.query.get_or_404(project_id)
        trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        results = []

        os.makedirs(mod_FOLDER, exist_ok=True)

        custom_model = None
        if custom_code:
            try:
                exec_env = {}
                exec(custom_code, {}, exec_env)
                if "model" not in exec_env:
                    return (
                        jsonify(
                            {"error": "Votre code doit définir une variable `model`."}
                        ),
                        400,
                    )
                custom_model = exec_env["model"]
                logging.info("✅ Modèle personnalisé chargé avec succès.")

                from sklearn.pipeline import Pipeline
                from sklearn.impute import SimpleImputer

                # Crée un pipeline avec imputation si ce n’est pas déjà fait
                if not isinstance(custom_model, Pipeline):
                    custom_model = Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="mean")),
                            ("clf", custom_model),
                        ]
                    )
            except Exception as e:
                logging.error(f"Erreur dans le code personnalisé : {e}")
                return (
                    jsonify({"error": f"Erreur dans le code personnalisé : {str(e)}"}),
                    400,
                )

        if method == "kfold":
            k = int(data["k"])
            if k < 2 or k > 10:
                return (
                    jsonify(
                        {
                            "error": "k doit être entre 2 et 10 pour la validation croisée"
                        }
                    ),
                    400,
                )
            if len(y) / k < 10:
                logging.warning(
                    "Taille des folds trop petite (< 10 échantillons), risque de résultats instables"
                )
                k = max(2, len(y) // 10)  # Ajustement dynamique de k
                logging.info(
                    f"Ajustement de k à {k} pour garantir des folds suffisamment grands"
                )
            cv = (
                StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
                if is_classification
                else KFold(n_splits=k, shuffle=True, random_state=SEED)
            )

            model_list = data.get("model", [])
            if custom_model:
                model_list.append("CustomModel")  # Nom arbitraire pour le modèle custom

            for model_name in model_list:
                if model_name == "CustomModel":
                    model = custom_model
                    param_grid = {}
                else:
                    model, param_grid = get_model(
                        model_name, is_classification, use_smote
                    )

                scores = []
                kfold_scores = []
                train_scores = []
                best_params = None
                best_model = None

                for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Vérification de la séparation des données
                    if np.any(np.isin(X_test.index, X_train.index)):
                        logging.error(
                            "Fuite de données : indices de test présents dans l’ensemble d’entraînement"
                        )
                        return (
                            jsonify(
                                {
                                    "error": "Fuite de données détectée dans la validation croisée"
                                }
                            ),
                            500,
                        )

                    # Vérification de la taille du fold
                    if len(y_test) < 5:
                        logging.warning(
                            f"Fold {fold+1} trop petit ({len(y_test)} échantillons), risque de résultats instables"
                        )
                        continue

                    # Ajustement du modèle
                    if use_grid_search:
                        grid = GridSearchCV(
                            model,
                            param_grid or {},
                            scoring="accuracy" if is_classification else "r2",
                            cv=3,
                            n_jobs=-1,
                            return_train_score=True,
                        )
                        grid.fit(X_train, y_train)
                        model = grid.best_estimator_
                        best_params = grid.best_params_
                        best_model = model
                    else:
                        model.fit(X_train, y_train)
                        best_model = model

                    # Prédictions sur l’ensemble de test
                    y_pred = safe_predict(model, X_test)
                    score = calculate_metrics(
                        y_test, y_pred, data["metrics"], data["task"], model, X_test
                    )
                    scores.append(score)
                    kfold_scores.append(score.get(data["metrics"][0], 0))

                    # Prédictions sur l’ensemble d’entraînement
                    y_train_pred = safe_predict(model, X_train)
                    train_score = calculate_metrics(
                        y_train,
                        y_train_pred,
                        data["metrics"],
                        data["task"],
                        model,
                        X_train,
                    )
                    train_scores.append(train_score)

                    # Vérification de l’overfitting
                    if (
                        is_classification
                        and score.get("Accuracy", 0) < 0.4
                        and train_score.get("Accuracy", 0) > 0.9
                    ):
                        logging.warning(
                            f"Overfitting détecté pour {model_name} dans le fold {fold+1}"
                        )

                # Calcul des métriques moyennes et écarts-types
                avg_score = {
                    m: np.mean([s[m] for s in scores if m in s])
                    for m in data["metrics"]
                }
                std_score = {
                    m: np.std([s[m] for s in scores if m in s]) for m in data["metrics"]
                }
                avg_train_score = {
                    m: np.mean([s[m] for s in train_scores if m in s])
                    for m in data["metrics"]
                }

                feature_importances = extract_feature_importance(
                    best_model.named_steps["clf"], X.columns
                )

                conf_matrix, labels = None, None
                if is_classification and len(np.unique(y_test)) > 1:
                    labels = np.unique(y_test)
                    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
                    logging.info(
                        f"Matrice de confusion pour {model_name} : {conf_matrix}"
                    )

                # Sauvegarde du modèle
                model_filename = f"model_{project_id}_{model_name}_{trained_at.replace(':', '-')}.joblib"
                model_path = os.path.join(mod_FOLDER, model_filename)
                joblib.dump(best_model, model_path)
                
                # Sauvegarder aussi les colonnes d'entraînement
                # Sauvegarde aussi les colonnes d'entraînement
                columns_path = model_path.replace(".joblib", "_columns.json")
                with open(columns_path, "w", encoding="utf-8") as f:
                    json.dump(list(X.columns), f)



                model_entry = Models(
                    modelname=model_name,
                    modelpath=model_path,
                    validpath=model_path,
                    Accuracy=avg_score.get("Accuracy", 0.0),
                    Precisionn=avg_score.get("Precision", 0.0),
                    Recall=avg_score.get("Recall", 0.0),
                    F1_Score=avg_score.get("F1 Score", 0.0),
                    ROC_AUC=avg_score.get("ROC AUC", 0.0),
                    MeanAbsoluteError=avg_score.get("Mean Absolute Error", 0.0),
                    MeanSquaredError=avg_score.get("Mean Squared Error", 0.0),
                    RScore=avg_score.get("R² Score", 0.0),
                    featureimportance=json.dumps(feature_importances),
                    project_id=project_id,
                    file_id=version_id,
                    trainingset=None,
                    testset=None,
                    k=k,
                    confusion_matrix=json.dumps(
                        conf_matrix.tolist() if conf_matrix is not None else None
                    ),
                    confusion_labels=json.dumps(
                        labels.tolist() if labels is not None else None
                    ),
                )

                results.append(
                    {
                        "model": model_name,
                        "method": f"{k}-Fold Cross Validation",
                        "metrics": avg_score,
                        "std_metrics": std_score,
                        "train_metrics": avg_train_score,
                        "kfold_scores": kfold_scores,
                        "feature_importances": feature_importances,
                        "best_params": best_params,
                        "used_smote": use_smote,
                        "version_id": version_id,
                        "confusion_matrix": (
                            conf_matrix.tolist() if conf_matrix is not None else None
                        ),
                        "confusion_labels": (
                            labels.tolist() if labels is not None else None
                        ),
                        "model_path": model_path,
                        "file_id": version_id,
                    }
                )

        else:
            train_size = float(data["trainset"]) / 100.0
            test_size = float(data.get("testset", 0)) / 100.0
            val_size = float(data.get("valtest", 0)) / 100.0

            if train_size + test_size + val_size > 1.0 or train_size <= 0:
                return (
                    jsonify(
                        {"error": "Les tailles de train/test/val ne sont pas valides"}
                    ),
                    400,
                )

            stratify = y if is_classification else None
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, train_size=train_size, random_state=SEED, stratify=stratify
            )

            X_val, X_test, y_val, y_test = None, None, None, None
            if val_size > 0 and test_size > 0:
                val_ratio = val_size / (val_size + test_size)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=1 - val_ratio,
                    random_state=SEED,
                    stratify=y_temp if is_classification else None,
                )
            elif test_size > 0:
                X_test, y_test = X_temp, y_temp
            else:
                X_val, y_val = X_temp, y_temp

            model_list = data.get("model", [])
            if custom_model:
                model_list.append("CustomModel")  # Nom arbitraire pour le modèle custom

            for model_name in model_list:
                if model_name == "CustomModel":
                    model = custom_model
                    param_grid = {}
                else:
                    model, param_grid = get_model(
                        model_name, is_classification, use_smote
                    )

                best_params = None
                best_model = None

                # Vérification de la séparation des données
                if X_test is not None and np.any(np.isin(X_test.index, X_train.index)):
                    logging.error(
                        "Fuite de données : indices de test présents dans l’ensemble d’entraînement"
                    )
                    return (
                        jsonify(
                            {
                                "error": "Fuite de données détectée dans la séparation train/test"
                            }
                        ),
                        500,
                    )

                if use_grid_search:
                    grid = GridSearchCV(
                        model,
                        param_grid or {},
                        scoring="accuracy" if is_classification else "r2",
                        cv=3,
                        n_jobs=-1,
                        return_train_score=True,
                    )
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    best_params = grid.best_params_
                    best_model = model
                else:
                    model.fit(X_train, y_train)
                    best_model = model

                eval_X = X_val if val_size > 0 else X_test
                eval_y = y_val if val_size > 0 else y_test
                if eval_X is None or eval_y is None:
                    return (
                        jsonify(
                            {"error": "Aucun ensemble de validation ou test disponible"}
                        ),
                        400,
                    )

                y_pred = safe_predict(model, eval_X)
                score = calculate_metrics(
                    eval_y, y_pred, data["metrics"], data["task"], model, eval_X
                )
                train_score = calculate_metrics(
                    y_train,
                    safe_predict(model, X_train),
                    data["metrics"],
                    data["task"],
                    model,
                    X_train,
                )

                test_score = None
                if test_size > 0 and X_test is not None:
                    y_test_pred = safe_predict(model, X_test)
                    test_score = calculate_metrics(
                        y_test,
                        y_test_pred,
                        data["metrics"],
                        data["task"],
                        model,
                        X_test,
                    )

                feature_importances = extract_feature_importance(
                    best_model.named_steps["clf"], X.columns
                )

                conf_matrix, labels = None, None
                if is_classification and len(np.unique(eval_y)) > 1:
                    labels = np.unique(eval_y)
                    conf_matrix = confusion_matrix(eval_y, y_pred, labels=labels)
                    logging.info(
                        f"Matrice de confusion pour {model_name} : {conf_matrix}"
                    )

                model_filename = f"model_{project_id}_{model_name}_{trained_at.replace(':', '-')}.joblib"
                model_path = os.path.join(mod_FOLDER, model_filename)
                joblib.dump(best_model, model_path)
                
                # Sauvegarder aussi les colonnes d'entraînement
                # Sauvegarde aussi les colonnes d'entraînement
                columns_path = model_path.replace(".joblib", "_columns.json")
                with open(columns_path, "w", encoding="utf-8") as f:
                    json.dump(list(X.columns), f)



                model_entry = Models(
                    modelname=model_name,
                    modelpath=model_path,
                    validpath=model_path,
                    Accuracy=score.get("Accuracy", 0.0),
                    Precisionn=score.get("Precision", 0.0),
                    Recall=score.get("Recall", 0.0),
                    F1_Score=score.get("F1 Score", 0.0),
                    ROC_AUC=score.get("ROC AUC", 0.0),
                    MeanAbsoluteError=score.get("Mean Absolute Error", 0.0),
                    MeanSquaredError=score.get("Mean Squared Error", 0.0),
                    RScore=score.get("R² Score", 0.0),
                    featureimportance=json.dumps(feature_importances),
                    project_id=project_id,
                    file_id=version_id,
                    trainingset=train_size * 100,
                    testset=test_size * 100,
                    k=None,
                    confusion_matrix=json.dumps(
                        conf_matrix.tolist() if conf_matrix is not None else None
                    ),
                    confusion_labels=json.dumps(
                        labels.tolist() if labels is not None else None
                    ),
                )

                results.append(
                    {
                        "model": model_name,
                        "method": "Train/Test Split",
                        "train_size": train_size,
                        "test_size": test_size,
                        "val_size": val_size,
                        "metrics": score,
                        "train_metrics": train_score,
                        "test_metrics": test_score,
                        "feature_importances": feature_importances,
                        "best_params": best_params,
                        "used_smote": use_smote,
                        "confusion_matrix": (
                            conf_matrix.tolist() if conf_matrix is not None else None
                        ),
                        "confusion_labels": (
                            labels.tolist() if labels is not None else None
                        ),
                        "model_path": model_path,
                        "file_id": version_id,
                    }
                )

        return (
            jsonify(
                {
                    "results": results,
                    "dataset_name": project.name,
                    "target_feature": target,
                    "task": data["task"],
                    "trained_at": trained_at,
                }
            ),
            200,
        )

    except Exception as e:
        db.session.rollback()
        traceback.print_exc()
        logging.error(f"Erreur dans train_model : {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/models/<int:project_id>", methods=["POST"])
@jwt_required()
def save_model(project_id):
    """
    Sauvegarde un modèle dans la base de données uniquement lorsque l'utilisateur appuie sur le bouton 'Enregistrer'.
    - Vérifie que le fichier modèle existe sur le disque.
    - Valide que file_id est un entier non nul.
    """
    try:
        data = request.get_json()
        required_fields = ["modelname", "modelpath", "validpath", "file_id"]
        print("Contenu reçu dans save_model :", data)

        missing = [
            f for f in required_fields if f not in data or data[f] in [None, "", []]
        ]
        if missing:
            logging.error(f"Champs manquants dans la requête : {', '.join(missing)}")
            return jsonify({"error": f"Champs manquants : {', '.join(missing)}"}), 400

        # Validation de file_id
        file_id = data.get("file_id")
        if not isinstance(file_id, int) or file_id <= 0:
            logging.error(f"file_id invalide : {file_id}")
            return (
                jsonify({"error": "file_id doit être un entier positif non nul"}),
                400,
            )
        logging.info(f"file_id reçu : {file_id}")

        # Vérifier si le fichier modèle existe
        model_path = data["modelpath"]
        if not os.path.exists(model_path):
            logging.error(f"Le fichier modèle {model_path} n'existe pas sur le disque")
            return (
                jsonify(
                    {
                        "error": f"Le fichier modèle {model_path} n'existe pas sur le disque"
                    }
                ),
                400,
            )

        train_size = data.get("trainingset", 0)
        test_size = data.get("testset", 0)
        logging.info(f"Sizes reçues : train={train_size}, test={test_size}")

        # Gestion de feature_importance
        feature_importance_data = data.get("featureimportance", {})
        if isinstance(feature_importance_data, str):
            try:
                feature_importance_data = json.loads(feature_importance_data)
            except Exception as e:
                logging.error(f"Erreur lors du parsing de featureimportance : {e}")
                feature_importance_data = {}

        # Création de l'entrée dans la base de données
        model = Models(
            modelname=data["modelname"],
            modelpath=data["modelpath"],
            validpath=data["validpath"],
            target_feature=data.get("target_feature"),
            trainingset=train_size,
            testset=test_size,
            Accuracy=data.get("Accuracy", 0.0),
            Precisionn=data.get("Precision", 0.0),
            Recall=data.get("Recall", 0.0),
            F1_Score=data.get("F1_Score", 0.0),
            ROC_AUC=data.get("ROC_AUC", 0.0),
            MeanAbsoluteError=data.get("MeanAbsoluteError", 0.0),
            MeanSquaredError=data.get("MeanSquaredError", 0.0),
            RScore=data.get("RScore", 0.0),
            featureimportance=json.dumps(feature_importance_data),
            confusion_matrix=data.get("confusion_matrix_json"),
            confusion_labels=data.get("confusion_labels_json"),
            project_id=project_id,
            file_id=file_id,
        )

        db.session.add(model)
        db.session.commit()

        return jsonify({"message": "Modèle sauvegardé avec succès"}), 201

    except Exception as e:
        db.session.rollback()
        logging.error(f"Erreur dans save_model : {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/models/<int:project_id>/version/<int:file_id>", methods=["GET"])
@jwt_required()
def get_models_by_version(project_id, file_id):
    try:
        app.logger.info(f"Fetching models for project {project_id}, file {file_id}")

        if not ModifiedFile.query.filter_by(id=file_id, project_id=project_id).first():
            return jsonify({"error": "File version not found"}), 404

        models = Models.query.filter_by(project_id=project_id, file_id=file_id).all()
        result = []

        for m in models:
            try:
                feature_importance = {}
                if m.featureimportance:
                    feature_importance = json.loads(m.featureimportance)

                model_data = {
                    "id": m.id,
                    "modelname": m.modelname,
                    "Accuracy": float(m.Accuracy) if m.Accuracy is not None else 0.0,
                    "Precisionn": (
                        float(m.Precisionn) if m.Precisionn is not None else 0.0
                    ),
                    "Recall": float(m.Recall) if m.Recall is not None else 0.0,
                    "F1_Score": float(m.F1_Score) if m.F1_Score is not None else 0.0,
                    "ROC_AUC": float(m.ROC_AUC) if m.ROC_AUC is not None else 0.0,
                    "MeanAbsoluteError": (
                        float(m.MeanAbsoluteError)
                        if m.MeanAbsoluteError is not None
                        else 0.0
                    ),
                    "MeanSquaredError": (
                        float(m.MeanSquaredError)
                        if m.MeanSquaredError is not None
                        else 0.0
                    ),
                    "RScore": float(m.RScore) if m.RScore is not None else 0.0,
                    "featureimportance": feature_importance,
                    "trainingset": getattr(m, "trainingset", None),
                    "testset": getattr(m, "testset", None),
                    "valset": getattr(m, "valset", None),
                }
                result.append(model_data)
            except json.JSONDecodeError as e:
                app.logger.error(
                    f"Invalid JSON in featureimportance for model {getattr(m, 'id', 'unknown')}: {e}"
                )
                continue
            except Exception as e:
                app.logger.error(
                    f"Unexpected error for model {getattr(m, 'id', 'unknown')}: {e}"
                )
                continue

        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Error in get_models_by_version: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/models/<int:model_id>", methods=["DELETE"])
@jwt_required()
def delete_model(model_id):
    model = Models.query.get_or_404(model_id)

    # Supprimer le fichier du modèle si nécessaire
    if os.path.exists(model.modelpath):
        os.remove(model.modelpath)
    if os.path.exists(model.validpath):
        os.remove(model.validpath)

    db.session.delete(model)
    db.session.commit()
    return jsonify({"message": "Modèle supprimé"}), 200


@app.route("/models/<int:project_id>", methods=["GET"])
def models(project_id):
    logging.debug(f"Received data: {project_id}")
    try:
        mods = Models.query.filter_by(project_id=project_id).all()
        return jsonify([{"id": mf.id, "name": mf.modelname} for mf in mods]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model/<int:file_id>", methods=["GET"])
def model(file_id):
    logging.debug(f"Received data: {file_id}")
    try:
        # Requête pour obtenir le modèle avec l'id spécifié
        mf = Models.query.filter_by(id=file_id).first()

        # Si le modèle n'est pas trouvé
        if mf is None:
            return jsonify({"error": "Model not found"}), 404

        # Créer la réponse avec les données du modèle
        return (
            jsonify(
                {
                    "name": mf.modelname,
                    "Accuracy": mf.Accuracy,
                    "Precisionn": mf.Precisionn,
                    "Recall": mf.Recall,
                    "F1_Score": mf.F1_Score,
                    "ROC_AUC": mf.ROC_AUC,
                    "MeanAbsoluteError": mf.MeanAbsoluteError,
                    "MeanSquaredError": mf.MeanSquaredError,
                    "RScore": mf.RScore,
                    "featureimportance": mf.featureimportance,
                    "trainingset": mf.trainingset,
                    "testset": mf.testset,
                    "k": mf.k,
                }
            ),
            200,
        )

    except Exception as e:
        logging.error(f"Error retrieving model: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/model-details/<int:model_id>", methods=["GET"])
@jwt_required()
def get_model_details(model_id):
    model = Models.query.get_or_404(model_id)

    # Convertir featureimportance (stocké comme JSON string) en dict
    try:
        feature_importance = (
            json.loads(model.featureimportance) if model.featureimportance else {}
        )
    except:
        feature_importance = {}

    # Récupérer la matrice de confusion si elle existe
    confusion_data = {}
    if model.confusion_matrix:  # Ajoutez ce champ à votre modèle si nécessaire
        try:
            confusion_data = {
                "matrix": json.loads(model.confusion_matrix),
                "labels": (
                    json.loads(model.confusion_labels) if model.confusion_labels else []
                ),
            }
        except:
            confusion_data = {}

    return (
        jsonify(
            {
                "model": model.modelname,
                "metrics": {
                    "Accuracy": model.Accuracy,
                    "Precision": model.Precisionn,
                    "Recall": model.Recall,
                    "F1 Score": model.F1_Score,
                    "ROC AUC": model.ROC_AUC,
                    "Mean Absolute Error": model.MeanAbsoluteError,
                    "Mean Squared Error": model.MeanSquaredError,
                    "R² Score": model.RScore,
                },
                "feature_importances": feature_importance,
                "method": (
                    f"{model.k}-Fold CV"
                    if model.k
                    else f"Train {model.trainingset}% / Test {model.testset}%"
                ),
                "train_size": model.trainingset,
                "test_size": model.testset,
                "k": model.k,
                "dataset_name": model.file.filename if model.file else "N/A",
                "target_feature": model.file.target_feature if model.file else "N/A",
                "task": "regression" if hasattr(model, "RScore") else "classification",
                "confusion_matrix": confusion_data.get("matrix"),
                "confusion_labels": confusion_data.get("labels"),
            }
        ),
        200,
    )


@app.route("/deploy/<int:file_id>", methods=["GET"])
def deploy_model(file_id):
    try:
        model = ModelFile.query.get(file_id)
        if not model:
            return jsonify({"error": "Modèle introuvable"}), 404

        with open(model.file_path, "rb") as f:
            loaded_model = pickle.load(f)

        metadata = json.loads(model.feature_metadata or "{}")

        return jsonify(
            {
                "model_name": model.name,
                "features": metadata.get("features", []),
                "target": metadata.get("target", ""),
                "created_at": model.created_at.isoformat(),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


def predict_from_dataframe(model, df, training_columns):
    """
    Nettoie les données d'entrée, aligne les colonnes et effectue une prédiction
    avec le modèle scikit-learn fourni.

    Args:
        model: modèle entraîné (scikit-learn, XGBoost, etc.)
        df (pd.DataFrame): données d'entrée à prédire
        training_columns (list): liste des colonnes attendues par le modèle

    Returns:
        list: prédictions
    """
    # 1. Remplacer les chaînes vides par NaN
    df.replace('', np.nan, inplace=True)

    # 2. Conversion explicite des colonnes numériques
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # Certaines colonnes resteront object/catégorielles

    # 3. Encodage des colonnes object (catégorielles)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("inconnu")
        df[col] = df[col].astype(str)
        df[col] = pd.factorize(df[col])[0]

    # 4. Imputation des valeurs manquantes restantes
    df.fillna(0, inplace=True)

    # 5. Ajouter les colonnes manquantes et aligner l'ordre
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]

    # 6. Prédiction
    if hasattr(model, "predict_proba"):
        predictions = model.predict(df)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(df)
        if len(decision.shape) > 1:
            predictions = np.argmax(decision, axis=1)
        else:
            predictions = (decision > 0).astype(int)
    else:
        predictions = model.predict(df)

    return predictions.tolist()

from sklearn.metrics import accuracy_score, confusion_matrix


@app.route("/projects/<int:project_id>/predict", methods=["POST"])
@jwt_required()
def predict_model(project_id):
    try:
        app.logger.info("✅ [PREDICT] Requête reçue")
        app.logger.info(f"🔍 Content-Type: {request.content_type}")

        # 🔍 Identifier le modèle à utiliser
        model_name = None
        if request.content_type.startswith("multipart/form-data"):
            model_name = request.form.get("model")
        else:
            try:
                json_data = request.get_json(force=True)
                model_name = json_data.get("model") if json_data else None
            except Exception as e:
                return jsonify({"error": f"Échec parsing JSON : {str(e)}"}), 400

        if not model_name:
            return jsonify({"error": "Aucun modèle spécifié"}), 400

        # 📦 Charger le modèle
        model_record = Models.query.filter_by(project_id=project_id, modelname=model_name).first()
        if not model_record or not os.path.exists(model_record.modelpath):
            return jsonify({"error": f"Modèle ou fichier introuvable : {model_name}"}), 404

        model_path = model_record.modelpath
        trained_model = joblib.load(model_path)
        app.logger.info("✅ Modèle chargé avec succès")

        # 📄 Charger les colonnes d'entraînement
        columns_path = model_path.replace(".joblib", "_columns.json")
        if not os.path.exists(columns_path):
            return jsonify({"error": f"Fichier colonnes introuvable : {columns_path}"}), 500

        with open(columns_path, "r", encoding="utf-8") as f:
            training_columns = json.load(f)

        # 📂 Lire les données utilisateur
        df = None
        if request.content_type.startswith("multipart/form-data") and "file" in request.files:
            uploaded_file = request.files["file"]
            if not uploaded_file.filename.lower().endswith(".csv"):
                return jsonify({"error": "Le fichier doit être un .csv"}), 400
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            try:
                json_data = request.get_json(force=True)
                if json_data and "data" in json_data:
                    input_data = json_data["data"]
                    if isinstance(input_data, dict):
                        input_data = [input_data]
                    df = pd.DataFrame(input_data)
            except Exception as e:
                return jsonify({"error": f"Erreur parsing JSON : {str(e)}"}), 400

        if df is None:
            return jsonify({"error": "Aucune donnée valide reçue pour la prédiction."}), 400

        app.logger.info(f"📋 Données reçues : {df.shape}")


        # Prédiction sans utiliser safe_predict
        try:
            predictions = predict_from_dataframe(trained_model, df, training_columns)
            result_lines = [f"{pred}" for pred in predictions]

            metrics = {}
            confusion = None
            labels = None

            # Si target dispo dans le fichier => calculer métriques
            if target_feature := model_record.target_feature:
                if target_feature in df.columns:
                    y_true = df[target_feature]
                    y_pred = predictions
                    metrics = {
                        "accuracy": accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
                    }
                    confusion = confusion_matrix(y_true, y_pred).tolist()
                    labels = sorted(list(set(y_true) | set(y_pred)))
                    
        
                # Avant le return
            response_payload = {
                "predictions": result_lines,
                "metrics": metrics,
                "confusion_matrix": confusion,
                "confusion_labels": labels
            }

            app.logger.info(f"📤 [PREDICT] Réponse envoyée : {json.dumps(response_payload, indent=2, ensure_ascii=False)}")

            return jsonify(response_payload)


            return jsonify({
                "predictions": result_lines,
                "metrics": metrics,
                "confusion_matrix": confusion,
                "confusion_labels": labels
            })
            
            

        except Exception as e:
            app.logger.error(f"❌ Erreur lors de la prédiction : {e}")
            return jsonify({"error": f"Erreur de prédiction : {str(e)}"}), 500


    except Exception as e:
        app.logger.exception(f"❌ Erreur globale : {str(e)}")
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from flask import send_file
import io

@app.route("/export-prediction-pdf/<int:project_id>", methods=["POST"])
@jwt_required()
def export_prediction_pdf(project_id):
    try:
        data = request.get_json()
        predictions = data.get("predictions", [])
        model_name = data.get("model", "Modèle inconnu")
        confusion_matrix_data = data.get("confusion_matrix", None)
        confusion_labels = data.get("confusion_labels", None)
        date_now = datetime.now().strftime("%d/%m/%Y %H:%M")
        user_id = get_jwt_identity()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"<b>Rapport de Prédiction</b>", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Projet ID : {project_id}", styles["Normal"]))
        story.append(Paragraph(f"Modèle utilisé : {model_name}", styles["Normal"]))
        story.append(Paragraph(f"Date : {date_now}", styles["Normal"]))
        story.append(Paragraph(f"Utilisateur ID : {user_id}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Résultats de prédiction
        story.append(Paragraph(f"<b>Résultats de prédiction :</b>", styles["Heading2"]))
        data_table = [["Index", "Prédiction"]]
        for i, pred in enumerate(predictions):
            data_table.append([str(i + 1), str(pred)])
        table = Table(data_table, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(table)

        # Métriques
        metrics = data.get("metrics", {})
        if metrics:
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Métriques du modèle :</b>", styles["Heading2"]))
            metrics_table = [["Métrique", "Valeur (%)"]]
            for name, value in metrics.items():
                metrics_table.append([name, f"{value*100:.2f}"])
            table = Table(metrics_table, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            story.append(table)

        # Matrice de confusion
        if confusion_matrix_data and confusion_labels:
            story.append(Spacer(1, 12))
            story.append(Paragraph("<b>Matrice de confusion :</b>", styles["Heading2"]))

            # Construire l'en-tête
            cm_table_data = [[""] + confusion_labels]
            # Ajouter chaque ligne avec le label en première colonne
            for label, row in zip(confusion_labels, confusion_matrix_data):
                cm_table_data.append([label] + row)

            cm_table = Table(cm_table_data, hAlign="LEFT")
            cm_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))
            story.append(cm_table)

        doc.build(story)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="rapport_prediction.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500







if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    with app.app_context():
        db.create_all()
    app.run(debug=True)
