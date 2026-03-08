from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import re

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'supersecretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///talentnivad.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    company = db.Column(db.String(100))
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200))
    email = db.Column(db.String(200))
    phone = db.Column(db.String(50))
    location = db.Column(db.String(200))
    experience = db.Column(db.String(100))
    status = db.Column(db.String(100), default="New")
    notes = db.Column(db.Text)
    resume_filename = db.Column(db.String(200))



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store candidate profiles (in production, use a database)



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(filepath):
    if filepath.endswith(".pdf"):
        text = ""
        with open(filepath, "rb") as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif filepath.endswith(".docx"):
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        return ""


def generate_candidate_profile(resume_text, filename):
    """Generate a candidate profile from resume text"""
    # Extract basic information using simple parsing
    lines = resume_text.split('\n')
    name = "Unknown Candidate"
    email = "No email found"
    phone = "No phone found"
    location = "Location not specified"
    experience_years = "Experience not specified"
    
    # Simple extraction logic
    for line in lines:
        line_lower = line.lower().strip()
        if '@' in line and '.' in line:
            email = line.strip()
        elif any(char.isdigit() for char in line) and any(word in line_lower for word in ['+91', 'phone', 'mobile', 'contact']):
            phone = line.strip()
        elif any(word in line_lower for word in ['mumbai', 'pune', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata']):
            location = line.strip()
        elif any(word in line_lower for word in ['years', 'year', 'experience']):
            experience_years = line.strip()
        elif len(line.strip()) > 3 and len(line.strip()) < 50 and not any(char.isdigit() for char in line) and not '@' in line:
            if name == "Unknown Candidate":
                name = line.strip()
    
    return {
        'name': name,
        'email': email,
        'phone': phone,
        'location': location,
        'experience_years': experience_years,
        'filename': filename,
        'resume_preview': resume_text[:300] + '...' if len(resume_text) > 300 else resume_text,
        'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'status': 'Available',
        'notes': ''
    }


def clean_text(text):
    text = text.lower()
    # Preserve important terms like AI, ML, R&D, API, etc.
    text = re.sub(r'\b(ai|ml|nlp|api|rd|git|github|aws|docker|flask|fastapi|tensorflow|pytorch|scikit-learn|numpy|pandas)\b', r' \1 ', text)
    # Remove only unnecessary punctuation but keep important separators
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills(resume_text):
    skills_list = [
        "python","java","javascript","html","css","sql",
        "machine learning","deep learning","nlp",
        "data science","flask","django","react","node",
        "tensorflow","pytorch","pandas","numpy",
        "docker","aws","git","github","api"
    ]

    text = resume_text.lower()

    detected_skills = []

    for skill in skills_list:
        if skill in text:
            detected_skills.append(skill)

    return detected_skills

def find_missing_skills(job_text, detected_skills):

    skill_database = [
        "python","java","javascript","html","css","sql",
        "machine learning","deep learning","nlp",
        "data science","flask","django","react","node",
        "tensorflow","pytorch","pandas","numpy",
        "docker","aws","git","github","api","rest"
    ]

    job_text = job_text.lower()

    required_skills = []

    for skill in skill_database:
        if skill in job_text:
            required_skills.append(skill)

    missing_skills = []

    for skill in required_skills:
        if skill not in detected_skills:
            missing_skills.append(skill)

    return missing_skills

def improved_similarity(job_text, resume_text):
    """Hybrid similarity calculation using multiple approaches for better accuracy"""
    
    # Clean texts while preserving important technical terms
    jd_clean = clean_text(job_text)
    resume_clean = clean_text(resume_text)
    
    # Handle empty texts
    if not jd_clean.strip() or not resume_clean.strip():
        return 0.0, jd_clean, resume_clean
    
    try:
        # Method 1: TF-IDF with better parameters
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased features
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            stop_words='english',
            min_df=1,
            max_df=0.9
        )
        
        combined_texts = [jd_clean, resume_clean]
        vectors = vectorizer.fit_transform(combined_texts)
        tfidf_similarity = cosine_similarity(vectors[0:1], vectors[1:2]).flatten()[0]
        
        # Method 2: Jaccard similarity for keyword overlap
        jd_words = set(jd_clean.split())
        resume_words = set(resume_clean.split())
        
        if len(jd_words) > 0 and len(resume_words) > 0:
            intersection = len(jd_words.intersection(resume_words))
            union = len(jd_words.union(resume_words))
            jaccard_similarity = intersection / union if union > 0 else 0
        else:
            jaccard_similarity = 0
        
        # Method 3: Length-adjusted similarity
        length_ratio = min(len(jd_clean), len(resume_clean)) / max(len(jd_clean), len(resume_clean))
        
        # Method 4: Technical term density
        technical_terms = {
            'html', 'css', 'javascript', 'python', 'java', 'react', 'angular', 'vue', 'node',
            'sql', 'mongodb', 'aws', 'docker', 'git', 'github', 'api', 'rest', 'json',
            'bootstrap', 'tailwind', 'sass', 'scss', 'typescript', 'php', 'mysql', 'postgresql'
        }
        
        jd_tech_terms = jd_words.intersection(technical_terms)
        resume_tech_terms = resume_words.intersection(technical_terms)
        
        if len(jd_tech_terms) > 0:
            tech_term_overlap = len(jd_tech_terms.intersection(resume_tech_terms)) / len(jd_tech_terms)
        else:
            tech_term_overlap = 0
        
        # Combine all methods with weights
        final_similarity = (
            tfidf_similarity * 0.4 +      # 40% weight to TF-IDF
            jaccard_similarity * 0.3 +    # 30% weight to keyword overlap
            length_ratio * 0.1 +          # 10% weight to length similarity
            tech_term_overlap * 0.2       # 20% weight to technical term overlap
        )
        
        # Ensure similarity is not NaN and is reasonable
        if final_similarity != final_similarity:  # Check for NaN
            final_similarity = 0.0
        
        return final_similarity, jd_clean, resume_clean
        
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0, jd_clean, resume_clean


def calculate_keyword_bonus(job_text, resume_text):
    """Calculate bonus points for exact keyword matches with improved scoring"""
    job_words = set(clean_text(job_text).split())
    resume_words = set(clean_text(resume_text).split())
    
    # Find common technical terms with weights
    technical_terms = {
        # High-value skills (5 points each)
        'python': 5, 'java': 5, 'javascript': 5, 'html': 5, 'css': 5, 'sql': 5, 
        'machine learning': 5, 'ai': 5, 'ml': 5, 'nlp': 5, 'deep learning': 5, 
        'data science': 5, 'web development': 5, 'frontend': 5, 'backend': 5,
        
        # Medium-value skills (3 points each)
        'tensorflow': 3, 'pytorch': 3, 'scikit-learn': 3, 'numpy': 3, 'pandas': 3,
        'flask': 3, 'fastapi': 3, 'django': 3, 'react': 3, 'angular': 3, 'vue': 3,
        'node.js': 3, 'express': 3, 'docker': 3, 'kubernetes': 3, 'aws': 3, 
        'azure': 3, 'git': 3, 'github': 3, 'bootstrap': 3, 'tailwind': 3,
        
        # Basic skills (2 points each)
        'api': 2, 'rest': 2, 'json': 2, 'xml': 2, 'database': 2, 'mysql': 2,
        'mongodb': 2, 'postgresql': 2, 'algorithms': 2, 'data structures': 2, 
        'oop': 2, 'version control': 2, 'responsive': 2, 'cross-browser': 2,
        'accessibility': 2, 'seo': 2, 'performance': 2, 'optimization': 2
    }
    
    # Calculate weighted bonus
    total_bonus = 0
    matched_terms = []
    
    for term, weight in technical_terms.items():
        if term in job_words and term in resume_words:
            total_bonus += weight
            matched_terms.append(term)
    
    # Add bonus for education level matches
    education_keywords = {'bachelor', 'master', 'phd', 'engineering', 'computer science', 'data science', 'information technology', 'web development'}
    education_matches = len(job_words.intersection(resume_words).intersection(education_keywords))
    total_bonus += education_matches * 3
    
    # Add bonus for experience level matches
    experience_keywords = {'fresher', 'freshers', 'entry level', 'junior', 'senior', 'lead', 'years experience'}
    experience_matches = len(job_words.intersection(resume_words).intersection(experience_keywords))
    total_bonus += experience_matches * 2
    
    # Cap the bonus at 40% to avoid inflating scores too much
    final_bonus = min(40, total_bonus)
    
    return final_bonus, matched_terms

@app.route("/")
def index():

    if current_user.is_authenticated:
        return render_template("index.html", user=current_user)

    return redirect("/login")

@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare_resumes():
    if request.method == 'POST':
        # Get form data
        job_title = request.form.get('job_title')
        job_desc = request.form.get('job_desc')
        skills = request.form.get('skills')
        education = request.form.get('education')
        experience = request.form.get('experience')
        
        # Combine everything into a job description text
        full_jd = f"{job_title} {job_desc} {skills} {education} {experience}"
        
        # Get all uploaded files
        files = request.files.getlist('resumes')
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Extract resume text
                resume_text = extract_text(filepath)
                detected_skills = extract_skills(resume_text)
                missing_skills = find_missing_skills(full_jd, detected_skills)
                
                # Generate candidate profile
                candidate_id = filename.replace('.pdf', '').replace('.docx', '').replace('.doc', '')

                profile = generate_candidate_profile(resume_text, filename)

                candidate = Candidate(
                    name=profile["name"],
                    email=profile["email"],
                    phone=profile["phone"],
                    location=profile["location"],
                    experience=profile["experience_years"],
                    resume_filename=filename,
                    status="New",
                    notes=""
                )

                db.session.add(candidate)
                db.session.commit()

                
                # Calculate similarity and bonus
                similarity, jd_clean, resume_clean = improved_similarity(full_jd, resume_text)
                score = round(similarity * 100, 2)
                keyword_bonus, matched_terms = calculate_keyword_bonus(full_jd, resume_text)
                final_score = min(100, score + keyword_bonus)
                
                # Add enhanced features for top 3 candidates
                rank_features = []
                if len(results) < 3:
                    rank_features.append('Top 3 Candidate')
                    if len(results) == 0:
                        rank_features.extend(['🏆 Gold Medal', 'Priority Contact', 'Fast Track Interview'])
                    elif len(results) == 1:
                        rank_features.extend(['🥈 Silver Medal', 'High Priority', 'Quick Response'])
                    elif len(results) == 2:
                        rank_features.extend(['🥉 Bronze Medal', 'Good Match', 'Standard Process'])
                
                results.append({
                    'filename': filename,
                    'candidate_id': candidate_id,
                    'similarity': similarity,
                    'base_score': score,
                    'keyword_bonus': keyword_bonus,
                    'final_score': final_score,
                    'matched_terms': matched_terms,
                    'resume_preview': resume_clean[:200] + '...' if len(resume_clean) > 200 else resume_clean,
                    'rank_features': rank_features,
                    "profile": candidate,
                    "skills": detected_skills,
                    "missing_skills": missing_skills
                })
        
        # Sort results by final score (highest first)
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Add ranking information AFTER sorting
        for i, result in enumerate(results):
            result['rank'] = i + 1
            result['rank_badge'] = ['🥇', '🥈', '🥉'][i] if i < 3 else f'#{i+1}'
            
            # Update rank features based on actual ranking
            if result['rank'] <= 3:
                result['rank_features'] = []
                result['rank_features'].append('Top 3 Candidate')
                if result['rank'] == 1:
                    result['rank_features'].extend(['🏆 Gold Medal', 'Priority Contact', 'Fast Track Interview'])
                elif result['rank'] == 2:
                    result['rank_features'].extend(['🥈 Silver Medal', 'High Priority', 'Quick Response'])
                elif result['rank'] == 3:
                    result['rank_features'].extend(['🥉 Bronze Medal', 'Good Match', 'Standard Process'])
        
        return render_template('compare_results.html', 
                             job_title=job_title,
                             results=results,
                             jd_clean=jd_clean[:500] + '...' if len(jd_clean) > 500 else jd_clean)
    
    return render_template('compare.html')

@app.route('/rank', methods=['POST'])
def rank():
    # Get form data
    job_title = request.form.get('job_title')
    job_desc = request.form.get('job_desc')
    skills = request.form.get('skills')
    education = request.form.get('education')
    experience = request.form.get('experience')

    # Combine everything into a job description text (simplified format)
    full_jd = f"{job_title} {job_desc} {skills} {education} {experience}"

    # Handle file upload
    file = request.files['resume']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract resume text
        resume_text = extract_text(filepath)

        # Use improved similarity calculation
        similarity, jd_clean, resume_clean = improved_similarity(full_jd, resume_text)
        
        # Calculate score with better scaling
        score = round((similarity * 80)+20, 2)
        
        # Add bonus for exact keyword matches
        keyword_bonus, matched_terms = calculate_keyword_bonus(full_jd, resume_text)
        final_score = min(100, score + keyword_bonus)

        return f"""
        <h2>Resume Match Result</h2>
        <p><strong>Job Title:</strong> {job_title}</p>
        <p><strong>Resume File:</strong> {filename}</p>
        <p><strong>Base Match Score:</strong> {score}%</p>
        <p><strong>Keyword Bonus:</strong> +{keyword_bonus}%</p>
        <p><strong>Final Score:</strong> {final_score}%</p>
        <hr>
        <h4>Matched Technical Terms:</h4>
        <p>{', '.join(matched_terms) if matched_terms else 'No technical terms matched'}</p>
        <hr>
        <h4>Debug Information:</h4>
        <p><strong>Job Description (cleaned):</strong></p>
        <pre style="background: #f5f5f5; padding: 10px; font-size: 12px;">{jd_clean[:500]}...</pre>
        <p><strong>Resume Text (cleaned):</strong></p>
        <pre style="background: #f5f5f5; padding: 10px; font-size: 12px;">{resume_clean[:500]}...</pre>
        <a href="/">Go Back</a>
        """
    else:
        flash('Invalid file type. Please upload a PDF or DOCX file.')
        return redirect(url_for('index'))

@app.route('/contact/<filename>', methods=['GET', 'POST'])
def contact_candidate(filename):
    """Contact a specific candidate"""
    if request.method == 'POST':
        action = request.form.get('action')
        candidate_id = request.form.get('candidate_id')
        
        if action == 'send_email':
            # Simulate email sending
            flash(f'Email sent to candidate {candidate_id}!', 'success')
        elif action == 'send_sms':
            # Simulate SMS sending
            flash(f'SMS sent to candidate {candidate_id}!', 'success')
        elif action == 'update_status':
            new_status = request.form.get('status')
            candidate = Candidate.query.filter_by(resume_filename=filename).first()
            if candidate:
                candidate.status = new_status
                db.session.commit()
            flash(f'Status updated to {new_status}!', 'success')
        elif action == 'add_notes':
            notes = request.form.get('notes')
            candidate = Candidate.query.filter_by(resume_filename=filename).first()
            if candidate:
                candidate.notes = notes
                db.session.commit()
            flash('Notes added successfully!', 'success')
        
        return redirect(url_for('contact_candidate', filename=filename))
    
    # Get candidate profile
    candidate_id = filename.replace('.pdf', '').replace('.docx', '').replace('.doc', '')
    profile = Candidate.query.filter_by(resume_filename=filename).first()
    
    return render_template('contact_candidate.html', profile=profile, filename=filename)

@app.route('/api/send_contact', methods=['POST'])
def send_contact():
    """API endpoint for sending contact messages"""
    data = request.get_json()
    candidate_id = data.get('candidate_id')
    message_type = data.get('type')  # 'email' or 'sms'
    message = data.get('message')
    
    # Simulate sending (in production, integrate with email/SMS services)
    if message_type == 'email':
        # Here you would integrate with Gmail, SendGrid, etc.
        response = {'success': True, 'message': f'Email sent to {candidate_id}'}
    elif message_type == 'sms':
        # Here you would integrate with Twilio, AWS SNS, etc.
        response = {'success': True, 'message': f'SMS sent to {candidate_id}'}
    else:
        response = {'success': False, 'message': 'Invalid message type'}
    
    return jsonify(response)

@app.route('/candidates')
def candidate_dashboard():
    """Dashboard to view all candidates and their statuses"""

    candidates = Candidate.query.all()

    total = len(candidates)
    new = len([c for c in candidates if c.status == 'New'])
    contacted = len([c for c in candidates if c.status == 'Contacted'])
    shortlisted = len([c for c in candidates if c.status == 'Shortlisted'])
    interviews = len([c for c in candidates if c.status == 'Interview Scheduled'])
        
    return render_template(
        'candidate_dashboard.html',
        candidates=candidates,
        total=total,
        new=new,
        contacted=contacted,
        shortlisted=shortlisted,
        interviews=interviews
    )

@app.route('/candidate/<candidate_id>')
def view_candidate(candidate_id):

    candidate = Candidate.query.filter_by(resume_filename=candidate_id + ".docx").first()

    if not candidate:
        flash("Candidate not found")
        return redirect(url_for("candidate_dashboard"))

    return render_template("view_candidate.html", profile=candidate)

@app.route('/api/schedule_interview', methods=['POST'])
def schedule_interview():

    candidate_id = request.form.get("candidate_id")

    candidate = Candidate.query.filter_by(resume_filename=candidate_id + ".docx").first()

    if candidate:
        candidate.status = "Interview Scheduled"
        db.session.commit()

        return jsonify({"success": True})

    return jsonify({"success": False, "message": "Candidate not found"})

@app.route('/api/add_to_shortlist', methods=['POST'])
def add_to_shortlist():

    candidate_id = request.form.get('candidate_id')

    candidate = Candidate.query.filter_by(resume_filename=candidate_id + ".docx").first()

    if candidate:
        candidate.status = "Shortlisted"
        db.session.commit()

        return jsonify({"success": True})

    return jsonify({"success": False, "message": "Candidate not found"})

@app.route('/api/reject_candidate', methods=['POST'])
def reject_candidate():

    candidate_id = request.form.get("candidate_id")

    candidate = Candidate.query.filter_by(resume_filename=candidate_id + ".docx").first()

    if candidate:
        candidate.status = "Rejected"
        db.session.commit()

        return jsonify({"success": True})

    return jsonify({"success": False, "message": "Candidate not found"})

@app.route('/api/edit_candidate/<candidate_id>', methods=['POST'])
def edit_candidate(candidate_id):
    """Edit candidate information"""
    candidate = Candidate.query.filter_by(resume_filename=candidate_id + ".docx").first()

    if not candidate:
        return jsonify({"success": False, "message": "Candidate not found"})
        
    data = request.form
    
    # Update candidate profile
    candidate.name = data.get('name', '')
    candidate.email = data.get('email', '')
    candidate.phone = data.get('phone', '')
    candidate.location = data.get('location', '')
    candidate.experience = data.get('experience', '')
    candidate.notes = data.get('notes', '')
    candidate.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Candidate updated successfully'})

@app.route('/api/delete_candidate/<candidate_id>', methods=['DELETE'])
def delete_candidate(candidate_id):

    candidate = Candidate.query.filter_by(resume_filename=candidate_id + ".docx").first()

    if candidate:
        db.session.delete(candidate)
        db.session.commit()

        return jsonify({"success": True})

    return jsonify({"success": False, "message": "Candidate not found"})
    
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            login_user(user)

            return redirect("/")

        flash("Invalid email or password")

    return render_template("login.html")

@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        company = request.form["company"]
        password = generate_password_hash(request.form["password"])

        user = User(
            name=name,
            email=email,
            company=company,
            password=password
        )

        db.session.add(user)
        db.session.commit()

        flash("Account created successfully")

        return redirect("/login")

    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect("/login")



if __name__ == '__main__':
    app.run(debug=True)

