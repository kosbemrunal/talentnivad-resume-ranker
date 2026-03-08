# TalentNivad – AI Resume Ranker

TalentNivad is an AI-powered resume screening platform that helps recruiters quickly identify the most relevant candidates for a job role. The system analyzes resumes and ranks them based on similarity with the job description using Natural Language Processing (NLP).

## 🚀 Live Demo

https://talentnivad.onrender.com

## 📌 Features

* 🔐 User authentication (Register / Login)
* 📄 Upload resumes (PDF / DOCX)
* 🤖 AI-based resume ranking using NLP
* 📊 Candidate dashboard
* 📑 Resume comparison tool
* 🗂 Candidate management system
* 🌐 Deployed online with Render

## 🧠 How It Works

1. Recruiter uploads resumes.
2. A job description is provided.
3. The system converts text into numerical vectors using **TF-IDF Vectorization**.
4. **Cosine Similarity** is calculated between the job description and each resume.
5. Candidates are ranked based on similarity score.

## 🛠 Tech Stack

**Backend**

* Python
* Flask
* SQLAlchemy
* Flask-Login

**Machine Learning**

* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

**Frontend**

* HTML
* CSS
* Bootstrap
* JavaScript

**Deployment**

* Gunicorn
* Render

## 📂 Project Structure

```
resume_ranker/
│
├── app.py
├── requirements.txt
├── Procfile
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── candidate_dashboard.html
│   └── compare_results.html
│
├── uploads/
├── resumes/
└── static/
```

## ⚙️ Installation (Run Locally)

Clone the repository

```
git clone https://github.com/kosbemrunal/talentnivad-resume-ranker.git
cd talentnivad-resume-ranker
```

Create virtual environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
python app.py
```

Open in browser

```
http://127.0.0.1:5000
```

## 📸 Screenshots
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/01871051-792d-4ff4-9210-f930c04dc0e0" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/40faaf3c-9e15-4e52-acce-97b9cc4ffbfd" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/81051c87-5763-4f17-b020-6eab3d7a9971" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/056e5629-3071-4606-a736-6c32a51f5fed" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/d7075d40-b09d-4e59-9eea-a09fe746d8c8" />


## 📈 Future Improvements

* AI resume feedback system
* Job description upload feature
* Better UI/UX design
* Recruiter analytics dashboard
* Resume keyword highlighting

## 👨‍💻 Author

**Mrunal Kosbe**

GitHub: https://github.com/kosbemrunal

---

⭐ If you like this project, consider giving it a star!
