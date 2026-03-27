from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from flask_cors import CORS
from email_validator import validate_email, EmailNotValidError
import MySQLdb.cursors
import hashlib
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask_mail import Mail, Message
import os
import base64
from itsdangerous import URLSafeTimedSerializer
import random
import re

# ---------------------------------
# APP INITIALIZATION
# ---------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'this_is_secret_key_change_it'
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
CORS(app)

# Email validation helper (real-world valid emails beyond regex)
def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

# Load chatbot model
chat_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully")
chat_embeddings = joblib.load("embeddings.pkl")
chat_data = joblib.load("chat_data.pkl")

# ---------------------------------
# EMAIL CONFIGURATION (GMAIL)
# ---------------------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

app.config['MAIL_USERNAME'] = 'sanjaykuruvella112@gmail.com'
app.config['MAIL_PASSWORD'] = 'uugvcphgexqelwkz'
app.config['MAIL_DEFAULT_SENDER'] = 'sanjaykuruvella112@gmail.com'

mail = Mail(app)

# ---------------------------------
# APP BASE URL (For Reset Links)
# ---------------------------------
app.config['BASE_URL'] = 'http://192.168.168.162:5000'


# ---------------------------------
# DATABASE CONFIG (XAMPP)
# ---------------------------------
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'futureai'
mysql = MySQL(app)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
model = None
try:
    model = joblib.load("future_ai_simulator.pkl")
except FileNotFoundError:
    print("Model file not found. Please train the model using /retrain_model endpoint.")


def detect_message_topic(message):
    lower_msg = (message or "").lower()

    topic_keywords = {
        "career": ["career", "job", "work", "office", "engineer", "manager", "promotion", "interview"],
        "finance": ["money", "finance", "investment", "rich", "salary", "pay", "wealth", "budget", "income"],
        "education": ["education", "study", "degree", "college", "school", "learn", "course", "skill", "exam"],
        "health": ["health", "fitness", "diet", "exercise", "sleep", "stress"],
        "prediction": ["predict", "future", "outcome", "happen", "forecast", "result", "simulation"]
    }

    for topic, keywords in topic_keywords.items():
        if any(word in lower_msg for word in keywords):
            return topic

    return "general"


def summarize_user_message(message, max_words=18):
    cleaned = re.sub(r"\s+", " ", (message or "")).strip()
    if not cleaned:
        return "your request"

    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned

    return " ".join(words[:max_words]) + "..."


def build_clear_chat_reply(message, role="General", context_data=None, matched_response=None):
    context_data = context_data or {}
    clean_message = (message or "").strip()

    if not clean_message:
        return "Please enter a clear message so I can respond based on what you need."

    topic = detect_message_topic(clean_message)
    summary = summarize_user_message(clean_message)
    lower_msg = clean_message.lower()
    is_question = "?" in clean_message or any(
        lower_msg.startswith(prefix) for prefix in ["what", "why", "how", "when", "where", "can", "should", "is", "do"]
    )

    if any(word in lower_msg for word in ["hello", "hi", "hey"]):
        return (
            f"Hello. I understand you want help with: \"{summary}\". "
            "Tell me your exact question or goal, and I will give you a focused answer."
        )

    role_openers = {
        "Entrepreneur": "Based on your business-focused message",
        "Student": "Based on your learning-focused message",
        "Doctor": "Based on your professional message"
    }
    opener = role_openers.get(role, "Based on your message")

    if topic == "prediction":
        if context_data.get("prob") is not None:
            return (
                f"{opener}, you are asking about your future outcome: \"{summary}\". "
                f"Your latest prediction shows {context_data.get('prob')}% success probability, "
                f"{context_data.get('sat', 'N/A')} life satisfaction, and "
                f"{context_data.get('impact', 'N/A')} financial impact over {context_data.get('timeline', 'the current timeline')}. "
                "If you want, I can next explain what this means for your career, money, or decision quality."
            )
        return (
            f"{opener}, you are asking about a future result: \"{summary}\". "
            "I need a saved prediction or simulation to answer this accurately. "
            "Run a prediction first, and then I can explain the result clearly."
        )

    if topic == "career":
        return (
            f"{opener}, your message is mainly about career growth: \"{summary}\". "
            "A practical next step is to define one target role, identify the missing skills, and build a short weekly action plan. "
            "If you share your current role and the role you want, I can give a more exact answer."
        )

    if topic == "finance":
        finance_context = ""
        if context_data.get("impact") is not None:
            finance_context = f" Your latest financial impact is {context_data.get('impact')}."
        return (
            f"{opener}, your message is about money or financial planning: \"{summary}\"."
            f"{finance_context} A useful next step is to review your income, risk level, and savings goal before choosing a strategy. "
            "If you want, I can help break this into a simple plan."
        )

    if topic == "education":
        return (
            f"{opener}, your message is about learning or education: \"{summary}\". "
            "The best next step is to choose one skill or course outcome, then study it in small weekly milestones. "
            "If you tell me what you want to learn, I can suggest a clearer path."
        )

    if topic == "health":
        return (
            f"{opener}, your message is about health or personal routine: \"{summary}\". "
            "A clear next step is to focus on one habit at a time and track it consistently. "
            "If you share your goal, I can help organize it into a simple routine."
        )

    if matched_response:
        return (
            f"{opener}, I understood your message as: \"{summary}\". "
            f"{matched_response}"
        )

    if is_question:
        return (
            f"{opener}, I understand your question: \"{summary}\". "
            "Please share a little more detail so I can answer it more precisely and avoid giving a generic reply."
        )

    return (
        f"{opener}, I understood your message as: \"{summary}\". "
        "Please tell me the exact problem, goal, or decision you want help with, and I will respond based on that."
    )

# ---------------------------------
# HOME ROUTE
# ---------------------------------
@app.route('/')
def home():
    return "FutureAI Backend Running Successfully"

@app.route('/db_status')
def db_status():
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        return jsonify({"status": "Connected", "database": app.config['MYSQL_DB']})
    except Exception as e:
        return jsonify({"status": "Error", "error": str(e)}), 500

@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()

        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not email:
            return jsonify({"error": "Email is required"}), 400

        if not is_valid_email(email):
            return jsonify({"error": "Invalid email format"}), 400

        cursor = mysql.connection.cursor()

        # check email exists
        cursor.execute("SELECT user_id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            cursor.close()
            return jsonify({"message": "Email already exists"}), 400

        # insert user
        cursor.execute(
            "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
            (name, email, password)
        )

        mysql.connection.commit()   # ⭐ VERY IMPORTANT

        user_id = cursor.lastrowid

        cursor.close()

        return jsonify({
            "status": True,
            "message": "Registration Successful",
            "user": {
                "user_id": user_id,
                "name": name,
                "email": email
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        email = data.get("email")
        password = data.get("password")

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        cursor.execute(
            "SELECT user_id,name,email,password FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()
        cursor.close()

        if not user:
            return jsonify({
                "status": False,
                "message": "Email not found"
            }), 401

        if user["password"] != password:
            return jsonify({
                "status": False,
                "message": "password is invalid"
            }), 401

        # Clear previous chat history for a fresh login session
        delete_cursor = mysql.connection.cursor()
        delete_cursor.execute("DELETE FROM chat_messages WHERE user_id=%s", (user["user_id"],))
        mysql.connection.commit()
        delete_cursor.close()

        user.pop("password")

        return jsonify({
            "status": True,
            "message": "Login Successful",
            "user": user
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# UPDATE PROFILE
# ---------------------------------
@app.route('/update_profile/<int:user_id>', methods=['PUT'])
def update_profile(user_id):
    try:
        data = request.json

        name = data.get('name')
        email = data.get('email')
        profile_photo = data.get('profile_photo')

        cursor = mysql.connection.cursor()

        cursor.execute("""
            UPDATE users
            SET name=%s,email=%s,profile_photo=%s
            WHERE user_id=%s
        """, (name, email, profile_photo, user_id))

        mysql.connection.commit()
        cursor.close()

        return jsonify({"status": True, "message": "Profile Updated Successfully"})
    except Exception as e:
        print("UPDATE PROFILE ERROR:", str(e))
        return jsonify({"status": False, "error": str(e)}), 500


@app.route('/update-email', methods=['PUT'])
def update_email():
    data = request.get_json() or {}
    email = data.get('email')
    new_email = data.get('new_email')

    if not email or not new_email:
        return jsonify({"error": "Email and new_email are required"}), 400

    if not is_valid_email(email) or not is_valid_email(new_email):
        return jsonify({"error": "Invalid email format"}), 400

    cursor = mysql.connection.cursor()
    cursor.execute("UPDATE users SET email=%s WHERE email=%s", (new_email, email))
    mysql.connection.commit()
    cursor.close()

    return jsonify({"message": "Email updated successfully"}), 200


# ---------------------------------
# UPDATE PROFILE PHOTO
# ---------------------------------
@app.route('/update-profile-photo', methods=['POST'])
def update_profile_photo():
    data = request.get_json()

    email = data.get("email")
    profile_photo = data.get("profile_photo")

    cur = mysql.connection.cursor()

    cur.execute(
        "UPDATE users SET profile_photo=%s WHERE email=%s",
        (profile_photo, email)
    )

    mysql.connection.commit()
    cur.close()

    return jsonify({"status": "success"})


# ---------------------------------
# UPLOAD PROFILE PHOTO
# ---------------------------------
@app.route('/upload-profile-photo', methods=['POST'])
def upload_profile_photo():
    try:
        data = request.json
        email = data.get("email")
        profile_photo = data.get("profile_photo")

        cursor = mysql.connection.cursor()

        query = "UPDATE users SET profile_photo=%s WHERE email=%s"
        cursor.execute(query, (profile_photo, email))
        mysql.connection.commit()
        cursor.close()

        return jsonify({
            "status": "success",
            "message": "Profile photo updated successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# GET PROFILE PHOTO
# ---------------------------------
@app.route('/get-profile-photo/<email>', methods=['GET'])
def get_profile_photo(email):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        query = "SELECT name,email,profile_photo FROM users WHERE email=%s"
        cursor.execute(query, (email,))
        user = cursor.fetchone()

        if user:
            return jsonify({
                "status": "success",
                "name": user["name"],
                "email": user["email"],
                "profile_photo": user["profile_photo"]
            })
        else:
            return jsonify({"message": "User not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# GET PROFILE
# ---------------------------------
@app.route('/get-profile/<email>', methods=['GET'])
def get_profile(email):

    cur = mysql.connection.cursor()

    cur.execute(
        "SELECT name,email,profile_photo FROM users WHERE email=%s",
        (email,)
    )

    user = cur.fetchone()
    cur.close()

    return jsonify(user)


# ---------------------------------
# FORGOT PASSWORD
# ---------------------------------
@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json() or {}

        email = data.get("email")
        new_password = data.get("new_password")

        if not email or not new_password:
            return jsonify({"error": "Email and new_password are required"}), 400

        if not is_valid_email(email):
            return jsonify({"error": "Invalid email format"}), 400

        cursor = mysql.connection.cursor()

        cursor.execute(
            "UPDATE users SET password=%s WHERE email=%s",
            (new_password, email)
        )

        mysql.connection.commit()
        cursor.close()

        return jsonify({"message": "Password Updated Successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/forgot-password', methods=['POST'])
def forgot_password_api():
    data = request.get_json() or {}
    email = data.get('email')

    if not email:
        return jsonify({"error": "Email is required"}), 400

    if not is_valid_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    # TODO: send password reset link via email, or integrate with existing forgot_password_email logic
    return jsonify({"message": "Password reset link sent"}), 200


@app.route('/reset-password', methods=['POST'])
def reset_password_api():
    data = request.get_json() or {}
    email = data.get('email')
    new_password = data.get('password')

    if not email or not new_password:
        return jsonify({"error": "Email and password required"}), 400

    if not is_valid_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    cursor = mysql.connection.cursor()
    cursor.execute("UPDATE users SET password=%s WHERE email=%s", (new_password, email))
    mysql.connection.commit()
    cursor.close()

    return jsonify({"message": "Password updated successfully"}), 200


# ---------------------------------
# SEND RESET LINK TO EMAIL
# ---------------------------------
@app.route('/forgot_password_email', methods=['POST'])
def forgot_password_email():
    try:
        data = request.get_json()
        email = data.get("email")

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT name FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if not user:
            return jsonify({"status": False, "message": "Email not found"}), 404

        # Generate secure token
        token = serializer.dumps(email, salt='reset-password')

        # 🔗 Reset link (uses config BASE_URL)
        base_url = app.config.get('BASE_URL', 'http://192.168.168.162:5000')
        reset_link = f"{base_url}/reset_password/{token}"

        msg = Message(
            subject="Reset Your Password",
            recipients=[email]
        )

        msg.body = f"""
Hello {user['name']},

Click the link below to reset your password:

{reset_link}

This link will expire in 10 minutes.

FutureAI Team
"""

        mail.send(msg)

        return jsonify({
            "status": True,
            "message": "Reset link sent to your email"
        })

    except Exception as e:
        return jsonify({"status": False, "error": str(e)}), 500


# ---------------------------------
# RESET PASSWORD PAGE + UPDATE
# ---------------------------------
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        if request.method == 'GET':
            return f"""
            <html>
            <body style="text-align:center;margin-top:100px;font-family:sans-serif;">
                <h2>Reset Your Password</h2>
                <form method="POST">
                    <input type="password" name="new_password" placeholder="Enter new password" required
                    style="padding:10px;width:250px;"><br><br>
                    <button type="submit" style="padding:10px 20px;">Reset Password</button>
                </form>
            </body>
            </html>
            """

        # POST → update password
        new_password = request.form.get("new_password")

        email = serializer.loads(token, salt='reset-password', max_age=600)

        cursor = mysql.connection.cursor()
        cursor.execute(
            "UPDATE users SET password=%s WHERE email=%s",
            (new_password, email)
        )
        mysql.connection.commit()
        cursor.close()

        return "<h3>Password updated successfully ✅</h3>"

    except Exception as e:
        return "<h3>Invalid or expired link ❌</h3>"





# ---------------------------------
# SAVE PREDICTION
# ---------------------------------
@app.route('/prediction', methods=['POST'])
def save_prediction():

    data = request.json

    cursor = mysql.connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (user_id,input_data,forecast_result,risk_level)
        VALUES(%s,%s,%s,%s)
    """, (
        data['user_id'],
        data['input_data'],
        data['forecast_result'],
        data['risk_level']
    ))

    mysql.connection.commit()
    cursor.close()
    return jsonify({"message": "Prediction Saved"})


# ---------------------------------
# FUTURE PREDICTION ROUTE
# ---------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    category = data.get("category")
    decision = data.get("decision")   # user decision from frontend

    # AI Better Suggestions
    if category == "career":
        better_option = "Upgrade skills in AI / Data Science and switch to a higher paying role"

    elif category == "finance":
        better_option = "Invest in diversified mutual funds with long term SIP strategy"

    elif category == "education":
        better_option = "Pursue higher education with industry certifications"

    else:
        better_option = "Improve skills and explore better opportunities"

    result = {
        "category": category,
        "success_probability": 85,
        "timeline": "6-12 months",
        "financial_impact": "+₹30K potential growth",
        "life_satisfaction": "+25% improvement",

        "scenarios": [

            # OPTION A → AI Suggested Better Option
            {
                "option": "A",
                "title": "AI Suggested Better Choice",
                "decision": better_option,
                "score": 90
            },

            # OPTION B → User Selected Option
            {
                "option": "B",
                "title": "Your Selected Decision",
                "decision": decision,
                "score": 70
            }

        ]
    }

    return jsonify(result)


# ---------------------------------
# SAVE SIMULATION
# ---------------------------------
@app.route('/save-simulation', methods=['POST'])
def save_simulation():
    try:
        data = request.json

        user_id = data.get("user_id")
        role = data.get("role")
        decision = data.get("decision")
        success_probability = data.get("success_probability", 50)

        cursor = mysql.connection.cursor()

        # safe conversion (handles "57.35" issue)
        try:
            sp = float(success_probability)
        except:
            sp = 50.0

        sql = """
        INSERT INTO simulations (user_id, role, decision, success_probability)
        VALUES (%s, %s, %s, %s)
        """

        cursor.execute(sql, (user_id, role, decision, sp))
        mysql.connection.commit()
        cursor.close()

        return jsonify({
            "status": "success",
            "message": "Simulation saved"
        })

    except Exception as e:
        if 'cursor' in locals() and cursor: cursor.close()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------------
# GET LATEST SIMULATION
# ---------------------------------
@app.route('/get-latest/<int:user_id>', methods=['GET'])
def get_latest(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        sql = """
        SELECT * FROM simulations
        WHERE user_id=%s
        ORDER BY id DESC
        LIMIT 1
        """

        cursor.execute(sql, (user_id,))
        result = cursor.fetchone()
        cursor.close()

        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------------
# GET TIMELINE (Uses Latest Simulation)
# ---------------------------------
@app.route('/get-timeline/<int:user_id>', methods=['GET'])
def get_timeline_new(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        sql = """
        SELECT success_probability FROM simulations
        WHERE user_id=%s
        ORDER BY id DESC
        LIMIT 1
        """

        cursor.execute(sql, (user_id,))
        last = cursor.fetchone()
        cursor.close()

        # fix decimal issue safely
        try:
            base_score = int(float(last.get('success_probability'))) if last else 50
        except:
            base_score = 50

        timeline = []

        for i in range(1, 7):
            timeline.append({
                "month": i,
                "probability": min(base_score + i * 4, 95)
            })

        return jsonify({
            "status": "success",
            "timeline": timeline
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ---------------------------------
# GET PREDICTION HISTORY (handled below)
# ---------------------------------


# ---------------------------------
# ADD JOURNAL
# ---------------------------------
@app.route('/journal', methods=['POST'])
def add_journal():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        user_id = data.get('user_id')
        entry_text = data.get('entry_text')
        mood = data.get('mood')

        if not user_id or not entry_text:
            return jsonify({"error": "Missing required fields"}), 400

        cursor = mysql.connection.cursor()

        cursor.execute(
            "INSERT INTO journal(user_id,entry_text,mood) VALUES(%s,%s,%s)",
            (user_id, entry_text, mood)
        )

        mysql.connection.commit()
        cursor.close()

        return jsonify({"message": "Journal Saved", "status": True}), 201
    except Exception as e:
        if 'cursor' in locals() and cursor: cursor.close()
        return jsonify({"error": str(e), "status": False}), 500


# ---------------------------------
# GET JOURNALS
# ---------------------------------
@app.route('/journal/<int:user_id>', methods=['GET'])
def get_journal(user_id):

    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute(
        "SELECT * FROM journal WHERE user_id=%s ORDER BY created_at DESC",
        (user_id,)
    )

    journals = cursor.fetchall()
    cursor.close()

    return jsonify(journals)


# ---------------------------------
# UPDATE JOURNAL
# ---------------------------------
@app.route('/journal/<int:journal_id>', methods=['PUT'])
def update_journal(journal_id):

    data = request.json

    cursor = mysql.connection.cursor()

    cursor.execute("""
        UPDATE journal
        SET entry_text=%s,mood=%s
        WHERE journal_id=%s
    """, (data['entry_text'], data['mood'], journal_id))

    mysql.connection.commit()
    cursor.close()

    return jsonify({"message": "Journal Updated"})


# ---------------------------------
# DELETE JOURNAL
# ---------------------------------
@app.route('/journal/<int:journal_id>', methods=['DELETE'])
def delete_journal(journal_id):
    try:
        cursor = mysql.connection.cursor()

        cursor.execute(
            "DELETE FROM journal WHERE journal_id=%s",
            (journal_id,)
        )

        mysql.connection.commit()
        cursor.close()

        return jsonify({"message": "Journal Deleted Successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_future', methods=['POST'])
def predict_future():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Features from Adult Dataset for simulation
        required_fields = ["Age", "Workclass", "Education", "Education_Number", "Marital_Status", 
                           "Occupation", "Relationship", "Race", "Gender", "Capital_Gain", 
                           "Capital_Loss", "Hours_Per_Week", "Country"]
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        user_id = data.get("user_id", 1)
        
        # Prepare for model (exclude target 'Income')
        features = [[
            data['Age'], data['Workclass'],
            data['Education'], data['Education_Number'],
            data['Marital_Status'], data['Occupation'],
            data['Relationship'], data['Race'], data['Gender'],
            data['Capital_Gain'], data['Capital_Loss'],
            data['Hours_Per_Week'], data['Country']
        ]]

        prediction_val = 0
        model_prob = 0.5
        using_model = False
        if model:
            try:
                # Expecting a classifier (0 or 1)
                pred_result = model.predict(features)[0]
                prediction_val = int(pred_result)
                
                # Fetch exact probability if available
                if hasattr(model, "predict_proba"):
                    model_prob = model.predict_proba(features)[0][1] # Probability of class 1 (Income > 50K)
                else:
                    model_prob = 0.8 if prediction_val == 1 else 0.3
                using_model = True
            except Exception as e:
                print("Model Predict Error:", e)
                # Fallback to heuristic if model error (e.g. multi-output expectation)
                if data['Education_Number'] > 12 or data['Capital_Gain'] > 5000:
                    prediction_val = 1
        else:
            # Enhanced Heuristic
            score = 0
            if data['Education_Number'] > 12: score += 40
            if data['Capital_Gain'] > 2000: score += 30
            if data['Hours_Per_Week'] > 35: score += 20
            if data['Age'] > 22 and data['Age'] < 50: score += 10
            
            if score >= 60:
                prediction_val = 1

        # Map to Life Simulation output values the app expects
        if prediction_val == 1:
            base_prob = 75.0 if not using_model else (model_prob * 100.0)
            base_prob = max(base_prob, 60.0)
            fin_impact = 70.0 + (min(data['Capital_Gain'], 10000) / 200.0)
            life_sat = 80.0 + (data['Education_Number'] * 1.5)
            timeline = "3-9 months" if data['Hours_Per_Week'] > 30 else "9-15 months"
            alt_scenario = 20.0 - (data['Capital_Gain'] / 1000.0)
            future_comp = 85.0
        else:
            base_prob = 35.0 if not using_model else (model_prob * 100.0)
            base_prob = min(base_prob, 55.0)
            fin_impact = 30.0 + (min(data['Capital_Gain'], 5000) / 200.0)
            life_sat = 40.0 + (data['Education_Number'] * 2.0)
            timeline = "18-24 months"
            alt_scenario = 60.0 + (data['Capital_Loss'] / 500.0)
            future_comp = 40.0

        # Extract adjustment variables (0-100 values)
        risk = data.get("risk", 50)
        timeframe = data.get("timeframe", 50)
        effort = data.get("effort", 50)
        investment = data.get("investment", 50)

        # Success probability calculation with sliders impact
        success_prob = round(base_prob + (data['Education_Number'] * 2) + ((effort - 50) * 0.4) + (np.random.random() * 5), 2)
        success_prob = min(max(success_prob, 5.0), 99.0)

        fin_impact = fin_impact + ((investment - 50) * 0.8) - ((risk - 50) * 0.2)
        life_sat = life_sat + ((effort - 50) * 0.5) - ((timeframe - 50) * 0.3)
        future_comp = future_comp + ((investment - 50) * 0.4) + ((effort - 50) * 0.4)
        
        fin_impact = min(max(fin_impact, 5.0), 100.0)
        life_sat = min(max(life_sat, 5.0), 100.0)
        future_comp = min(max(future_comp, 5.0), 100.0)
        alt_scenario = min(max(alt_scenario + ((risk - 50) * 0.6), 5.0), 100.0)

        # Save to DB
        cursor = mysql.connection.cursor()
        try:
            category = data.get("category", "Career")
            decision_val = f"{category} Simulation (Risk:{risk}, Effort:{effort})"
            cursor.execute("""
                INSERT INTO predictions(
                    user_id, decision_input, success_probability, timeline, 
                    financial_impact, life_satisfaction, alternative_scenario, future_comparison
                )
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                user_id, decision_val, success_prob, timeline,
                fin_impact, life_sat, alt_scenario, future_comp
            ))
            


            mysql.connection.commit()
            cursor.close()
        except Exception as db_err:
            print("DB SAVE ERROR:", db_err)
            cursor.close()
            return jsonify({"error": "Failed to save to database", "details": str(db_err)}), 500

        return jsonify({
            "message": "Prediction Success",
            "success_probability": success_prob,
            "timeline": timeline,
            "financial_impact": round(fin_impact, 2),
            "life_satisfaction": round(life_sat, 2),
            "alternative_scenario": round(max(alt_scenario, 0), 2),
            "future_comparison": round(future_comp, 2)
        }), 200

    except Exception as e:
        print("PREDICT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/predictions/<int:user_id>', methods=['GET'])
def get_predictions_history(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            "SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC",
            (user_id,)
        )
        predictions = cursor.fetchall()
        cursor.close()
        return jsonify(list(predictions))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat_assistant', methods=['POST'])
def chat_assistant():
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id", 1)
        original_message = (data.get("message") or "").strip()

        if not original_message:
            return jsonify({"reply": "Please enter a message so I can respond clearly."}), 400

        user_msg = original_message.lower()

        # Try to get user's latest prediction to contextualize response
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 1", (user_id,))
        last_prediction = cursor.fetchone()
        cursor.close()

        prediction_context = ""
        context_data = {}
        if last_prediction:
            context_data = {
                "prob": last_prediction['success_probability'],
                "sat": last_prediction['life_satisfaction'],
                "impact": last_prediction['financial_impact'],
                "timeline": last_prediction['timeline'],
                "comp": last_prediction['future_comparison']
            }
            prediction_context = f" With a {context_data['prob']}% success probability and a {context_data['timeline']} timeline,"
        reply = None
        best_score = 0
        best_idx = -1
        matched_response = None
        try:
            user_vec = chat_model.encode([user_msg])
            scores = cosine_similarity(user_vec, chat_embeddings)[0]
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            if best_score > 0.4:
                matched_response = str(chat_data.iloc[best_idx]["response"]).strip()
        except Exception as model_err:
            print("Semantic Search Model Error:", model_err)

        # High confidence match from trained dataset triggers response
        if best_score > 0.65:
            reply = chat_data.iloc[best_idx]["response"]
        
        else:
            replies = []
            if any(w in user_msg for w in ["predict", "future", "outcome", "happen"]):
                if last_prediction:
                    replies = [
                        f"My analysis shows {context_data['prob']}% probability of success.{prediction_context} things look optimistic if you maintain your current effort.",
                        f"The future looks dynamic!{prediction_context} Your life satisfaction score of {context_data['sat']} suggests this is a high-value path for you.",
                        f"Based on the AI model, your financial impact ({context_data['impact']}) is a strong driver. How do you feel about the {context_data['timeline']} timeline?"
                    ]
                else:
                    replies = ["I need more data! Please use the Future Predictor so I can analyze your specific situation.", "Run a simulation first, and I'll be able to tell you exactly how your future looks!"]
            
            elif any(w in user_msg for w in ["career", "job", "work", "office", "engineer", "manager"]):
                replies = [
                    f"Career growth is looking solid.{prediction_context} I recommend investing in new technical skills to push that satisfaction score higher than {context_data.get('sat', 'current levels')}.",
                    f"Your work profile shows potential for a {context_data.get('impact', 'significant')} financial impact. Are you satisfied with a {context_data.get('timeline', '9-12 month')} growth period?",
                    "The current job market favors specialists. Have you considered how increasing your weekly hours might change your outcome?",
                    f"Based on your data, exploring leadership roles or lateral moves could significantly boost your future comparison score of {context_data.get('comp', 'solid levels')}.",
                    "Networking is key. Connecting with mentors in your desired field can reduce the expected timeline and improve your overall success probability."
                ]
            
            elif any(w in user_msg for w in ["money", "finance", "investment", "rich", "salary", "pay", "wealth"]):
                replies = [
                    f"Financially, this path yields a score of {context_data.get('impact', '70')}.{prediction_context} Small adjustments in your risk levels could shift this drastically.",
                    f"Wealth building takes time. Your projected future comparison is {context_data.get('comp', '80')} points above baseline! That's impressive.",
                    "To maximize your financial impact, we should look at balancing your capital gains vs losses in the simulator.",
                    "Have you considered diversifying your portfolio? A balanced mix of conservative and aggressive assets often yields the best long-term stability.",
                    f"Your current trajectory suggests a {context_data.get('timeline', 'positive')} timeline for reaching significant financial milestones. Stay consistent with your savings!"
                ]

            elif any(w in user_msg for w in ["education", "study", "degree", "college", "school", "learn", "course", "skill"]):
                replies = [
                    f"Furthering your education almost always correlates with higher life satisfaction.{prediction_context} Are you considering a specific certification?",
                    "Taking on a new course could delay immediate earnings, but drastically increase your long-term success probability. It's a trade-off worth simulating.",
                    f"With a timeline of {context_data.get('timeline', '1-2 years')}, integrating part-time studies into your schedule is highly feasible.",
                    "Upskilling is the best way to future-proof your career. Online certifications are heavily valued in today's tech-driven market.",
                    "I recommend focusing on specialized skills rather than general degrees. It usually provides a much faster and noticeable financial impact."
                ]

            elif any(w in user_msg for w in ["right", "ethic", "rule", "law", "policy", "integrity"]):
                replies = [
                    "AI ethics is our priority. FutureAI is designed flat and bias-controlled, keeping simulated workspace outputs objective.",
                    "We respect data safety. Algorithms avoid manipulative trends. Your privacy framework details are in Settings > Privacy & Policy.",
                    "AI rights concern user control over predictive context. You decide what metrics to stress test and manage.",
                    "Ethical considerations ensure our trajectory mapping is conservative yet motivational, rather than overly speculative.",
                    "Your personal scenarios are confidential. Model weights are trained solely on broad public demographic vectors to ensure fairness."
                ]

            elif any(w in user_msg for w in ["hello", "hi", "hey", "help"]):
                replies = [
                    "Hello! I'm your FutureAI assistant. I can help you interpret your simulation results or guide your next career move.",
                    "Hey there! Ready to optimize your future? Ask me about your latest prediction or career paths.",
                    "Hi! I've been analyzing your data. We have some interesting trends to discuss!",
                    "Greetings! I'm here to provide actionable insights about your future. What category should we explore today?",
                    "Hello! Whether it's career, finance, or education, I'm ready to assist you."
                ]

            # Moderate confidence Semantic Search match fallback
            if not replies and best_score > 0.4:
                reply = chat_data.iloc[best_idx]["response"]

            if not reply and replies:
                reply = random.choice(replies)

            # Absolute final fallback if both model & keywords fail
            if not reply:
                if last_prediction:
                    reply = random.choice([
                        f"Interesting point!{prediction_context} I'm also seeing a {context_data['comp']} future comparison score. What do you think about this timeline?",
                        f"I've considered that.{prediction_context} The AI model suggests your life satisfaction ({context_data['sat']}) is the most important variable here.",
                        "That's a unique perspective. Let's look at how we can improve your success probability above the current level."
                    ])
                else:
                    reply = random.choice([
                        "That's a great question! I'd love to give you a specific answer—try running a Future Prediction first so I have your data.",
                        "Our goal is to use AI to navigate these complex life decisions. What specific part of your life should we simulate next?"
                    ])

        reply = build_clear_chat_reply(
            original_message,
            role="General",
            context_data=context_data,
            matched_response=matched_response if matched_response else reply
        )

        # Save to DB
        cursor = mysql.connection.cursor()
        try:
            # Save User Message
            cursor.execute(
                "INSERT INTO chat_messages (user_id, sender, message) VALUES (%s, %s, %s)",
                (user_id, 'user', original_message)
            )
            # Save AI Reply
            cursor.execute(
                "INSERT INTO chat_messages (user_id, sender, message) VALUES (%s, %s, %s)",
                (user_id, 'ai', reply)
            )
            mysql.connection.commit()
            cursor.close()
        except Exception as db_err:
            print("CHAT DB SAVE ERROR:", db_err)
            if 'cursor' in locals(): cursor.close()

        return jsonify({"reply": reply, "match_score": round(float(best_score), 3)})

    except Exception as e:
        import traceback
        print("\n=== CHAT ASSISTANT EXCEPTION TRACEBACK ===")
        traceback.print_exc()
        raise e


@app.route('/chat_support', methods=['POST'])
def chat_support():
    try:
        data = request.get_json()
        user_msg = data.get("message", "").lower()
        
        # Support-focused trained responses
        if any(w in user_msg for w in ["password", "login", "locked", "signin"]):
            reply = "To reset your password, go to the Login screen and tap 'Forgot Password'. We'll send a recovery email to your registered address."
        elif any(w in user_msg for w in ["photo", "avatar", "picture", "profile"]):
            reply = "You can update your profile photo from the 'Edit Profile' section. Just tap on your current avatar to upload a new one!"
        elif any(w in user_msg for w in ["delete", "remove", "account", "history"]):
            reply = "You can manage your data and delete simulation history from Settings > Privacy & Policy. Be careful, this action cannot be undone."
        elif any(w in user_msg for w in ["accurate", "correct", "wrong", "true", "prediction"]):
            reply = "Our AI uses probabilistic models based on your input. Accuracy improves as you provide more details through journaling and goal setting."
        elif any(w in user_msg for w in ["help", "human", "contact", "support", "call", "email"]):
            reply = "I'm your AI support assistant! If I can't help, you can email us at support@futureai.com or use the 'Contact Support' button in the Help Center."
        elif any(w in user_msg for w in ["slow", "bug", "error", "working", "crash"]):
            reply = "We're sorry to hear that. Please try clearing the app cache or restarting. If the issue persists, let us know the exact error message."
        elif any(w in user_msg for w in ["goal", "target", "career", "finance"]):
            reply = "You can explore career and finance simulations using the Future Predictor. Our AI will help analyze different scenarios for you."
        elif any(w in user_msg for w in ["right", "ethic", "rule", "law", "policy", "integrity"]):
            reply = "We prioritse AI ethics and user safety. FutureAI ensures predictive models remain secure, objective, and unbiased."
        elif any(w in user_msg for w in ["hello", "hi", "hey"]):
            reply = "Hello! I'm the FutureAI Support Assistant. How can I help you with the app today?"
        else:
            # Semantic search fallback
            try:
                user_vec = chat_model.encode([user_msg])
                scores = cosine_similarity(user_vec, chat_embeddings)[0]
                best_idx = int(np.argmax(scores))
                
                if scores[best_idx] > 0.6: # High confidence match
                    reply = chat_data.iloc[best_idx]["response"]
                elif scores[best_idx] > 0.4: # Moderate confidence, reframe correctly
                    reply = f"Based on our general knowledge: " + chat_data.iloc[best_idx]["response"]
                else:
                    reply = "That's an interesting question! While I focus on app support, you might find specific insights by running a new Future Simulation."
            except Exception as model_err:
                print("Support chat model error:", model_err)
                reply = "I'm here to help with any app-related questions! Could you please clarify your request?"

        return jsonify({"reply": reply})

    except Exception as e:
        print("SUPPORT CHAT ERROR:", e)
        return jsonify({"reply": "I'm having trouble connecting to support. Please try again later."}), 500


@app.route('/chat_history/<int:user_id>', methods=['GET'])
def get_chat_history(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            "SELECT sender, message, created_at FROM chat_messages WHERE user_id=%s ORDER BY created_at ASC",
            (user_id,)
        )
        history = cursor.fetchall()
        cursor.close()
        return jsonify(list(history))
    except Exception as e:
        print("CHAT HISTORY ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# AI CHATBOT (NO API)
# ---------------------------------
@app.route('/chat_ai', methods=['POST'])
def chat_ai():
    try:
        data = request.get_json() or {}
        user_msg = (data.get("message") or "").strip()

        if not user_msg:
            return jsonify({"reply": "Please enter a message"}), 400

        # Convert user message → vector
        user_vec = chat_model.encode([user_msg])

        # Compare with dataset
        scores = cosine_similarity(user_vec, chat_embeddings)[0]

        # Get best match
        best_idx = int(np.argmax(scores))

        matched_response = str(chat_data.iloc[best_idx]["response"]).strip()
        category = chat_data.iloc[best_idx]["category"]
        reply = build_clear_chat_reply(user_msg, matched_response=matched_response)

        return jsonify({
            "reply": reply,
            "category": category
        })

    except Exception as e:
        print("CHAT ERROR:", e)
        return jsonify({
            "reply": "Something went wrong"
        }), 500


# -------------------- AI CHATBOT ROUTE --------------------

chat_history = []

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}

        message = (data.get("message") or "").strip()
        role = data.get("role", "General")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        reply = build_clear_chat_reply(message, role=role)

        # Save chat history (optional)
        chat_history.append({
            "message": message,
            "role": role,
            "reply": reply
        })

        return jsonify({
            "status": "success",
            "role": role,
            "user_message": message,
            "reply": reply
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# RETRAIN MODEL
# ---------------------------------
@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    global model
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        csv_path = "dataset.csv"
        if not os.path.exists(csv_path):
            # Try searching in parent or subfolders
            csv_path = "c:/Users/HP Personal/AndroidStudioProjects/FutureAI/dataset.csv"

        if not os.path.exists(csv_path):
            return jsonify({"status": False, "message": "dataset.csv not found"}), 404

        df = pd.read_csv(csv_path)
        df = df.drop(columns=['Final_Weight'], errors='ignore')
        
        y = df['Income']
        X = df.drop(columns=['Income'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        new_model = RandomForestClassifier()
        new_model.fit(X_train, y_train)

        joblib.dump(new_model, "future_ai_simulator.pkl")
        joblib.dump(new_model, "model.pkl")
        
        model = new_model
        return jsonify({"status": True, "message": "Model Retrained Successfully"})

    except Exception as e:
        return jsonify({"status": False, "message": str(e)}), 500

# ---------------------------------
# PREDICTION INSIGHTS (ALERTS, SUMMARY, FORECAST)
# ---------------------------------
@app.route('/prediction_insights/<int:user_id>', methods=['GET'])
def prediction_insights(user_id):
    try:
        import traceback
        from datetime import datetime, timedelta
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # 1. Fetch all predictions for this user
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
        history = list(cursor.fetchall()) # Convert to list to avoid tuple issues
        
        if not history:
            return jsonify({
                "alerts": [],
                "weekly_summary": {
                    "count": 0,
                    "avg_success": 0,
                    "status": "No data yet",
                    "date_range": "Start your first simulation!"
                },
                "forecast": {
                    "trajectory": "+0%",
                    "message": "Start simulations to see your future growth trajectory.",
                    "years": [2026, 2027, 2028, 2029, 2030],
                    "career_score": 0,
                    "finance_score": 0,
                    "life_balance_score": 0
                }
            })

        latest = history[0]
        
        # Ensure latest has all needed keys with defaults if missing
        latest_success = float(latest.get('success_probability') or 50)
        latest_fin = float(latest.get('financial_impact') or 50)
        latest_life = float(latest.get('life_satisfaction') or 50)

        # --- ALERTS GENERATION ---
        alerts = []
        if latest_success < 45:
            alerts.append({
                "title": "Low Success Probability",
                "message": f"Your latest simulation shows only {latest_success}% success. Consider adjusting your variables.",
                "type": "risk",
                "recommendation": "Try increasing your education level or capital gains."
            })
        
        if latest_fin < 30:
            alerts.append({
                "title": "Financial Risk",
                "message": "Predicted financial impact is low. This path may not yield the expected wealth.",
                "type": "warning",
                "recommendation": "Review your investment strategy."
            })
            
        if latest_life < 50:
            alerts.append({
                "title": "Burnout Warning",
                "message": "Life satisfaction is dipping. Your balance between work and personal life needs attention.",
                "type": "wellbeing",
                "recommendation": "Reduce hours per week in the next simulation."
            })
            
        # --- High Alternate Risk Alert ---
        latest_alt = float(latest.get('alternative_scenario') or 0)
        if latest_alt >= 70:
            alerts.append({
                "title": "High Alternate Risk",
                "message": f"Your latest simulation carries a ({round(latest_alt,1)}%) alternate risk failure. Consider adjusting your variables.",
                "type": "risk",
                "recommendation": "Try reducing your risk factor or increasing capital gains."
            })
        if not alerts:
            alerts.append({
                "title": "On the Right Track",
                "message": "Your current trajectory is stable and shows consistent growth.",
                "type": "info",
                "recommendation": "Keep up the consistent effort!"
            })



        # --- WEEKLY SUMMARY ---
        seven_days_ago = datetime.now() - timedelta(days=7)
        weekly_data = []
        for p in history:
            ts = p['created_at']
            if isinstance(ts, str):
                try:
                    ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except:
                    continue
            if ts >= seven_days_ago:
                weekly_data.append(p)
        
        if weekly_data:
            avg_success = sum(float(p.get('success_probability') or 50) for p in weekly_data) / len(weekly_data)
            weekly_summary = {
                "count": len(weekly_data),
                "avg_success": round(avg_success, 1),
                "status": "Positive Trending" if avg_success > 60 else "Review Needed",
                "date_range": f"{seven_days_ago.strftime('%b %d')} - {datetime.now().strftime('%b %d, %Y')}"
            }
        else:
            weekly_summary = {
                "count": 0,
                "avg_success": 0,
                "status": "No activity this week",
                "date_range": "Past 7 Days"
            }

        # --- LONG TERM FORECAST ---
        base_growth = (latest_success / 100) * 20
        forecast = {
            "trajectory": f"+{round(base_growth, 1)}%",
            "message": f"Based on your current {latest_success}% success rate, your 5-year outlook is strong.",
            "years": [datetime.now().year + i for i in range(5)],
            "career_score": int(latest_success),
            "finance_score": int(latest_fin),
            "life_balance_score": int(latest_life)
        }

        cursor.close()
        return jsonify({
            "alerts": alerts,
            "weekly_summary": weekly_summary,
            "forecast": forecast
        })

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print("INSIGHTS ERROR:", err_msg)
        return jsonify({"error": str(e), "trace": err_msg}), 500

# ---------------------------------
# GET TIMELINE
# ---------------------------------
@app.route('/get_timeline/<int:user_id>', methods=['GET'])
def get_timeline(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 1", (user_id,))
        last = cursor.fetchone()
        cursor.close()

        score = int(float(last.get('success_probability') or 50)) if last else 50
        input_dec = str(last.get('decision_input') or "career").lower() if last else "career"

        if "edu" in input_dec and not "work" in input_dec:
            nodes = [
                {"title": "Begin Program", "date": "Month 1", "desc": "Enroll in advanced degree", "active": True},
                {"title": "First Milestone", "date": "Month 6", "desc": "Complete foundational courses", "active": score > 40},
                {"title": "Skill Application", "date": "Month 12", "desc": "Start practical projects", "active": score > 50},
                {"title": "Networking", "date": "Month 18", "desc": "Connect with industry leaders", "active": score > 60},
                {"title": "Graduation", "date": "Month 24", "desc": "Receive credentials", "active": score > 70},
                {"title": "Job Placement", "date": "Month 26", "desc": "Secure higher-paying role", "active": score > 80}
            ]
        elif "finance" in input_dec or "cap" in input_dec:
            nodes = [
                {"title": "Initial Investment", "date": "Month 1", "desc": "Setup diversified portfolio", "active": True},
                {"title": "Market Adjustment", "date": "Month 3", "desc": "Rebalance assets based on risk", "active": score > 40},
                {"title": "First Returns", "date": "Month 6", "desc": "Observe initial growth", "active": score > 50},
                {"title": "Compound Interest", "date": "Month 12", "desc": "Reinvest dividends", "active": score > 60},
                {"title": "Portfolio Maturity", "date": "Month 24", "desc": "Achieve stable yield", "active": score > 70},
                {"title": "Wealth Milestone", "date": "Month 36+", "desc": "Hit financial independence goal", "active": score > 80}
            ]
        else:
            nodes = [
                {"title": "Skill Acquisition", "date": "Month 1", "desc": "Start learning modern tech", "active": True},
                {"title": "Certification", "date": "Month 3", "desc": "Pass industry exams", "active": score > 40},
                {"title": "Pivot Strategy", "date": "Month 6", "desc": "Apply for tech roles", "active": score > 50},
                {"title": "First Offer", "date": "Month 9", "desc": "Receive competitive salary", "active": score > 60},
                {"title": "Promotion", "date": "Month 18", "desc": "Move to senior position", "active": score > 70},
                {"title": "Leadership", "date": "Month 36+", "desc": "Lead projects and teams", "active": score > 80}
            ]

        return jsonify(nodes)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print("TIMELINE ERROR:", err)
        return jsonify([{"title": "Backend Error", "date": "N/A", "desc": str(e), "active": False}])


# ---------------------------------
# GET PROFILE STATS
# ---------------------------------
@app.route('/profile-stats/<email>', methods=['GET'])
def profile_stats(email):
    try:
        cursor = mysql.connection.cursor()

        # total predictions
        cursor.execute("SELECT COUNT(*) FROM simulations WHERE user_email=%s", (email,))
        predictions = cursor.fetchone()[0]

        # average score
        cursor.execute("SELECT AVG(score) FROM simulations WHERE user_email=%s", (email,))
        avg_score = cursor.fetchone()[0]

        # days active
        cursor.execute("SELECT COUNT(DISTINCT created_at) FROM simulations WHERE user_email=%s", (email,))
        days_active = cursor.fetchone()[0]

        cursor.close()

        return jsonify({
            "avg_score": round(avg_score, 2) if avg_score else 0,
            "predictions": predictions or 0,
            "days_active": days_active or 0
        })
    except Exception as e:
        print("PROFILE STATS ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ---------------------------------
# GET USER STATS (Retrofit)
# ---------------------------------
@app.route('/user-stats/<int:user_id>', methods=['GET'])
def user_stats(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        
        # user details
        cursor.execute("SELECT name, email, profile_photo FROM users WHERE user_id=%s", (user_id,))
        user_info = cursor.fetchone()
        
        if not user_info:
            cursor.close()
            return jsonify({"error": "User not found"}), 404
            
        # stats
        cursor.execute(
            "SELECT COUNT(*) as total_predictions, AVG(success_probability) as avg_score, COUNT(DISTINCT DATE(created_at)) as days_active FROM predictions WHERE user_id=%s", 
            (user_id,)
        )
        stats = cursor.fetchone()
        cursor.close()
        
        return jsonify({
            "avg_score": round(float(stats['avg_score'] or 0.0), 2),
            "total_predictions": stats['total_predictions'] or 0,
            "days_active": stats['days_active'] or 0,
            "name": user_info['name'],
            "email": user_info['email'],
            "profile_photo": user_info['profile_photo']
        })
    except Exception as e:
        print("USER STATS ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# ALTERNATE SCENARIOS
# ---------------------------------
@app.route('/alternate_scenarios/<int:user_id>', methods=['GET'])
def alternate_scenarios(user_id):
    try:
        # Generate some contextual scenarios
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT decision_input, success_probability FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 1", (user_id,))
        last_prediction = cursor.fetchone()
        cursor.close()

        base_score = int(float(last_prediction.get('success_probability') or 50)) if last_prediction else 50
        
        # Decide category based on input text if possible, else generic
        input_text = last_prediction['decision_input'].lower() if last_prediction else "career"
        if "edu" in input_text and not "work" in input_text:
            scenarios = [
                {
                    "title": "Pursue Advanced Degree",
                    "score": min(100, base_score + 15),
                    "pros": "Higher salary ceiling, specialized roles",
                    "cons": "2-3 years of tuition, delayed entry"
                },
                {
                    "title": "Bootcamp Certification",
                    "score": min(100, base_score + 5),
                    "pros": "Fast track to job market (6 mos)",
                    "cons": "Lower starting salary, highly competitive"
                },
                {
                    "title": "Self-Taught Portfolio",
                    "score": min(100, base_score + 10),
                    "pros": "No debt, flexible timeline",
                    "cons": "Lacks formal credential, requires discipline"
                }
            ]
        elif "cap" in input_text or "finance" in input_text:
             scenarios = [
                {
                    "title": "Aggressive ETF Portfolio",
                    "score": min(100, base_score + 12),
                    "pros": "High potential returns, compounded growth",
                    "cons": "Higher market volatility risk"
                },
                {
                    "title": "Real Estate Re-investment",
                    "score": min(100, base_score + 8),
                    "pros": "Tangible asset, passive income",
                    "cons": "High barrier to entry, illiquid"
                },
                {
                    "title": "Conservative Bonds & Savings",
                    "score": min(100, base_score - 5),
                    "pros": "Guaranteed returns, highly liquid",
                    "cons": "Returns may not beat inflation"
                }
            ]
        else:
            scenarios = [
                {
                    "title": "Pivot to Tech Sector",
                    "score": min(100, base_score + 18),
                    "pros": "High growth potential, remote work options",
                    "cons": "Steep learning curve required"
                },
                {
                    "title": "Internal Promotion Path",
                    "score": min(100, base_score + 4),
                    "pros": "Stable income, known environment",
                    "cons": "Slower financial growth"
                },
                {
                    "title": "Freelance Consultation",
                    "score": min(100, base_score + 9),
                    "pros": "Be your own boss, uncapped earning",
                    "cons": "Inconsistent income, no benefits"
                }
            ]
            
        return jsonify(scenarios)
    except Exception as e:
        print("SCENARIOS ERROR:", e)
        return jsonify([{"title": "Error generating scenarios", "score": 0, "pros": "", "cons": str(e)}])

# ---------------------------------
# COMPARE FUTURES
# ---------------------------------
@app.route('/compare_futures/<int:user_id>', methods=['GET'])
def compare_futures(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 1", (user_id,))
        last_prediction = cursor.fetchone()
        cursor.close()

        input_decision = "Current Path"
        base_score = 50
        fin_impact = 50.0
        life_sat = 50.0
        
        if last_prediction:
            input_decision = last_prediction.get('decision_input') or 'Current Path'
            base_score = int(float(last_prediction.get('success_probability') or 50))
            fin_impact = float(last_prediction.get('financial_impact') or 50)
            life_sat = float(last_prediction.get('life_satisfaction') or 50)

        # Option B is the user's input, Option A is the better AI alternative
        improved_score = min(100, base_score + 15)
        improved_fin = fin_impact * 1.3
        improved_life = min(100, life_sat + 12)
        
        rows = [
            {
                "label": "Success Probability",
                "valA": f"{improved_score}%",
                "valB": f"{base_score}%",
                "isWinnerA": True
            },
            {
                "label": "Financial Impact",
                "valA": f"₹{int(improved_fin)}K",
                "valB": f"₹{int(fin_impact)}K",
                "isWinnerA": True
            },
            {
                "label": "Life Satisfaction",
                "valA": f"{int(improved_life)}/100",
                "valB": f"{int(life_sat)}/100",
                "isWinnerA": True
            },
            {
                "label": "Time Investment",
                "valA": "High (30h/w)",
                "valB": "Low (10h/w)",
                "isWinnerA": False
            },
            {
                "label": "Risk Level",
                "valA": "Moderate",
                "valB": "Low",
                "isWinnerA": False
            },
            {
                "label": "Stress Factor",
                "valA": "Elevated",
                "valB": "Stable",
                "isWinnerA": False
            },
            {
                "label": "Career Growth",
                "valA": "Exponential",
                "valB": "Linear",
                "isWinnerA": True
            }
        ]
        category_str = input_decision.lower()
        if "edu" in category_str and not "work" in category_str:
            if improved_score <= 50:
                option_a = "Online Certification"
                verdict = f"Option A ({option_a}) requires less commitment and provides foundational skills, fitting a safer success probability."
            elif improved_score <= 75:
                option_a = "Part-Time Master's Degree"
                verdict = f"Option A ({option_a}) balances current responsibilities while significantly boosting your credentials."
            else:
                option_a = "Full-Time Advanced Degree"
                verdict = f"Option A ({option_a}) offers the highest long-term satisfaction and drastically accelerates your career trajectory."
        elif "cap" in category_str or "finance" in category_str:
            if improved_score <= 50:
                option_a = "Conservative Bonds & Savings"
                verdict = f"Option A ({option_a}) minimizes risk and secures capital, ideal for uncertain economic conditions."
            elif improved_score <= 75:
                option_a = "Balanced Index Funds"
                verdict = f"Option A ({option_a}) offers steady financial growth with moderate risk exposure for long-term wealth."
            else:
                option_a = "Aggressive Investment Portfolio"
                verdict = f"Option A ({option_a}) offers significantly higher financial returns but requires managing more volatility."
        else:
            if improved_score <= 50:
                option_a = "Gradual Skill Building"
                verdict = f"Option A ({option_a}) allows you to steadily modernise your tech skills without risking your current stability."
            elif improved_score <= 75:
                option_a = "Internal Lateral Move"
                verdict = f"Option A ({option_a}) leverages your current company's network to transition into a tech-focused role safely."
            else:
                option_a = "Tech Sector Pivot"
                verdict = f"Option A ({option_a}) aggressively accelerates your career growth with substantially higher salary potential."

        return jsonify({
            "rows": rows,
            "optionA": option_a,
            "optionB": input_decision,
            "verdict": verdict,
            "scoreA": improved_score,
            "scoreB": base_score
        })
    except Exception as e:
        print("COMPARE ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------------------
# FORECAST ROUTE
# ---------------------------------
@app.route('/forecast/<int:user_id>', methods=['GET'])
def get_forecast(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        cursor.execute(
            "SELECT success_probability, financial_impact, life_satisfaction FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        )

        latest = cursor.fetchone()
        cursor.close()

        if not latest:
            return jsonify({
                "growth": "+0%",
                "years": [2026, 2027, 2028, 2029, 2030],
                "career": [50,55,60,65,70],
                "finance": [45,50,55,60,65],
                "life_balance": [50,52,55,58,60]
            })

        # Safe value extraction
        success = float(latest['success_probability'] or 50)
        finance = float(latest['financial_impact'] or 50)
        life = float(latest['life_satisfaction'] or 50)

        # Generate 5-year forecast
        career_scores = [
            round(success * 0.8),
            round(success * 0.9),
            round(success * 1.0),
            round(success * 1.05),
            round(success * 1.1)
        ]

        finance_scores = [
            round(finance * 0.7),
            round(finance * 0.8),
            round(finance * 0.9),
            round(finance * 1.0),
            round(finance * 1.1)
        ]

        life_scores = [
            round(life * 0.8),
            round(life * 0.9),
            round(life * 1.0),
            round(life * 1.05),
            round(life * 1.1)
        ]

        growth = round((career_scores[-1] - career_scores[0]) / career_scores[0] * 100, 1)

        return jsonify({
            "growth": f"+{growth}%",
            "years": [2026, 2027, 2028, 2029, 2030],
            "career": career_scores,
            "finance": finance_scores,
            "life_balance": life_scores
        })

    except Exception as e:
        print("FORECAST ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/home_data/<int:user_id>', methods=['GET'])
def get_home_data(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            "SELECT success_probability, financial_impact, life_satisfaction FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        )
        latest = cursor.fetchone()
        
        if not latest:
            cursor.close()
            return jsonify({
                "future_score": 0,
                "trend": "+0 this week",
                "career": 0,
                "finance": 0,
                "balance": 0
            })
            
        future_score = int(float(latest['success_probability'] or 0))
        career_score = future_score
        finance_score = int(float(latest['financial_impact'] or 0))
        balance_score = int(float(latest['life_satisfaction'] or 0))
        
        # Calculate trend based on history
        cursor.execute("SELECT success_probability FROM predictions WHERE user_id=%s ORDER BY created_at DESC LIMIT 10", (user_id,))
        recent = cursor.fetchall()
        
        if len(recent) > 1:
            prev_sum = sum(float(r['success_probability'] or 0) for r in recent[1:])
            prev_avg = prev_sum / (len(recent) - 1)
            diff = future_score - int(prev_avg)
            trend = f"{'↗' if diff >= 0 else '↘'} {'+' if diff >= 0 else ''}{diff} this week"
        else:
            trend = "+0 this week"
            
        cursor.close()
        return jsonify({
            "future_score": future_score,
            "trend": trend,
            "career": career_score,
            "finance": finance_score,
            "balance": balance_score
        })
    except Exception as e:
        print("HOME DATA ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# CONTACT SUPPORT
# ---------------------------------
@app.route('/contact_support', methods=['POST'])
def contact_support():
    try:
        data = request.get_json()

        name = data.get("name")
        email = data.get("email")
        subject = data.get("subject")
        message = data.get("message")

        # ✅ Validate fields
        if not name or not email or not subject or not message:
            return jsonify({
                "status": False,
                "message": "All fields are required"
            }), 400

        # ✅ Create email
        msg = Message(
            subject=f"{subject} (from {name})",
            sender=app.config['MAIL_USERNAME'],
            recipients=['sanjaykuruvella112@gmail.com']
        )

        # ✅ Reply directly to user
        msg.reply_to = email

        # ✅ Email content
        msg.body = f"""
New Support Message

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}
"""

        # ✅ Send email
        mail.send(msg)

        return jsonify({
            "status": True,
            "message": "Support message sent successfully"
        }), 200

    except Exception as e:
        print("CONTACT SUPPORT ERROR:", e)
        return jsonify({
            "status": False,
            "message": str(e)
        }), 500


# ---------------------------------
# RUN SERVER
# ---------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
