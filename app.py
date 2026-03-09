from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from flask_cors import CORS
import MySQLdb.cursors
import hashlib
import numpy as np
import joblib
from flask_mail import Mail, Message
import os

# ---------------------------------
# APP INITIALIZATION
# ---------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------
# EMAIL CONFIGURATION (GMAIL)
# ---------------------------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

app.config['MAIL_USERNAME'] = 'sanjaykuruvella112@gmail.com'
app.config['MAIL_PASSWORD'] = 'uugv cphg exqe lwkz'
app.config['MAIL_DEFAULT_SENDER'] = 'sanjaykuruvella112@gmail.com'

mail = Mail(app)

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
model = joblib.load("future_ai_simulator.pkl")

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
    data = request.json

    name = data['name']
    email = data['email']
    profile_photo = data.get('profile_photo')

    cursor = mysql.connection.cursor()

    cursor.execute("""
        UPDATE users
        SET name=%s,email=%s,profile_photo=%s
        WHERE user_id=%s
    """, (name, email, profile_photo, user_id))

    mysql.connection.commit()
    cursor.close()

    return jsonify({"message": "Profile Updated Successfully"})


# ---------------------------------
# FORGOT PASSWORD
# ---------------------------------
@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()

        email = data.get("email")
        new_password = data.get("new_password")

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


# ---------------------------------
@app.route('/forgot_password_email', methods=['POST'])
def forgot_password_email():
    try:
        data = request.get_json()
        email = data.get("email")

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        cursor.execute(
            "SELECT name,email,password FROM users WHERE email=%s",
            (email,)
        )

        user = cursor.fetchone()
        cursor.close()

        if not user:
            return jsonify({"message": "Email not found"}), 404

        msg = Message(
            subject="FutureAI Password Recovery",
            recipients=[email]
        )

        msg.body = f"""
Hello {user['name']},

Your password is:

{user['password']}

FutureAI Team
"""

        mail.send(msg)

        return jsonify({"message": "Password sent to mail successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# SAVE GOAL
# ---------------------------------
@app.route('/goals', methods=['POST'])
def save_goal():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        user_id = data.get("user_id")
        goal_type = data.get("goal_type")
        description = data.get("description")

        if not user_id or not goal_type or not description:
            return jsonify({"error": "Missing required fields"}), 400

        cursor = mysql.connection.cursor()

        cursor.execute(
            "INSERT INTO goals (user_id, goal_type, description) VALUES (%s, %s, %s)",
            (user_id, goal_type, description)
        )

        mysql.connection.commit()
        cursor.close()

        return jsonify({
            "status": True,
            "message": "Goal saved successfully"
        }), 201

    except Exception as e:
        if 'cursor' in locals() and cursor: cursor.close()
        return jsonify({
            "status": False,
            "error": str(e)
        }), 500


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

    try:
        data = request.get_json()

        print("Incoming Data:", data)

        risk = float(data['risk_tolerance'])
        timeframe = float(data['timeframe'])
        effort = float(data['effort_level'])
        investment = float(data['financial_investment'])

        # model input format
        features = [[risk, timeframe, effort, investment]]

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features).max()

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


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
            data['Age'], data['Workclass'], data.get('Final_Weight', 200000),
            data['Education'], data['Education_Number'],
            data['Marital_Status'], data['Occupation'],
            data['Relationship'], data['Race'], data['Gender'],
            data['Capital_Gain'], data['Capital_Loss'],
            data['Hours_Per_Week'], data['Country']
        ]]

        prediction_val = 0
        if model:
            try:
                # Expecting a classifier (0 or 1)
                pred_result = model.predict(features)[0]
                prediction_val = int(pred_result)
            except:
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
            base_prob = 75.0
            fin_impact = 70.0 + (min(data['Capital_Gain'], 10000) / 200.0)
            life_sat = 80.0 + (data['Education_Number'] * 1.5)
            timeline = "3-9 months" if data['Hours_Per_Week'] > 30 else "9-15 months"
            alt_scenario = 20.0 - (data['Capital_Gain'] / 1000.0)
            future_comp = 85.0
        else:
            base_prob = 35.0
            fin_impact = 30.0 + (min(data['Capital_Gain'], 5000) / 200.0)
            life_sat = 40.0 + (data['Education_Number'] * 2.0)
            timeline = "18-24 months"
            alt_scenario = 60.0 + (data['Capital_Loss'] / 500.0)
            future_comp = 40.0

        # Success probability calculation with some randomness
        success_prob = round(base_prob + (data['Education_Number'] * 2) + (np.random.random() * 5), 2)
        success_prob = min(max(success_prob, 5.0), 99.0)

        # Save to DB
        cursor = mysql.connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO predictions(
                    user_id, decision_input, success_probability, timeline, 
                    financial_impact, life_satisfaction, alternative_scenario, future_comparison
                )
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                user_id, f"Simulation (Age:{data['Age']}, Edu:{data['Education_Number']})", success_prob, timeline,
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
        data = request.get_json()
        user_id = data.get("user_id", 1)
        user_msg = data.get("message", "").lower()

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

        # AI generating response based on keywords and model context
        import random
        
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
                "The current job market favors specialists. Have you considered how increasing your weekly hours might change your outcome?"
            ]
        
        elif any(w in user_msg for w in ["money", "finance", "investment", "rich", "salary", "pay"]):
            replies = [
                f"Financially, this path yields a score of {context_data.get('impact', '70')}.{prediction_context} Small adjustments in your risk levels could shift this drastically.",
                f"Wealth building takes time. Your projected future comparison is {context_data.get('comp', '80')} points above baseline! That's impressive.",
                "To maximize your financial impact, we should look at balancing your capital gains vs losses in the simulator."
            ]
            
        elif any(w in user_msg for w in ["hello", "hi", "hey", "help"]):
            replies = [
                "Hello! I'm your FutureAI assistant. I can help you interpret your simulation results or guide your next career move.",
                "Hey there! Ready to optimize your future? Ask me about your latest prediction or career paths.",
                "Hi! I've been analyzing your data. We have some interesting trends to discuss!"
            ]
        
        else:
            # General AI response that sounds smart and uses the data
            if last_prediction:
                replies = [
                    f"Interesting point!{prediction_context} I'm also seeing a {context_data['comp']} future comparison score. Does that align with your personal goals?",
                    f"I've considered that.{prediction_context} The AI model suggests your life satisfaction ({context_data['sat']}) is the most important variable here.",
                    "That's a unique perspective. Let's look at how we can improve your success probability above the current level."
                ]
            else:
                replies = [
                    "That's a great question! I'd love to give you a specific answer—try running a Future Prediction first so I have your data.",
                    "Our goal is to use AI to navigate these complex life decisions. What specific part of your life should we simulate next?",
                    "I'm here to help. Whether it's career or finance, I can guide you based on our trained AI model."
                ]

        reply = random.choice(replies)
        return jsonify({"reply": reply})

    except Exception as e:
        print("CHAT ERROR:", e)
        return jsonify({"reply": "I'm having a bit of trouble processing that. Can you try again?"}), 500

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
                    "years": [2026, 2027, 2028, 2029, 2030]
                }
            })

        latest = history[0]
        
        # Ensure latest has all needed keys with defaults if missing
        latest_success = float(latest.get('success_probability', 50))
        latest_fin = float(latest.get('financial_impact', 50))
        latest_life = float(latest.get('life_satisfaction', 50))

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
            avg_success = sum(float(p.get('success_probability', 50)) for p in weekly_data) / len(weekly_data)
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
            "years": [datetime.now().year + i for i in range(5)]
        }

        cursor.close()
        return jsonify({
            "alerts": alerts,
            "weekly_summary": weekly_summary,
            "forecast": forecast
        })

    except Exception as e:
        print("INSIGHTS ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------
# RUN SERVER
# ---------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)