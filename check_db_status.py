import MySQLdb

def check_db():
    try:
        db = MySQLdb.connect(host="localhost", user="root", passwd="", db="futureai")
        cursor = db.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"Total Users: {user_count}")
        
        cursor.execute("SELECT * FROM users LIMIT 5")
        users = cursor.fetchall()
        print("Users:", users)
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        print(f"Total Predictions: {pred_count}")
        
        cursor.execute("SELECT * FROM predictions LIMIT 5")
        preds = cursor.fetchall()
        print("Predictions:", preds)
        
        db.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
