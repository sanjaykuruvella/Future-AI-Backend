from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# MySQL Configuration (XAMPP default)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''   # XAMPP default is empty
app.config['MYSQL_DB'] = 'decision_app'

mysql = MySQL(app)

# ------------------------------
# GET All Scenarios
# ------------------------------
@app.route('/scenarios', methods=['GET'])
def get_scenarios():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM scenarios")
    rows = cur.fetchall()

    scenarios = []
    for row in rows:
        scenarios.append({
            "id": row[0],
            "title": row[1],
            "description": row[2],
            "result": row[3],
            "created_at": row[4]
        })

    cur.close()
    return jsonify(scenarios)


# ------------------------------
# POST New Scenario
# ------------------------------
@app.route('/scenarios', methods=['POST'])
def create_scenario():
    data = request.get_json()

    title = data['title']
    description = data['description']
    result = data.get('result', '')

    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO scenarios (title, description, result) VALUES (%s, %s, %s)",
        (title, description, result)
    )
    mysql.connection.commit()
    cur.close()

    return jsonify({"message": "Scenario created successfully"}), 201


# ------------------------------
# DELETE Scenario
# ------------------------------
@app.route('/scenarios/<int:id>', methods=['DELETE'])
def delete_scenario(id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM scenarios WHERE id = %s", (id,))
    mysql.connection.commit()
    cur.close()

    return jsonify({"message": "Scenario deleted successfully"})


if __name__ == '__main__':
    app.run(debug=True)
