from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load and train the model
data = pd.read_csv("data/student-mat.csv")
X = data.drop("G3", axis=1)
y = data["G3"]
X = pd.get_dummies(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        
        # Extract input data
        age = request_data.get('age')
        studytime = request_data.get('studytime')
        absences = request_data.get('absences')
        g1 = request_data.get('G1')
        g2 = request_data.get('G2')
        
        if None in [age, studytime, absences, g1, g2]:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Create input dataframe
        input_data = pd.DataFrame({
            "age": [age],
            "studytime": [studytime],
            "absences": [absences],
            "G1": [g1],
            "G2": [g2]
        })
        
        # Match columns with training data
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({"prediction": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
