from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

app = Flask(__name__)

# Load dataset
data = pd.read_csv("fuel_consumption.csv")

# Input columns
X = data[['Engine_Size', 'Cylinders', 'Horsepower', 'Weight']]

# Target column
y = data['Fuel_Consumption']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    engine = float(request.form['engine'])
    cylinders = int(request.form['cylinders'])
    horsepower = float(request.form['horsepower'])
    weight = float(request.form['weight'])

    prediction = model.predict([[engine, cylinders, horsepower, weight]])

    result = round(prediction[0], 2)

    return render_template('index.html',
                           prediction=result)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)

