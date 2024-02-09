from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('insurance\model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
        
    age = request.form['age']
    sex = request.form['sex']
    bmi = request.form['bmi']
    steps = request.form['steps']
    children = request.form['children']
    smoker = request.form['smoker']
    region = request.form['region']
    charges = request.form['charges']
    arr = np.array([age,sex,bmi,steps,children,smoker,region,charges])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])
    
    if pred == 1:
        result = "YES"
    else:
        result = "NO"
    return render_template('index.html', prediction=result)

if __name__ == '_main_':
    app.run(debug=True)