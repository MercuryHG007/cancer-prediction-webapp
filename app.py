from flask import Flask, render_template, request
import jsonify
import requests
import joblib
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from cancer_prediction import sc

app = Flask(__name__)


model = joblib.load(r'cancer_model.pkl')

@app.route('/',methods=['GET'])

def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])

def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        # to_predict_list = list(map(float, to_predict_list))
        # to_predict_list = list(map(format(to_predict_list,'f')))
        for i in range(0,len(to_predict_list)):
            to_predict_list[i] = float(to_predict_list[i])
            to_predict_list[i] = format(to_predict_list[i],'.3f')

        to_predict = np.array(to_predict_list).reshape(1,-1)
        print(to_predict)

        to_predict = sc.transform(to_predict)

        prediction=model.predict(to_predict)
        print(to_predict)

        output=prediction[0]
        print(output)
        
        if output==0:
            return render_template('result.html',prediction_text="Your Cancer is Benign!")
        else:
            return render_template('result.html',prediction_text="Your Cancer is Malignant!")
    else:
        return render_template('result.html',prediction_text="Enter Correct DATA")


if __name__=="__main__":
    app.run(debug=True, port=5001)
