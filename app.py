import os
from flask import Flask,jsonify,request
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("customer_churn_hackathon.h5")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



# model = tf.keras.models.load_model("customer_churn_hackathon.h5")
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# model.evaluate()


# dataset = pd.read_csv('Churn_Modelling.csv')
# x = dataset.iloc[:, 3:-1].values
# y = dataset.iloc[:, -1].values
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# x[:, 2] = le.fit_transform(x[:, 2])
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# x = np.array(ct.fit_transform(x))
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# model.evaluate(x_test, y_test)




@app.route('/')
def home():
    return "Customer Churn Prediction API is running!"







scalar=joblib.load('scaler.pkl')
onehot_encoder=joblib.load('onehot_encoder.pkl')
label_encoder=joblib.load('label_encoder.pkl')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data=request.get_json()

        features=[ data["CreditScore"], data["Geography"], data["Gender"], data["Age"], data["Tenure"], data["Balance"], data["NumOfProducts"], data["HasCrCard"], data["IsActiveMember"],data["EstimatedSalary"]]

        features=np.array(features).reshape(1,-1)
        features[:,2]=label_encoder.transform(features[:,2])
        features=onehot_encoder.transform(features).toarray()
        features=scalar.transform(features)
        prediction=model.predict(features)[0][0]

        return jsonify({"churn probability":float(prediction),"churn":bool(prediction>0.5)})

    except Exception as e:
        return jsonify({"error":str(e)})

        if __name__ == '__main__':
            port = int(os.environ.get('PORT', 5000))  # Use Render-assigned port
            app.run(host='0.0.0.0', debug=True,port=port)



















# @app.route('/',methods=['GET'])
# def get_data():
#     data = {
#         "message":"Hello World"
#     }
#     return jsonify(data)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',debug=True) 