from flask import Flask, request, render_template, jsonify
from flask_cors import cross_origin
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open("rf_model_inventory.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("deploy.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":


        
        PRODUCT_TYPE = int(request.form["PRODUCT_TYPE"])
        PROMOTION_APPLIED = int(request.form["PROMOTION_APPLIED"])
        GENERIC_HOLIDAY = int(request.form["GENERIC_HOLIDAY"])
        DAY_OF_WEEK = int(request.form["DAY_OF_WEEK"])
      
        
            
        prediction=model.predict([[
            PRODUCT_TYPE,
            PROMOTION_APPLIED,
            GENERIC_HOLIDAY,
            DAY_OF_WEEK,
        ]])
           

        
        output=int(prediction[0])

        return render_template('deploy.html',prediction_text="Number of inventory to be maintained {}".format(output))


    return render_template("deploy.html")



if __name__ == "__main__":
    app.run(debug=True)
