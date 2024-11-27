from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import joblib
app=Flask(__name__)
@app.route("/",methods=["GET"])
def home():
    return render_template("diaresult.html")
@app.route("/predict_price",methods=["POST"])
def prepri():
    if request.method=='POST':
        carat=float(request.form["carat"])
        cut=int(request.form["cut"])
        color=int(request.form["color"])
        clarity=int(request.form["clarity"])
        depth=float(request.form["depth"])
        table=float(request.form["table"])
        x=float(request.form["x"])
        y=float(request.form["y"])
        z=float(request.form["z"])

        train=joblib.load("C:/Users/Ankita/price.pkl")
        test_data=[[carat,cut,color,clarity,depth,table,x,y,z]]
        test_data=np.array(test_data)
        test_data=pd.DataFrame(test_data)
        prediction = train.predict(test_data)
        return render_template('diaresult1.html',prediction=prediction) 
        
if __name__=="__main__":
    app.run(debug=True)
