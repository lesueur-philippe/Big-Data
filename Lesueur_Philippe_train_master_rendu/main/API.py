import sys
import pandas as pd
from flask import Flask
from flask import request
from Train_master_learning import Train_master_learning
import json

app = Flask(__name__)

@app.route("/")
def score():
    tml = Train_master_learning()
    return '{\n\n\t'+ '"score" : {} '.format(tml.score) + '\n\n}\n'

@app.route("/predict", methods = ['POST'])
def predict():
    try :
        dict = request.get_json(force=True)
    except json.decoder.JSONDecodeError:
        return '{ "review" : "xxx", "title" : "yyy", "stars" : "7" } type excpected\n'
    arr = [dict[i] for i in dict]
    data = pd.DataFrame(columns=[i for i in dict])
    data.loc[0] = arr
    try :
        data["stars"] = pd.to_numeric(data["stars"], downcast = 'integer')
    except ValueError:
        return 'stars need to be an integer value.\n'
    if any(data["stars"] > 5) or any(data["stars"] < 1):
        return 'stars should be betwin 1 and 5.\n'
    data = data.rename(columns={'stars':'review_stars', 'title':'review_title', 'review':'review_content'})
    tml = Train_master_learning()
    return  '{\n\n\t'+ '"prediction" : {}, \n\t"score" : {} '.format(tml.predict(data), tml.score) + '\n\n}\n'

if __name__ == '__main__':
    app.run()