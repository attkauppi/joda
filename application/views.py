from flask import Flask, render_template, request
from application import app

import flask
import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np
import joblib
import traceback
from flask_restful import reqparse

import math
from math import expm1
import numpy as np
import pandas as pd

from math import expm1

import flask
import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np
import joblib
import traceback
from flask_restful import reqparse

import math
import numpy as np
import pandas as pd

# Plotting
from .plotting import do_plot, helsinki_median_price_guest_nmbr, category_counts, reviews_score_rating, hel_review_scores


model_typistetty_man = pickle.load(open('application/pickles/model_typistetty_man.pkl', 'rb'))
model_columns = joblib.load("application/pickles/model_cols_typistetty_man.pkl", 'rb')
model_transformed = pickle.load(open('application/pickles/model_man_transformed.pkl', 'rb'))

app.config['TEMPLATES_AUTO_RELOAD'] = True


model1 = pickle.load(open('application/pickles/model_typistetty_hel.pkl', 'rb'))
model1_columns = pickle.load(open('application/pickles/model_cols_typistetty_hel.pkl', 'rb'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_hel_typistetty/", methods=['POST'])
#@app.route('/', methods=['POST'])
def predict1():
    # get data
    data = request.get_json(force=True)
    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    #result = model.predict(data_df)
    result = model1.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)



@app.route("/transformed_predict", methods=["POST"])
def predict_transformed():
    data = request.get_json(force=True)

    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions

    result = model_transformed.predict(data_df)

    output = {"results": int(result[0])}

    return jsonify(results=output)

# Plotting paths
# @app.route('/plots/manchester/time_since_last_review', methods=["GET"])
# def time_since_last_review_man():
#     #<img src="http://localhost:5000/plots/manchester/time_since_last_review"/>
#     bytes_obj = do_plot()

#     return flask.send_file(bytes_obj, attachment_filename='time_since_last_review_man.png', mimetype='image/png')

@app.route('/plots/helsinki/median_price_guest_number/', methods=["GET"])
def helsinki_median_price_guest_number():
    bytes_obj = helsinki_median_price_guest_nmbr()
    print('Views metodin bytes_obj')
    print(bytes_obj)

    return flask.send_file(bytes_obj, attachment_filename='helsinki_median_price_guest_number.png', mimetype='image/png')

# @app.route('/plots/helsinki/reviews_score_rating/', methods=["GET"])
# def reviews_score_rating():
#     bytes_obj = reviews_score_rating()

    # return flask.send_file(bytes_obj, attachment_filename='reviews_score_rating.png', mimetype='image/png')

@app.route('/plots/helsinki/hel_review_scores/', methods=["GET"])
def hel_review_scores():
    """
    Returns a collage of all df columns starting with the string reviews_scores
    """
    bytes_obj = reviews_score_rating()
    return flask.send_file(bytes_obj, attachment_filename='reviews_score_rating.png', mimetype="image/png")

#### Olisi edellyttänyt jonkinlaista lista viritelmää
# @app.route("/plots/helsinki/category_counts/", methods=["GET"])
# def helsinki_category_counts():
#     bytes_objects = category_counts()
#     j = 0
#     for i in bytes_objects:
#         bytes_obj = i
#         name = 'helsinki_category_counts'+str(j)+'.png'
#         j += 1
#         return flask.send_file(bytes_obj, attachment_filename=name, mimetype='image/png')
        


# @app.route('/plots/helsinki/review_category_counts', methods=["GET"])
# def helsinki_review_category_counts():


## Map plotting paths
## # https://stackoverflow.com/questions/36137161/using-flask-to-embed-a-local-html-page
@app.route("/map")#, methods=["GET"])
def map():
    ##return render_template('templates/number_of_listings_hel.html')
    return flask.send_file("templates/number_of_listings_hel.html")

@app.route("/map_median_price")
def map_median_price():
    return flask.send_file("templates/median_price_hel.html")


# Api-paths

# @app.route("/API/transformed_predict", methods=["POST"])
# def predict_transformed():
#     data = request.get_json(force=True)

#     data.update((x, [y]) for x, y in data.items())
#     data_df = pd.DataFrame.from_dict(data)

#     # predictions

#     result = model_transformed.predict(data_df)

#     output = {"results": int(result[0])}

#     return jsonify(results=output)

@app.route("/predict_man_typistetty/", methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    #result = model.predict(data_df)
    result = model_typistetty_man.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)