from flask import Flask, render_template

import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json

app = Flask(__name__)



def create_plot():


    N = 40
    x = np.linspace(0, 1, N)
    y = np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe


    data = [
        go.Bar(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@app.route('/')
def index():
    #return "Hello world!"
    bar = create_plot()
    return render_template('index.html', plot=bar)


@app.route('/bar', methods=['GET', 'POST'])
def change_features():

    feature = request.args['selected']
    graphJSON= create_plot(feature)

    return graphJSON

if __name__=='__main__':
    app.run()