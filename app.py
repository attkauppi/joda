from flask import Flask, render_template

import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

@app.route('/')
def index():
    #return "Hello world!" 
    #bar = create_plot()
    return render_template('index.html')
    #return render_template('index.html', plot=bar)



if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)