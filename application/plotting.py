import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from sklearn.datasets import load_breast_cancer

# Lähde: https://gitlab.com/dice89/python-plotting-api/-/blob/master/python_plotting_api/plotting.py

sns.set()


def do_plot():
    df = pd.read_csv('data/listings_cleaned_modeling_manchester.csv', index_col=0)
    fig = df.time_since_last_review.hist(figsize=(15,5), bins=30)


    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    return bytes_image


def get_manchester_df():
    df = pd.read_csv('data/listings_cleaner_modeling_manchester.csv', index_col=0)

    #edinburgh_venues = pd.read_csv('/resources/Data_Science_Capstone/Edinburgh_Venues.csv', index_col=0)
    #m_df = pd.DataFrame(data[])
    return df

def get_manchester_time_since_last_review(df):
    #df.first_review = pd.to_datetime(df.first_review)
    #df['time_since_last_review'] = (pd.datetetime.datetime.now() - df.first_review).astype('timedelta64[D]')

    fig = df.time_since_last_review.hist(figsize=(9,3), bins=30)


    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    return bytes_image