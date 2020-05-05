import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from sklearn.datasets import load_breast_cancer

# Lähde: https://gitlab.com/dice89/python-plotting-api/-/blob/master/python_plotting_api/plotting.py
#plt.style.use('default')

df = pd.read_csv('application/data/hy-ennen_mallinnusta.csv', index_col=0)


# def helsinki_category_counts():
#     df = pd.read_csv("application/data/hy-ennen_mallinnusta.csv", index_col=0)
#     fig = 


def category_count_plot(col, figsize=(8,4)):
    """
    Plots a simple bar chart of the total count for each category in the column specified.
    A figure size can optionally be specified.
    """
    plt.figure(figsize=figsize)
    df[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.xticks(rotation=0)

    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')

    bytes_image.seek(0)
    return bytes_image
    #plt.show()

def category_counts():
    list_categories = []

    for col in list(df.columns[df.columns.str.startswith("reviews_scores") == True]):
        list_categories.append(category_count_plot(col, figsize=(5,3)))
    
    return list_categories

def reviews_score_rating():
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.set_title('Overall listing rating', fontsize=14)
    df['reviews_scores_rating'].value_counts().sort_index(ascending=False).plot(kind='bar', color=['silver', 'darkgreen', 'yellowgreen', '#d1f28a' ], ax=ax)
    ax.set_xticklabels(labels=['no reviews', '95-100%', '80-94%', '0-79%'], rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('Number of properties', fontsize=13)

    bytes_image = io.BytesIO()

    plt.savefig(bytes_image, format='png')

    bytes_image.seek(0)
    return bytes_image


def hel_review_scores():
    """
    df columns that begin with the string review_scores to plot

    This works despite the IO approach because we're making one collage of plots, i.e. one single image. 
    """
    df_reviews = pd.read_csv("application/data/hel/before_binnings_hel.csv", index_col=0)
    variables_to_plot = list(df_reviews.columns[df.columns.str.startswith("review_scores") == True])
    fig = plt.figure(figsize=(12,8))
    for i, var_name in enumerate(variables_to_plot):
        ax = fig.add_subplot(3,3,i+1)
        df_reviews[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()

    bytes_image = io.BytesIO()

    plt.savefig(bytes_image, format='png')

    bytes_image.seek(0)
    return bytes_image
    
    
def helsinki_median_price_guest_nmbr():
    sns.set()
    df = pd.read_csv('application/data/hy-ennen_mallinnusta.csv', index_col=0)
    #fig = plt.figure(figsize=(10,5))
    plt.figure(figsize=(10,5))
    #plt.figure(figsize=(10,5))
    df.groupby('person_capacity').price_string.median().plot(kind='bar')
    plt.title('Median price of Airbnbs accommodating different number of guests', fontsize=14)
    plt.xlabel('Number of guests accommodated', fontsize=13)
    plt.ylabel('Median price (€)', fontsize=13)
    plt.xticks(rotation=0)
    plt.xlim(left=0.5)
    #plt.savefig('applications/data/Median_price_of_Airbnbs_accommodating_different_number_of_guests-helsinki.png')

    bytes_image = io.BytesIO()

    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    #file = io.open("applications/data/Median_price_of_Airbnbs_accommodating_different_number_of_guests-helsinki.png", "rb", buffering=0)
    #print(file)

    #return file
    # fig = df.groupby('person_capacity').price_string.median().bar(figsize=(10,5))
    # fig.title('Median price of Airbnbs accommodating different number of guests', fontsize=14)
    # fig.xlabel('Number of guests accommodated', fontsize=13)
    # fig.ylabel('Median price (€)', fontsize=13)
    # fig.xticks(rotation=0)
    # fig.xlim(left=0.5)
    #fig = plt.figure()
    #plt.show()
    
    #bytes_image = io.BytesIO()

    #plt.savefig(bytes_image, format='png')
    #bytes_image.seek(0)

    return bytes_image



def do_plot():
    df = pd.read_csv('application/data/listings_cleaned_modeling_manchester.csv', index_col=0)
    fig = df.time_since_last_review.hist(figsize=(15,5), bins=30)


    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)

    return bytes_image


def get_manchester_df():
    df = pd.read_csv('application/data/listings_cleaner_modeling_manchester.csv', index_col=0)

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