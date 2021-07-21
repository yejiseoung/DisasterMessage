import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster.db')
df = pd.read_sql_table('disaster', engine)

# load model
model = joblib.load("../models/random_m1.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)

    # create lists for different types of categories
    energy_disaster = ['water', 'food', 'electricity']
    nat_disaster = ['floods', 'storm', 'fire', 'earthquake', 'other_weather']
    health_disaster = ['aid_related', 'medical_help', 'medical_products','hospitals', 'aid_centers']   

    health_data = pd.DataFrame(df[health_disaster])
    health_counts = health_data[health_data==1].count().sort_values(ascending=False)
    health_names = list(health_counts.index)

    energy_data = pd.DataFrame(df[energy_disaster])
    energy_counts = energy_data[energy_data==1].count().sort_values(ascending=False)
    energy_names = list(energy_counts.index)

    natural_data = pd.DataFrame(df[nat_disaster])
    natural_counts = natural_data[natural_data==1].count().sort_values(ascending=False)
    natural_names = list(natural_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=health_names,
                    y=health_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Health Related Message',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Health related disaster messages'
                }
            }
        },
        {
            'data': [
                Bar(
                    x=energy_names,
                    y=energy_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Energy Related Message',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Energy related disaster messages'
                }
            }
        },
        {
            'data': [
                Bar(
                    x=natural_names,
                    y=natural_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Natural Related Message',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Natural related disaster messages'
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()