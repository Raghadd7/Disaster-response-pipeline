import json
import plotly
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Description: tokenize and normalize the message data 

    Arguments:
    text: the text to be tokenized
    
    Return:
    tokens: the tokenized text 
    
    """
    
    # remove special characters and lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  

    # tokenize
    tokens = word_tokenize(text)  

    # lemmatize, remove stopwords   
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    aid_related_counts=df[df['aid_related']==1].groupby('genre').count()['aid_related']
    medical_help_counts=df[df['medical_help']==1].groupby('genre').count()['medical_help']
    weather_counts=df[df['weather_related']==1].groupby('genre').count()['weather_related']
    
    # color data dictionary 
    d = {'medical_help': 'rgb(115, 198, 182)',
     'medical_products': 'rgb(115, 198, 182)',
     'hospitals': 'rgb(115, 198, 182)',
          
     'weather_related':'rgb(174, 214, 241)',
     'floods':'rgb(174, 214, 241)',
     'storm':'rgb(174, 214, 241)',
     'earthquake':'rgb(174, 214, 241)',
     'other_weather': 'rgb(174, 214, 241)',
          
     'aid_related':'yellow',
     'other_aid':'yellow',
     'aid_centers':'yellow',
        
     'related':'rgb(49,54,149)',
     'request':'rgb(49,54,149)',
     'offer':'rgb(49,54,149)',
     'search_and_rescue':'rgb(49,54,149)',
     'security':'rgb(49,54,149)',
     'military':'rgb(49,54,149)',
     'child_alone':'rgb(49,54,149)',
     'water':'rgb(49,54,149)',
     'food':'rgb(49,54,149)',
     'shelter':'rgb(49,54,149)',
     'clothing':'rgb(49,54,149)',
     'money':'rgb(49,54,149)', 
     'missing_people':'rgb(49,54,149)',
     'refugees':'rgb(49,54,149)',
     'death':'rgb(49,54,149)',
     'infrastructure_related':'rgb(49,54,149)',
     'transport':'rgb(49,54,149)',
     'buildings':'rgb(49,54,149)',
     'electricity':'rgb(49,54,149)',
     'tools':'rgb(49,54,149)',
     'shops':'rgb(49,54,149)',
     'other_infrastructure':'rgb(49,54,149)',
     'fire': 'rgb(49,54,149)',
     'cold': 'rgb(49,54,149)',
     'direct_report': 'rgb(49,54,149)'
         
    }

    # category counts and names 
    category_counts=df.drop(['id','message','original','genre'], axis=1).mean()
    category_names = list(category_counts.index)

    # getting viualization colors from dictionary d
    colors = [d[k] for k in category_names]

    

    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=aid_related_counts,
                    name = 'aid related',
                    marker = dict(
                        opacity=0.8,
                        color = 'yellow',
                        )
                ),
                Bar(
                    x = genre_names,
                    y = medical_help_counts,
                    name = 'medical help',
                    marker = dict(
                        opacity=1,
                        color='rgb(115, 198, 182)'
                        )

                ),
                Bar(
                    x = genre_names,
                    y = weather_counts,
                    name = 'weather related',
                    marker = dict(
                        opacity=1,
                        color='rgb(174, 214, 241)'
                        )

                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                 },
                'xaxis': {
                    'title': "Genre"
                 },
                'barmode' : 'group',
                'title_x' : 0.5
            }
        },

        {
            'data': [
                Bar(
                    x = category_names,
                    y = category_counts,
                    marker_color=colors
                    )
                ],
            'layout' : {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                 },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 40,
                    'dtick': 1
                    
                 },
                'title_pad_t': 0.9
                }
        }
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
