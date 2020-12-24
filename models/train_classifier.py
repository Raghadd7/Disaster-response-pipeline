import sys

from sqlalchemy import create_engine
import sqlite3
import pandas as pd


import re
import nltk
nltk.download(['punkt','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from functools import partial
import pickle


def load_data(database_filepath):
    """
    Description: load data from database 

    Arguments:
    database_filepath: database file path
    
    Return:
    X: messages  
    Y: categories 
    category_names: message categories names
    
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql_table('DisasterTable', conn)


    # assiging X, Y, and category_names values from dataframe
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns.tolist()

    return X, Y, category_names


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
    
def build_model():
    """
    Description: building the ExtraTree model with grid search 

    Arguments:
    None
    
    Return:
    cv: the model
    
    """

    # building pipeline
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=partial(tokenize))),
                ('tfidf', TfidfTransformer()),  
                ('clf', MultiOutputClassifier(ExtraTreesClassifier()))
])

    parameters = {
    'clf__estimator__max_depth': [50],
    'clf__estimator__n_estimators': [250],
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.8, 1.0),
    'vect__max_features': (None, 10000),
    'tfidf__use_idf': (True, False)
}

    # building the model with GridSearchCV 
    cv = GridSearchCV(pipeline, param_grid = parameters, cv=2, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description: evaluating the model  

    Arguments:
    model: the model from build_model()
    X_test: the X testing data
    Y_test: the Y testing data
    category_names: the category names 
    
    Return:
    None
    
    """

    # assigning the model's predictions to y_pred value 
    y_pred = model.predict(X_test)

    # evaluating the model using model's accuracy and classification report 
    class_report = classification_report(Y_test, y_pred, target_names=category_names, zero_division=0)
    print("The model's Accuracy:  ",(y_pred == Y_test).mean())
    print("Classification Report: \n",class_report)

def save_model(model, model_filepath):
    """
    Description: saving the model into pickle file 

    Arguments:
    model: the model from build_model()
    model_filepath: the model file path 
    
    Return:
    None
    
    """

    # saving the model into pickle file
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

