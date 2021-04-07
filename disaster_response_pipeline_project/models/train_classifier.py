import sys
# import libraries
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import numpy as np
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    '''
    this function loads data fromo SQL database
    Args: 
        database_filepath: string
    Returns:
        None
    '''
    # load data from database
    engine = create_engine(database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_msg_categories",engine)


    Y = df.drop(['id', 'message', 'original', 'genre', 'related'], axis=1
           ).values
    X = df['message'].values
    pass


def tokenize(text):
    '''
    this function processes the text data by tokenizing it
    Args:
        text: string
    Returns:
        clean_tokens: list
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    this function builds a machine learning pipeline
    Returns:
        pipeline: Pipeline
    '''
    pipeline_ada = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters_ada = {
            'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [50, 60, 70]
    }

    cva = GridSearchCV(pipeline_ada, parameters_ada)
    return cva


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    this function displays the evaluation of the model
    Args:
        model: model
        X_test: list
        Y_test: list
        category_names: string
    Returns: 
        None
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred))
    pass


def save_model(model, model_filepath):
    '''
    this function exports the model as a pickle file
    Args:
        model: model
        model_filepath: string
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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