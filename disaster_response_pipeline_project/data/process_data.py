import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    this function loads message and category dataframes from csv files and 
    returns a merged df, merging by 'id' column
    Args: 
        messages_filepath: string
        categories_filepath: string
    Returns:
        pandas.df
    '''
    messages = pd.read_csv('messages_filepath')
    categories = pd.read_csv('categories_filepath')
    df=messages.merge(categories, how='left', on=['id'])
    return df


def clean_data(df):
    '''
    this function cleans the dataframe by splitting the df's category column
    into category columns and removes duplicates in the df
    Args: 
        df: pandas.df
    Returns:
        pandas.df
    '''
    categories = df['categories'].str.split(';' , expand=True)
    row = categories.iloc[0]
    liste=[]
    #extract a list of new column names for categories
    for i in range(len(row)):
        liste.append(row[i][:(len(row[i])-2)] )   
    category_colnames = liste
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = pd.to_numeric(categories[column].astype(str).str.strip().str[-1])
    #concat
    df_no_old_categories=df.drop(['categories'], axis=1)
    df=pd.concat([df_no_old_categories, categories], axis=1)
    #remove duplicates
    df=df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    this function saves the clean dataset into an sqlite database
    Args: 
        df: pandas.df
        database_filename: string
    Returns:
        None
    '''
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql(database_filename, engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()