import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Description: load messages and categories data files

    Arguments:
    messages_filepath: messages data file path
    categories_filepath: categories data file path
    
    Return:
    df: the merged dataframe 
    
    """

    # read messages and categories data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge messages and categories data on id 
    df = messages.merge(categories, on='id')

    return df 

def clean_data(df):
    """
    Description: clean the dataframe

    Arguments:
    df: dataframe
    
    Return:
    df: the cleaned dataframe
    
    """
    # splitting the values in the categories dataframe 
    categories = df.categories.str.split(';', expand =True)

    # use the first row of categories dataframe to create column names 
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')


    # replace categories column in df with new category columns 
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1, sort=False)

    # remove duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    Description: save the clean dataset into an sqlite database

    Arguments:
    df: dataframe
    database_filename: the file path to save the database file
    
    Return:
    None
    
    """

    # save the clean dataset into an sqlite database 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterTable', engine, index=False, if_exists='replace')  


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

