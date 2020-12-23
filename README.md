
# Disaster Response Pipeline Project

# Table of Contents

1. [Installation](#installation)
2. [Project Description](#project)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing and Acknowledgements](#licensing)

## Installation<a name="installation"></a>

The libraries used in this project are available in Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

To execute the app:

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```python
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and save it
        ```python
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run the web app.
    ```python
    python run.py
    ```

3. Go to http://0.0.0.0:3001/




## Project Description<a name='project'></a>

The Disaster Response Pipeline project analyze messages from tweets and text messages for disaster response by a web app that displays a message classification result in several categories from a new input message and a data visualization of the training data.

This tool is helpful for emergency workers to analyze the messages in real time and classify them into different categories for organizations to act accordingly.

## File Descriptions<a name="files"></a>

There are 3 folders in this repository, structered as follows:

- app
  - template
    - master.html  is the main page of web app
    - go.html  is the classification result page of web app
  - run.py  the Flask file that runs app

- data
  - disaster_categories.csv  the categories data to process
  - disaster_messages.csv  the messages data to process
  - process_data.py  the code to process the data
  - InsertDatabaseName.db   the database to save clean data to

- models
  - train_classifier.py  the code to build and train the message classifier
  - classifier.pkl  the saved model pickle file


## Results<a name="results"></a>

The outcome of this project is a web app that analyze message data for disaster response & dispalys a data visualization of the disaster data as shown in the following screen shots:
<br>
<p align="center">
  <img width="540" height="420" src="https://github.com/Raghadd7/Disaster-response-pipeline/blob/master/web%20app%20screen%20shot1.png">
</p>

<p align="center">
    <img width="570" height="350" src="https://github.com/Raghadd7/Disaster-response-pipeline/blob/master/web%20app%20screen%20shot2.png">
</p>

<br><br>

The web app also enables the user to enter a disaster message and then display the categories of the message as presented in the following screen shot:

<br><br>

<p align="center">
    <img width="540" height="450" src="https://github.com/Raghadd7/Disaster-response-pipeline/blob/master/web%20app%20screen%20shot3.png">
</p>

## Licensing and Acknowledgements<a name="licensing"></a>

The contents of this repository are covered under the [MIT License.](https://github.com/Raghadd7/Disaster-response-pipeline/blob/main/LICENSE)

This project is part of the Udacity Data Scientist Nanodegree program and the disaster data is provided by [Figure Eight (appen).](https://appen.com)
