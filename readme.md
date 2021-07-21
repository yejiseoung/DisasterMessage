# Disaster Messages Pipeline Project
## by Yeji Zoe Seoung

## Proejct Description
This project is the part of Data Science Nanodegree Program by Udacity by using disaster messages provided by Figure Eight. The dataset contains labelled tweet and messages from real-life disaster events. 

The purpose of this project is to build a Natural Language Processing (NLP) model to classify disaster messages. The web app helps categorize the disaster messages into 36 categories. The web app provides emergency workes to input a new message and get classification results. The web app also displays visualizations of the data. 


## Dependencies
- Python 3.5+
- Machine Learning Libraries: Numpy, Pandas, Sciki-Learn
- NLP Libraries: NLTK 
- SQLite Database: SQLalchemy
- Model Load: Pickle
- Web App: Flask, Ploty


## Executing Program:
1. Run the following commends in terminal to clean data and store the cleaned data in the database

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`



2. Run the following commends to load data from DB, and save model as a pickle file

`python models/train_classifier.py data/disaster_response.db models/random_m1.pkl`


3. Run `python app/run.py`

4. Go to http://0.0.0.0:3001/


## Additional Material
1. **ETL_Pipleline_Preparation.ipynb** shows how to clean data and save it as sql file.
2. **ML_Pipeline_Preparation.ipynb** shows how to build a NLP model by using pipeline and train and test the model


## License
This is part of Data Science Nanodegree Udacity program project. Code templates were provided by Udacity and data were provided by Figure Eight.
