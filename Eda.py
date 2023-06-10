# Instagram Data Analysis#
# Author: Matthew Dolan

# Features:
# - Impressions: Number of impressions in a post (Reach)
# - From Home: Reach from home
# - From Hashtags: Reach from Hashtags
# - From Explore: Reach from Explore
# - From Other: Reach from other sources
# - Saves: Number of saves
# - Comments: Number of comments
# - Shares: Number of shares
# - Likes: Number of Likes
# - Profile Visits: Number of profile visits from the post
# - Follows: Number of Follows from the post
# - Caption: Caption of the post
# - Hashtags: Hashtags used in the post

import importlib
import re
import subprocess
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from flair.data import Sentence
from flair.models import TextClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor

# Function to install missing packages
from sklearn.model_selection import train_test_split


def install_packages(packages):
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"{package} is already imported.")
        except ImportError:
            print(f"{package} is not imported. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} has been successfully installed and imported.")


# List of packages necessary to be installed
package_list = ["numpy", "seaborn", "matplotlib", "matplotlib.pyplot", "ipython", "nltk"]
install_packages(package_list)


# Function to read CSV file
def read_csv(filepath, encodings=['utf-8', 'latin1', 'iso-8859-1']):
    try:
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        raise Exception(f"Unable to decode the file using the specified encodings: {', '.join(encodings)}")
    except FileNotFoundError:
        print("File not found. Please check the filepath.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


# Reading in file:
filepath = '/Users/mattdolan/Documents/pythondata/Instagram data.csv'
df = read_csv(filepath)

# Check if DataFrame is not None before displaying the first 5 lines
if df is not None:
    print(df.head(5))

# checking column names
col_names = print(list(df.columns.values))

# displaying all data for eda for clearer view of df
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df.head(5))

# function that provides a list of null and non-null counts per column and provides list of distinct values
# less insightful for caption as variance in responses
def analyze_dataframe(df):
    # Count null and non-null values for each column
    null_counts = df.isnull().sum()
    non_null_counts = df.notnull().sum()

    # Print count of null and non-null values for each column
    print("Null Counts:")
    print(null_counts)
    print("\nNon-Null Counts:")
    print(non_null_counts)

    # Print distinct values in each column
    print("\nDistinct Values:")
    for column in df.columns:
        distinct_values = df[column].unique()
        print(f"{column}: {distinct_values}")

# Call the function to analyze the DataFrame
analyze_dataframe(df)


#looking at location of engagements:
home = df["From Home"].sum()
hashtags = df["From Hashtags"].sum()
explore = df["From Explore"].sum()
other = df["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(df, values=values, names=labels,
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()



def lowercase_columns(df):
    df.columns = df.columns.str.lower()
    return df
#calling function to make column names lower-case
lowercase_columns(df)
print(df)

# List of columns to keep
columns_to_keep = ['impressions', 'saves', 'comments', 'shares', 'likes', 'profile visits', 'follows', 'caption',
                   'hashtags']
# Filter the DataFrame to keep only the specified columns
df2 = df.reindex(columns=columns_to_keep, fill_value=None)
# Print the resulting DataFrame
print(df2)

# function for data cleaning for qualitative columns
def prepare_text_column(column):
    # Lowercase the text
    column = column.str.lower()
    # Remove special characters and digits
    column = column.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Tokenize the text
    column = column.apply(lambda x: word_tokenize(x))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    column = column.apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    column = column.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Join the words back into a sentence
    column = column.apply(lambda x: ' '.join(x))

    return column
# using function on qualitative columns to clean for best analysis
df2['hashtags'] = prepare_text_column(df2['hashtags'])
df2['caption'] = prepare_text_column(df2['caption'])
df2.head()

##EDA

#generating bar chart coints for popular words used in each qualitative field
def generate_word_counts(column):
    # Combine all the text from the column into a single string
    text = ' '.join(column)

    # Split the text into individual words
    words = text.split()

    # Create a frequency distribution of the words
    word_counts = pd.Series(words).value_counts()

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    word_counts[:20].plot(kind='bar')
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Top 20 Word Counts')

generate_word_counts(df2['hashtags'])
generate_word_counts(df2['caption'])

# defining list of columns for value count
# profile_visits = associated with impresssions = removed -> could be different hypothesis if later used
# relationship between sentiment scores and profile visits e.g.
def plot_counts(df, col):
    plt.figure(figsize=(9, 9))
    sns.countplot(data=df, y=col,
                  order=df[col].value_counts().index[:50])
    Title = f'{col}'
    plt.title(Title, fontsize=15)
    plt.show()

# defining list of columns for value count
# profile_visits = associated with impresssions = removed -> could be different hypothesis if later used
# follows included as direct involvement
# relationship between sentiment scores and profile visits e.g.
# taken out 'saves', 'comments', 'shares', 'likes' - impressions gives us an overview already
# can drilldown for further analysis
# MD - how hashtags and captions play a role in user engagement?
# MD-  From an impressions (interactive perspective), profile visits and also number of followers
cols = ['impressions', 'profile visits', 'follows', 'caption', 'hashtags']

# producing count of different values in above defined columns
for col in cols:
    plot_counts(df2, col)

print(df2)



##Nlp Analysis
df_nlp = df2.copy()

print(df_nlp)

# defining columns we want to keep
cols_nlp = ['impressions', 'likes', 'follows', 'caption', 'hashtags']

# Filter the DataFrame to keep only the specified columns
df_nlp = df_nlp.reindex(columns=cols_nlp, fill_value=None)

print(df_nlp)

# wordcloud generation function
def generate_wordcloud(data_column):
    # Flatten the list of tokens into a single list
    flattened_tokens = [token for sublist in data_column for token in sublist]

    # Count the frequency of each word
    word_counts = Counter(flattened_tokens)

    # Generate the word cloud
    wordcloud = ' '.join(flattened_tokens)

    # Create a bar chart of the word frequencies
    plt.figure(figsize=(10, 6))
    plt.bar(word_counts.keys(), word_counts.values())
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency')
    plt.xticks(rotation=90)
    plt.show()

# generating wordcloud for two columns of interest
generate_wordcloud(df_nlp['caption'])
generate_wordcloud(df_nlp['hashtags'])

# Sentiment Analysis - Caption
# Load the sentiment classification model
sia = TextClassifier.load('en-sentiment')

def flair_prediction(column):
    sentiment_scores = []

    for text in column:
        sentence = Sentence(text)
        sia.predict(sentence)
        score = sentence.labels[0].score
        sentiment_scores.append(score)

    return sentiment_scores

# Caption analysis
# Apply flair_prediction function to the caption field and store the sentiment scores in a new column
# Extract the independent variable (X) and dependent variable (y)
df_nlp['sentiment_score_caption'] = flair_prediction(df_nlp['caption'])
y = df_nlp['impressions'].values
X = df_nlp['sentiment_score_caption'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
# plot regression
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Sentiment Score')
plt.ylabel('Impressions')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# correlation matrix of results for caption analysis touchpoint
df_nlp_2_caption = pd.DataFrame(df_nlp)
corr_matrix = df_nlp_2_caption.corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True)
plt.show()

# hashtag Regression Analysis
# Extract the independent variable (X) and dependent variable (y)
df_nlp['sentiment_score_hashtags'] = flair_prediction(df_nlp['hashtags'])
y = df_nlp['impressions'].values
X = df_nlp['sentiment_score_hashtags'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
# plot regression
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Sentiment Score')
plt.ylabel('Impressions')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


# correlation matrix of results for hashtag analysis touchpoint
df_nlp_2_hashtag = pd.DataFrame(df_nlp)
corr_matrix2 = df_nlp_2_hashtag .corr()
print(corr_matrix2)
sns.heatmap(corr_matrix2, annot=True)
plt.show()

#NLP Analysis v2




##Analysis - Machine Learning
##seeing how much engagement future posts would achieve based on various predictors

x = np.array(df[['likes', 'saves', 'comments', 'shares',
                   'profile visits', 'follows']])
y = np.array(df["impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=42)

model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)
print(score)

# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
predictions = model.predict(features)
print(features)
print(predictions)
