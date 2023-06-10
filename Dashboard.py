import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from flair.data import Sentence
from flair.models import TextClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
import plotly.io as pio

import plotly.io as plt_io
import plotly.express as px

# create our custom_dark theme from the plotly_dark template
plt_io.templates["custom_dark"] = plt_io.templates["plotly_dark"]

plotly_template = pio.templates["plotly_dark"]
print (plotly_template)




# Function to install missing packages
# Assuming you have two visualizations: fig1 and fig2
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
print(df)

#Create the subplots with 1 row and 3 columns
fig = make_subplots(rows=2, cols=3)

#dropping nans
df = df.dropna()

def lowercase_columns(df):
    df.columns = df.columns.str.lower()
    return df

# calling function to make column names lower-case
lowercase_columns(df)
print(df)

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
df['hashtags'] = prepare_text_column(df['hashtags'])
df['caption'] = prepare_text_column(df['caption'])

def generate_word_counts(column, group_column=None):
    # Combine all the text from the column into a single string
    text = ' '.join(column)

    # Split the text into individual words
    words = text.split()

    # Create a frequency distribution of the words
    word_counts = pd.Series(words).value_counts()
    return word_counts

# Generate word counts for 'impressions' column, grouped by 'hashtags' column
word_counts1 = generate_word_counts(df['hashtags'], group_column=df['impressions'])
print(word_counts1)

# Create a bar chart
fig1 = px.bar(word_counts1, x=word_counts1.index, y=word_counts1.values)

# Update the layout of the chart
fig1.update_layout(
    title='Word Counts',
    xaxis=dict(title='Words'),
    yaxis=dict(title='Count')
)

# Generate word counts for 'impressions' column, grouped by 'hashtags' column
word_counts2 = generate_word_counts(df['caption'], group_column=df['impressions'])
print(word_counts2)

# Create a bar chart
fig2 = px.bar(word_counts2, x=word_counts2.index, y=word_counts2.values)
print(fig2)

# Update the layout of the chart
fig2.update_layout(
    title='Word Counts',
    xaxis=dict(title='Words'),
    yaxis=dict(title='Count')
)

# looking at location of engagements:
home = df["from home"].sum()
hashtags = df["from hashtags"].sum()
explore = df["from explore"].sum()
other = df["from other"].sum()

labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

##new
df_nlp = df.copy()

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
    #plt.show()

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

# Create subplots with 2 rows and 1 column
fig = make_subplots(rows=2, cols=3)

#Hashtag Regression Analysis
df_nlp['sentiment_score_hashtags'] = flair_prediction(df_nlp['hashtags'])
y = df_nlp['impressions'].values
X = df_nlp['sentiment_score_hashtags'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
# plot regression
fig5 = plt.scatter(X, y, color='blue', label='Actual')
#fig5 = plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Sentiment Score')
plt.ylabel('Impressions')
plt.title('Simple Linear Regression')
plt.legend()
#plt.show()

# Add traces to the subplot
#fig.add_trace(scatter_trace, row=2, col=1)
#fig.add_trace(regression_trace, row=2, col=1)

#formatting dashboard

#Create the subplots with 2 rows and 3 columns
fig = make_subplots(rows=2, cols=3)

# Add fig1 to the first subplot
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)

# Add fig2 to the second subplot
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)

#creating fig.3
fig3 = go.Scatter(x=X, y=y, mode='markers', marker=dict(color='blue'), name='Actual')
#regression_trace = go.Scatter(x=X, y=y_pred, mode='lines', line=dict(color='red'), name='Regression Line')
fig.add_trace(fig3, row=2, col=1)

# Data for the bar plot
labels = ['From Home', 'From Hashtags', 'From Explore', 'Other']
values = [home, hashtags, explore, other]

# Create the bar plot trace
bar_trace = go.Bar(x=values, y=labels, orientation='h')

# Add the bar plot trace to the figure
fig.add_trace(bar_trace, row=2, col=2)

# Update the layout
# Create menu bar
menu_bar = go.FigureWidget([go.Bar(x=['Home', 'Hashtags', 'Explore', 'Other'], y=[home, hashtags, explore, other],
                                  marker=dict(color=['#002633', '#005A80', '#0088B0', '#00B4CC']),
                                  textposition='auto', hoverinfo='none')])

# Update menu bar layout
menu_bar.update_layout(
    title='Menu',
    xaxis=dict(title='Menu Options'),
    yaxis=dict(title='Counts'),
    plot_bgcolor='#F8F8F8',
    paper_bgcolor='#F8F8F8',
    height=300,
    margin=dict(l=20, r=20, t=60, b=20),
)

# Create menu bar
menu_bar = go.FigureWidget([go.Bar(x=['Home', 'Hashtags', 'Explore', 'Other'], y=[home, hashtags, explore, other],
                                  marker=dict(color=['#002633', '#005A80', '#0088B0', '#00B4CC']),
                                  textposition='auto', hoverinfo='none')])

# Update menu bar layout
menu_bar.update_layout(
    title='Menu',
    xaxis=dict(title='Menu Options'),
    yaxis=dict(title='Counts'),
    #plot_bgcolor='#F8F8F8',
    #paper_bgcolor='#F8F8F8',
    height=300,
    margin=dict(l=20, r=20, t=0, b=0),
)

# Update the layout of the dashboard

# Update the layout
fig.update_layout(title="Dashboard",
    showlegend=False,
    grid=dict(rows=2, columns=3),
    width=900,
    height=800,
template = 'plotly_dark'
)


# Show the dashboard
fig.show()

