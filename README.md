Overview
This report outlines the exploratory data analysis (EDA) performed on Instagram post data. The main goal was to understand the different variables affecting the reach and engagement of Instagram posts.

The dataset contained the following columns:

Impressions: Number of impressions in a post (Reach)
From Home: Reach from home
From Hashtags: Reach from Hashtags
From Explore: Reach from Explore
From Other: Reach from other sources
Saves: Number of saves
Comments: Number of comments
Shares: Number of shares
Likes: Number of Likes
Profile Visits: Number of profile visits from the post
Follows: Number of Follows from the post
Caption: Caption of the post
Hashtags: Hashtags used in the post
Data Preparation
The data was read from a CSV file. A series of data preparation steps were carried out including checking for missing values and inspecting the unique values for each column.

The reach of posts from different sources was summarized in a pie chart. This visualization highlighted the distribution of impressions across various channels such as Home, Hashtags, Explore, and Other sources.

Column names were then converted to lower case for consistency and a subset of the columns was selected for further analysis.

Text columns (caption and hashtags) underwent additional preprocessing including:

Conversion to lower case
Removal of special characters and digits
Tokenization
Stop word removal
Lemmatization
Exploratory Data Analysis (EDA)
As part of EDA, frequency distributions of words in hashtags and caption fields were plotted. This provided insights into the most common words used in captions and hashtags.

The count of different values in the 'impressions', 'profile visits', 'follows', 'caption', and 'hashtags' columns were visualized, revealing the most popular words and engagement metrics.

Natural Language Processing (NLP) Analysis
Further, NLP analysis was conducted on the caption and hashtags fields. Word clouds were generated for each field, emphasizing the most frequently used words.

Sentiment analysis was conducted on the caption and hashtags fields using the Flair Text Classifier. The sentiment scores were then correlated with impressions using a simple linear regression model. This allowed for an examination of how sentiment scores might affect post impressions.

Machine Learning Analysis
Finally, a machine learning model was trained using a subset of the features (likes, saves, comments, shares, profile visits, follows) to predict the impressions of a post. The model used was a Passive Aggressive Regressor, which was trained on 80% of the data and tested on the remaining 20%.

The performance of the model was evaluated, and it was also used to predict the impressions for a hypothetical post given certain feature values.

Conclusion
Through this exploratory data analysis and machine learning, it is possible to gain insights into which factors contribute most to the reach and engagement of Instagram posts. These insights can help in strategizing content creation and post scheduling to maximize reach and engagement.

Further work could involve tuning the machine learning model and exploring additional features for prediction. Additionally, the impact of sentiment in captions and hashtags on post performance could be further investigated.
