import praw
from textblob import TextBlob
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up Reddit API client using environment variables
reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'),
                     client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                     user_agent=os.getenv('REDDIT_USER_AGENT'))

# Choose subreddit
subreddit = reddit.subreddit("stocks")

# List of stocks you want to track (you can add more stocks)
tracked_stocks = ["Tesla", "Apple", "Rivian", "Nvidia"]

# Function to perform sentiment analysis
def get_sentiment(text):
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Sentiment polarity range: -1 (negative) to 1 (positive)
    
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Open a CSV file to write the data
with open('reddit_posts.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Content", "Sentiment", "Stocks Mentioned", "Post Length", "Stock Mentions"])  # Write header row

    # Loop through the top 5000 hot posts and write both the title and content to CSV
    for post in subreddit.hot(limit=5000):  # Get top 5000 hot posts
        title = post.title
        content = post.selftext

        # Sentiment analysis on the content
        sentiment = get_sentiment(content)

        # Check if any tracked stocks are mentioned in the content
        mentioned_stocks = [stock for stock in tracked_stocks if stock.lower() in content.lower()]

        # Calculate post length (number of characters)
        post_length = len(content)

        # Write the data to the CSV file
        writer.writerow([title, content, sentiment, ', '.join(mentioned_stocks), post_length, len(mentioned_stocks)])

        # Print the results to the terminal for reference
        print(f"Title: {title}")
        print(f"Sentiment: {sentiment}")
        print(f"Stocks mentioned: {', '.join(mentioned_stocks)}")
        print("-" * 50)  # Separator for readability

# After collecting data and saving it to CSV, load it into pandas for cleaning and analysis
df = pd.read_csv('reddit_posts.csv')

# 1. Handle Missing Data
df.dropna(subset=['Content'], inplace=True)  # Drop rows with missing content
df.drop_duplicates(inplace=True)  # Remove duplicate posts

# Optionally save the cleaned data
df.to_csv('cleaned_reddit_posts.csv', index=False)

# 2. Feature Engineering: Count Stock Mentions
df['stock_mentions'] = df['Content'].apply(lambda x: sum([stock in x for stock in tracked_stocks]))

# 3. Define Target Labels: Based on Sentiment (Positive = 1, Negative = 0)
df['label'] = df['Sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)

# 4. Convert Sentiment to numeric values: Positive = 1, Negative = 0, Neutral = -1
df['Sentiment_numeric'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': -1})

# 5. Shuffle the data
df = shuffle(df, random_state=42)

# 6. Feature extraction: Apply CountVectorizer for text features (unigrams and bigrams)
vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigrams and bigrams
X_ngrams = vectorizer.fit_transform(df['Content'])
X = pd.concat([df[['Sentiment_numeric', 'stock_mentions']], pd.DataFrame(X_ngrams.toarray())], axis=1)

# Ensure that column names are strings
X.columns = X.columns.astype(str)

# 7. Define target variable 'y'
y = df['label']  # Target variable (sentiment label)

# 8. Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 9. Split the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 10. Train the Model (Logistic Regression with Regularization)
model = LogisticRegression(class_weight='balanced', solver='liblinear')
model.fit(X_train, y_train)  # Train the model with training data

# 11. Evaluate the Model
predictions = model.predict(X_test)  # Make predictions on the test data
print("Accuracy:", accuracy_score(y_test, predictions))  # Print the accuracy

# 12. Print detailed evaluation metrics (Precision, Recall, F1-score)
print(classification_report(y_test, predictions))

# 13. Perform Cross-Validation to get a better understanding of model's generalization
scores = cross_val_score(model, X_res, y_res, cv=10)  # 10-fold cross-validation
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

# 14. ROC-AUC Score for further evaluation
roc_auc = roc_auc_score(y_test, predictions)
print(f"ROC-AUC Score: {roc_auc}")

# 15. Optionally, visualize sentiment distribution
df['Sentiment'].value_counts().plot(kind='bar', title='Sentiment Distribution', color=['green', 'red', 'gray'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
