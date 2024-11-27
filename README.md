# Stock Prediction Using Sentiment Analysis from Reddit Posts

## Project Overview
This project focuses on predicting stock market trends using **sentiment analysis** of Reddit posts, specifically from the **r/Stocks** subreddit. By analyzing the sentiment around specific stocks mentioned in Reddit discussions, the model predicts whether a stock's value is likely to increase (positive sentiment) or decrease (negative sentiment). This can provide valuable insights to investors by detecting market sentiment and trends early.

The model leverages **Natural Language Processing (NLP)** techniques to analyze the textual content of Reddit posts and classify the sentiment. It then uses machine learning to predict stock price movements based on this sentiment and stock mentions.

### Key Features:
- **Reddit Scraping**: Collects recent posts from the r/Stocks subreddit.
- **Sentiment Analysis**: Classifies the sentiment of posts (positive, negative, or neutral) using **TextBlob**.
- **Stock Mentions**: Identifies which tracked stocks are mentioned in the posts.
- **Machine Learning Model**: Trains a **Logistic Regression** model to predict stock movements based on sentiment and stock mentions.
- **Cross-Validation**: Evaluates the model using cross-validation to ensure it generalizes well on unseen data.

## Setup Instructions

### Prerequisites:
- **Python 3.x** (preferably the latest version)
- **Pip** for installing dependencies
- **Reddit API Credentials** (client_id, client_secret, user_agent)

### Steps to Set Up the Project:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd stockPrediction

2. **Create a Virtual Environment**:    
   ```bash
    python -m venv venv
    
3. **Activate the Virtual Environment: On Windows**:
   ```bash
   .\venv\Scripts\activate
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate

4.**Install Dependencies: Install the required libraries using pip**:
bash
pip install -r requirements.txt

You can create a requirements.txt with the following contents:
praw
textblob
pandas
scikit-learn
imbalanced-learn
matplotlib

5. **Reddit API Setup**:

> Visit Reddit's Developer Portal to create an application and get your client_id, client_secret, and user_agent.
> Add your credentials in the scrape-reddit.py file where the praw.Reddit() object is initialized.

6. **Run the Script: After setting everything up, you can run the script**:

bash
python scrape-reddit.py

This will collect data from the r/Stocks subreddit, perform sentiment analysis on the posts, and save the results to a CSV file.

**Usage**
After running the script, the model trains and evaluates on the sentiment of Reddit posts. You can:

. View the Sentiment Distribution: A bar chart showing the distribution of positive, negative, and neutral posts.
. Model Accuracy: The script prints the accuracy of the model based on test data, along with other metrics like precision, recall, and f1-score.
. Cross-Validation: The model is evaluated using 10-fold cross-validation to check its robustness.
. You can also extend the stock list by modifying the tracked_stocks list in the scrape-reddit.py file to track additional stocks.

**Model Evaluation**
Performance Metrics:
After training the machine learning model, the following evaluation metrics were used to assess its performance:

. Accuracy: Measures the percentage of correct predictions.
Precision: The percentage of relevant results out of all retrieved results.
. Recall: The percentage of relevant results retrieved out of all possible relevant results.
. F1-Score: The harmonic mean of precision and recall, balancing both metrics.
. ROC-AUC: Evaluates the ability of the model to distinguish between classes.

**Model Evaluation Results**:
. Accuracy: The model achieved an accuracy of approximately 86%.
. Precision, Recall, F1-Score: The model performed well with balanced precision and recall scores for both positive and negative sentiment.
. Cross-Validation: The cross-validation scores range from 63% to 96%, indicating good generalization on unseen data.

**Example Sentiment Predictions**:
. Title: "Possible scam?" - Sentiment: Positive
. Title: "Cisco reports fourth straight quarter of declining revenue" - Sentiment: Negative
. Title: "TSMC Predictions" - Sentiment: Positive

**Future Improvements**
. Data Augmentation: The model can be improved by collecting more data, possibly from different subreddits or news sources.
. Fine-Tuning the Model: Experiment with advanced models like Random Forest, XGBoost, or LSTM to improve prediction accuracy.
. Real-Time Predictions: Implement a system for real-time Reddit scraping and sentiment analysis.
. Better Feature Engineering: Consider adding features such as post engagement metrics (e.g., upvotes, comments).

Feel free to fork this repository, contribute to the project, or use it for your own stock prediction applications!

**Important Notes**:
. Make sure you replace the Reddit API credentials with your own before running the script.
. This project is designed for educational purposes and demonstrates the use of sentiment analysis for stock predictions, but its accuracy and utility for real-world financial trading should be verified further.