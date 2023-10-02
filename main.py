"""Sentiment analyser  by Apex862"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# download vader lexicon (for sentiment analysis)
nltk.download('vader_lexicon')

def sentiment_analyzer(paragraph, threshold=0.5):
    """
    This function takes in a paragraph of text as input and returns its sentiment score.
    If the sentiment score's confidence value (previously called "compound") is below the threshold, the sentiment is classified as "neutral".
    The sentiment score is a scoreionary with 4 keys - 'neg', 'neu', 'pos', and 'confidence',
    each containing a float value between -1 and 1.
    """
    # initialize SentimentIntensityAnalyzer object
    sid = SentimentIntensityAnalyzer()
    # obtain sentiment score for paragraph
    sentiment_score = sid.polarity_scores(paragraph)

    print("Overall sentiment score dictionary is : ", sentiment_score)
    print("sentence was rated as ", sentiment_score['neg'] * 100, "% Negative")
    print("sentence was rated as ", sentiment_score['neu'] * 100, "% Neutral")
    print("sentence was rated as ", sentiment_score['pos'] * 100, "% Positive")
    print("Sentence Overall Rated As", end=" ")
    if sentiment_score['compound'] >= 0.05:
        print("Positive")
    elif sentiment_score['compound'] <= - 0.05:
        print("Negative")
    else:
        print("Neutral")

    # check if sentiment is confident enough to be reliable
    if sentiment_score['compound'] < threshold:
        sentiment = 'neutral'
    else:
        sentiment = 'positive' if sentiment_score['compound'] > 0 else 'negative'
    # rename the 'compound' key to 'confidence'
    sentiment_score['confidence'] = sentiment_score.pop('compound')
    sentiment_score['sentiment'] = sentiment
    return sentiment_score


# test function
paragraph = """
Guinea pigs are fascinating creatures that make wonderful pets. Did you know that guinea pigs are not actually pigs, nor are they from Guinea? In fact, they are a type of rodent that originated from the Andean region of South America. Despite their small size, guinea pigs are incredibly social animals that thrive in the company of others. Did you also know that they can weigh up to two pounds and can live up to seven years when well taken care of? They have a varied diet, are very vocal, and have a unique set of behaviors. Guinea pigs make great pets and are sure to bring joy and entertainment to any household lucky enough to have one.

"""
print(sentiment_analyzer(paragraph))
# expected output: {'neg': 0.136, 'neu': 0.607, 'pos': 0.257, 'confidence': 0.4588, 'sentiment': 'positive'}
