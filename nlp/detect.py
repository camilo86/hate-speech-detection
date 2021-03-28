from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def is_hate_speech(text):
    vs = analyzer.polarity_scores(text)

    return vs['compound'] <= -0.05
