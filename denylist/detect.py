from nltk.tokenize import word_tokenize

with open("data.txt") as f:
    denylist = f.readlines()

denylist = [x.strip() for x in denylist]


def is_hate_speech(text):
    text_tokens = word_tokenize(text.lower())

    return not set(text_tokens).isdisjoint(denylist)
