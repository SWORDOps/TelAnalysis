import json
import re
import string
import collections
import nltk
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from utils import remove_chars_from_text, remove_emojis, read_conf
import stopwords_list

# Initializing the stemmer
stemmer = SnowballStemmer("russian")

# Special characters for cleaning text
spec_chars = string.punctuation + '\n\xa0«»\t—…"<>?!.,;:꧁@#$%^&*()_-+=№%༺༺\༺/༺-•'

# Actions to ignore in the text
action_map = ['Invite Member', 'Kicked Members', 'Joined by Link', 'Pinned Message']

def analyse(data, most_com):
    """
    Analyzes a given text data (expected to be a string or list of strings representing messages)
    to find the most common words.
    It performs text cleaning, tokenization, stemming (optional), stop word removal,
    and frequency distribution.
    This version applies an action_map to remove certain phrases and filters tokens with length >= 3.

    Args:
        data: Input text data (e.g., a string from a single user's messages).
        most_com: The number of most common words to return.

    Returns:
        A tuple containing:
            - fdist (list): A list of [word, count] for the most common words.
            - text_tokens (list): A list of all processed and filtered tokens from the input.
    """
    # Setting up stop words
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(['это', 'ну', 'но', 'еще', 'ещё', 'оно', 'типа'])
    english_stopwords = stopwords.words("english")

    # Converting text to lowercase and removing extra characters
    text = str(data).lower().replace("'", "").replace(",", "").replace("[", "").replace("]", "").replace("-", " ")
    
    for action in action_map:
        text = text.replace(action.lower(), "")
    
    text = remove_chars_from_text(text, spec_chars)
    text = remove_chars_from_text(text, string.digits)

    # Checking that the text is not empty
    if len(text) < 1:
        return [], []

    # Tokenizing the text
    text_tokens = word_tokenize(text)

    # Stemming tokens, if selected
    if read_conf('select_type_stem') == 'On':
        text_tokens = [stemmer.stem(word) for word in text_tokens]

    # Filtering tokens
    text_tokens = [token.strip() for token in text_tokens if 
                   token not in russian_stopwords and 
                   len(token) >= 3 and 
                   len(token) < 26 and 
                   token not in english_stopwords and 
                   'http' not in token and 
                   token not in stopwords_list.stopword_txt]

    # Frequency distribution
    text = nltk.Text(text_tokens)
    fdist = FreqDist(text)
    fdist = fdist.most_common(most_com)

    return fdist, text_tokens

def analyse_all(data, most_com):
    """
    Analyzes a given text data (potentially a list of pre-processed tokens or a large text corpus)
    to find the most common words.
    It performs text cleaning, tokenization, stemming (optional), stop word removal,
    and frequency distribution.
    This version filters tokens with length >= 4 and its second return value is a list of words
    from the frequency distribution (not all processed tokens). It does not use the action_map.

    Args:
        data: Input text data (e.g., a list of all tokens from a chat, or a channel's entire text).
        most_com: The number of most common words to return.

    Returns:
        A tuple containing:
            - fdist (list): A list of [word, count] for the most common words.
            - words_from_fdist (list): A list of words [word] from the fdist.
    """
    # Setting up stop words
    russian_stopwords = stopwords.words("russian")
    english_stopwords = stopwords.words("english")
    russian_stopwords.extend(['это', 'ну', 'но', 'еще', 'ещё', 'оно', 'типа'])

    # Converting text to lowercase and removing extra characters
    text = str(data).lower().replace("'", "").replace(",", "").replace("[", "").replace("]", "").replace("-", " ")
    text = remove_chars_from_text(text, spec_chars)
    text = remove_chars_from_text(text, string.digits)
    #text = remove_emojis(text)

    if len(text) >= 1:
        text_tokens = word_tokenize(text)
    else:
        return [], []

    # Stemming tokens, if selected
    if read_conf('select_type_stem') == 'On':
        text_tokens = [stemmer.stem(word) for word in text_tokens]

    # Filtering tokens
    text_tokens = [token.strip() for token in text_tokens if 
                   token not in russian_stopwords and 
                   len(token) >= 4 and 
                   len(token) < 26 and 
                   token not in english_stopwords and 
                   'http' not in token and 
                   token not in stopwords_list.stopword_txt]

    # Frequency distribution
    text = nltk.Text(text_tokens)
    fdist = FreqDist(text)
    fdist = fdist.most_common(most_com)

    words_from_fdist = [i[0] for i in fdist]
    return fdist, words_from_fdist
