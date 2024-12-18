#%%
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    # Handling large numbers
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    # removing all the non word characters as we are going to check text similarity and it not make much sense
    pattern = re.compile('\W')
    

    x = re.sub(pattern, ' ', x)
    
    x = BeautifulSoup(x, "html.parser").get_text()
    x_words = x.split()
    stemmed = [porter.stem(words) for words in x_words]

    final_preprocessed_words = " ".join(stemmed)
    # removes tabs, carriage returns , from feeds and vertical tabs respectively
    return_seq = re.sub('r[\t\r\f\v]+', ' ', final_preprocessed_words).strip()
    return return_seq
# %%
