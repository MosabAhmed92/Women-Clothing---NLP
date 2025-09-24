import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def to_lower(text):
    """
    Convert a text to lower Case
    """
    return text.lower()


def handle_negation(text: str) -> str:
    """
    Merge short negations with the following token so their polarity is preserved.
    Examples:
      "not good"      -> "not_good"
      "I don't like"  -> "I do not_like"
      "no complaints" -> "no_complaints"
      "never again"   -> "never_again"
    """
    # 1) Normalize contractions so n't becomes a separate token (don't -> do n't, isn’t -> is n’t)
    text = re.sub(r"n['’]t\b", " n't", text)

    NEGATORS = {"not", "no", "never", "n't"}
    tokens = text.split()

    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        tok_lc = to_lower(tok)

        # Is there a next token?
        if tok_lc in NEGATORS and (i + 1) < len(tokens):
            nxt = tokens[i + 1]

            # Merge only if the next token looks like a word (avoid punctuation like ',' or '!')
            if re.match(r"^\w+$", nxt):
                out.append("not_" + nxt)
                i += 2  # skip the consumed next token
                continue

            # Otherwise keep the negator as-is
            out.append(tok)
            i += 1
        else:
            out.append(tok)
            i += 1

    return " ".join(out)


def strip_punctuation(text):
    return "".join([ch for ch in text if ch not in string.punctuation])


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def lemmatize(text):
    words = text.split()
    tagged_tokens = pos_tag(words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for word, tag in tagged_tokens:
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        elif tag.startswith("JJ"):
            pos = "a"
        elif tag.startswith("R"):
            pos = "r"
        else:
            pos = "n"  # Default to noun if no POS tag is found
        lemmatized_words.append(lemmatizer.lemmatize(word, pos))
    return " ".join(lemmatized_words)



def preprocess_text(text, cfg):
    if cfg['preprocessing']['lowercase']:
        text = to_lower(text)
    if cfg['preprocessing']['strip_punctuation']:
        text = strip_punctuation(text)
    if cfg['preprocessing']['remove_numbers']:
        text = remove_numbers(text)
    if cfg['preprocessing']['handle_negations']:
        text = handle_negation(text)
    if cfg['preprocessing']['stopwords'] == 'english':
        STOPWORDS = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in STOPWORDS])    
    if cfg['preprocessing']['lemmatize']:
        text = lemmatize (text)
    return text

