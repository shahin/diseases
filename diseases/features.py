"""Functions useful for building features from a Wikipedia HTML file."""

from bs4 import BeautifulSoup
from textblob import TextBlob

def parse_html(html_file):
    """Return a BeautifulSoup object for the given HTML file handle."""
    contents = html_file.read()
    return BeautifulSoup(contents, 'html.parser')

def parse_paragraphs(html_soup):
    """Given a BeautifulSoup object, return a list of text from each paragraph."""
    paragraphs = html_soup.findAll('p')
    paragraph_text = [p.get_text() for p in paragraphs]
    return paragraph_text

def get_important_phrases(soup):
    """Given a BeautifulSoup object, return a set of important phrases.
    
    Currently, only noun phrases from the first three paragraphs are considered important.
    """

    paragraphs = parse_paragraphs(soup)
    paragraph_noun_phrases = [TextBlob(p).noun_phrases for p in paragraphs]

    all_noun_phrases = []
    for idx in range(3):
        try:
            all_noun_phrases += paragraph_noun_phrases[idx]
        except IndexError:
            break

    return set(all_noun_phrases)

def get_page_features(html_soup):  
    """Given a BeautifulSoup object, return a set of features based on page structure.
    
    Currently, only (1) existence of a 'Classification and external resources' infobox and (2) an
    ICD9 element in the infobox are considered important.
    """
    
    infobox = html_soup.findAll('table', {'class': 'infobox'})

    if len(infobox) > 0:

        has_classification_infobox = \
                len(infobox[0].findAll('th', text = 'Classification and external resources')) > 0
        has_infobox_icd9 = \
                len(infobox[0].findAll('a', text = 'ICD9')) > 0

    else:
        has_infobox_icd9 = False
        has_classification_infobox = False

    features = []
    if has_classification_infobox:
        # this feature has very high precision on the training set
        features.append('_HAS_CLASSIFICATION_INFOBOX_')
    if has_infobox_icd9:
        features.append('_HAS_INFOBOX_ICD9_')

    return features
