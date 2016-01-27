"""Determine whether a Wikipedia HTML file describes a disease, and extract the article title.

Usage:
    classify.py [-m] EXAMPLE_DIR CLASSIFIER_PATH

Arguments:
    EXAMPLE_DIR     the path to the Wikipedia HTML file to inspect
    CLASSIFIER_PATH the path to the pickled disease-page classification model

Options:
    -h --help       Show this message
    -m              Extract the article title (name of the disease, if it is a disease) and output it
"""

import os
import multiprocessing
from docopt import docopt

from features import parse_html
from diseases.train import load_classifier, get_feature_string, get_files

from textblob.classifiers import DecisionTreeClassifier

def get_disease_name(html_file_path):
    """Return the name of the disease described in the given HTML file."""
    with open(html_file_path) as f:
        html = parse_html(f)
        page_title = html.find('title').get_text()

        # page title usually begins with disease name and ends with an extra 35 characters
        # e.g. "Abdominal aortic aneurysm - Wikipedia, the free encyclopedia"
        return page_title[:-35]

def classify(example_dir_path, classifier_path):
    example_files = get_files(example_dir_path)

    n_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(int(n_cpus * 0.7))
    examples = pool.map(get_feature_string, example_files)

    cl = load_classifier(classifier_path)
    example_features = [cl.extract_features(e) for e in examples]

    predicted_labels = cl.classifier.classify_many(example_features)
    names = [get_disease_name(f) for f in example_files]
    return predicted_labels, names

if __name__ == '__main__':
    args = docopt(__doc__)
    results, names = classify(args['EXAMPLE_DIR'], args['CLASSIFIER_PATH'])
    print(results)

    if args['-m']:
        print(names)
