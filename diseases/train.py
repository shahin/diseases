"""Train a classifier and save it to a file.

Usage:
    train.py [--positive-sample=<n>] [--negative-sample=<m>] [--train-fraction=0.6] [-p DIR] [-n DIR] [-o FILE]

Arguments:

Options:
    -h --help               Show this message
    -p DIR                  path to the directory containing HTML pages of diseases [default: training/negative]
    -n DIR                  path to the directory containing HTML pages of non-diseases [default: training/negative]
    -o FILE                 path of the file to save the pickled trained model to [default: cl.pickle]
    --positive-sample=<n>   number of positive examples to take [default: all]
    --negative-sample=<m>   number of negative examples to take [default: all]
    --train-fraction=0.6    percent of sample to use to train [default: 0.6], complement is used for test
"""

import os
import random
import pickle
import multiprocessing
from docopt import docopt

from textblob.classifiers import NaiveBayesClassifier
from nltk.metrics import ConfusionMatrix

from diseases.features import get_page_features, get_important_phrases, parse_html

import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def get_files(dir_path):
    """Return a list of file paths in the given directory path."""
    return [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, f))
    ]

def get_feature_string(file_name):
    """Given an HTML file name, return a string of feature tokens."""
    with open(file_name) as f:
        soup = parse_html(f)

    page_features = get_page_features(soup)
    paragraph_features = get_important_phrases(soup)

    document = ' '.join(['; '.join(page_features), '; '.join(paragraph_features)])
    return document

def get_pos_record(file_name):
    return (get_feature_string(file_name), 'pos', )

def get_neg_record(file_name):
    return (get_feature_string(file_name), 'neg', )

def train(pos_examples, neg_examples, train_fraction=0.6):
    """Train a classifier, holding out train_fraction of pos_examples and neg_examples as a test set.
    Return the tuple:
        
        (the classifier, accuracy, positive test example list, negative test example list, )

    """

    pos_split = int(train_fraction * len(pos_examples))
    pos_train, pos_test = pos_examples[0:pos_split], pos_examples[pos_split:]
    neg_split = int(train_fraction * len(neg_examples))
    neg_train, neg_test = neg_examples[0:neg_split], neg_examples[neg_split:]

    cl = NaiveBayesClassifier(pos_train + neg_train)
    return cl, cl.accuracy(pos_test + neg_test), pos_test, neg_test

def dump_classifier(classifier, path):
    with open(path, 'w') as dump_file:
        # delete lots of unneeded data after training, before dump
        classifier.train_features = []
        pickle.dump(classifier, dump_file)

def load_classifier(path):
    with open(path, 'r') as load_file:
        return pickle.load(load_file)

def main(positive_dir, negative_dir, output_path, n_pos_sample=None, n_neg_sample=None, train_faction=None):

    n_cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(int(n_cpus * 0.7))

    pos_files = get_files(positive_dir)
    neg_files = get_files(negative_dir)

    pos_sample = random.sample(pos_files, n_pos_sample) if n_pos_sample is not None else pos_files
    neg_sample = random.sample(neg_files, n_neg_sample) if n_neg_sample is not None else neg_files

    LOG.info("Building features on {0} positive examples and {1} negative examples ...".format(
        len(pos_sample), len(neg_sample)
    ))

    pos_examples = pool.map(get_pos_record, pos_sample)
    neg_examples = pool.map(get_neg_record, neg_sample)

    cl, accuracy, _, _ = train(pos_examples, neg_examples, train_fraction)
    LOG.info("Trained accuracy on test set: {}".format(accuracy))
    dump_classifier(cl, output_path)
    
if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    n_pos_sample = int(args['--positive-sample']) if args['--positive-sample'] != 'all' else None
    n_neg_sample = int(args['--negative-sample']) if args['--negative-sample'] != 'all' else None
    train_fraction = float(args['--train-fraction'])

    main(args['-p'], args['-n'], args['-o'], n_pos_sample, n_neg_sample, train_fraction)

