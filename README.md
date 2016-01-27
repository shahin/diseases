# diseases

Train a classifier to identify disease-topic HTML pages from Wikipedia.

## Requirements

* Python 2.7

## Install

```bash
git clone https://github.com/shahin/diseases
cd diseases
pip install -e.
python -m textblob.download_corpora
```

## Run

To train a classifier:

```bash
python diseases/train.py -o my_classifier.pickle -p training/positive -n training/negative --positive-sample=100 --negative-sample=200
```

To use that classifier to classify an HTML file (and name the disease, if it is one):

```bash
python diseases/classify.py <path-to-unlabeled-dir> my_classifier.pickle -m
```

## Performance

The confusion matrix for the included classifier (nb_2000p_3000n.pickle) is:

| True Positives  | False Negatives |
|-----------------|-----------------|
|  97             | 3               |


| False Positives | True Negatives  |
|-----------------|-----------------|
| 3               | 297             |


Training this classifier on an unbalanced sample of 2000 positive training examples and 3000
negative training examples tool ~3h.

Applying this classifier to a 100-300 P-N test set took ~10 min.

## TODO

1. switch to bare NLTK (speed)
2. narrow features extracted from HTML scraping (accuracy, speed)
3. add cross-validation (accuracy)
4. attempt to improve decision trees (speed over NB, accuracy)
5. ...
