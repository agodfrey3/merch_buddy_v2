from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder


class SKLearnMultiLabel(object):

    def __init__(self, clf, name=None):
        self.clf = clf
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        self.model = make_pipeline(self.vectorizer, self.clf)
        self.label_binarizer = LabelEncoder()
        self.name = name

    def fit(self, X, y):
        y_binary = self.label_binarizer.fit_transform(y)
        self.model.fit(X, y_binary)

    def score(self, X, y):
        y_binary = self.label_binarizer.transform(y)

        return self.model.score(X, y_binary)

    def predict(self, X, convert=False, proba=False):
        if proba:
            preds = self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)

        if convert:
            return self.label_binarizer.inverse_transform(preds)
        else:
            return preds

    def convert_labels(self, labels):
        return self.label_binarizer.inverse_transform(labels)

    def transform_labels(self, labels):
        return self.label_binarizer.transform(labels)
