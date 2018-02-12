import pandas as pd
import numpy as np
import os
from models import model_dict
from ClassifierWrappers import SKLearnMultiLabel
from sklearn import metrics
import json
from file_helpers import save_pkl_file


def build_name_and_descr_corpus(dir, verbose=0):
    """
    Returns a DataFrame consisting of the unique item_names
    and their descriptions found within the passed directory.
    Note: Currently only parses csv files.
    :param dir: Directory of files to be parsed into a df.
    :param verbose: Verbosity of log messages. verbose > 0 will
                    give updates throughout the function.
    :return: A DataFrame consisting of the unique item_names and their
             descriptions found in the passed directory.
    """
    df = pd.DataFrame()

    for file in os.listdir(dir):
        if file.endswith(".csv"):
            if verbose > 0:
                print(f"Creating df from {file}...")
            file_path = os.path.join(dir, file)
            temp_df = pd.read_csv(file_path, encoding='utf-8')

            df = df.append(temp_df)

    if verbose > 0:
        print(f"Dropping duplicate rows and saving to new df...")
    final_df = df[['item_name', 'item_desc']].copy()
    final_df = final_df.dropna()
    final_df = final_df.drop_duplicates(subset=['item_name'])

    return final_df


def split_train_test(df, percent=0.8):
    df = df.sample(frac=1).reset_index(drop=True)
    num_rows = len(df)

    split_index = int(percent * num_rows)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index + 1:]

    return train_df, test_df


def order_labels(labels, model):
    transformed = model.transform_labels(np.asarray([labels]).T)
    trans_labels = zip(labels, transformed)

    ordered_labels = []
    for label, vector in trans_labels:
        idx = np.argmax(vector)
        ordered_labels.append((label, idx))
    ordered_labels.sort(key=lambda x: x[1])
    labels, _ = list(zip(*ordered_labels))
    labels = list(labels)

    return labels


def calculate_blank_stats(test_df, model):
    test_df['new_predictions'] = model.predict(test_df['X'].values, convert=True)
    num_blank = 0
    num_rows = len(test_df)

    for i, row in test_df.iterrows():
        tags = row['new_predictions']
        if len(tags) < 1:
            num_blank += 1

    percent_blank = num_blank / num_rows

    blank_stats = {
        'num_blank': num_blank,
        'percent_blank': percent_blank
    }

    return blank_stats



def model_metrics(clf, test_df, ordered_labels):
    y_true = test_df['label'].values
    y_true = clf.transform_labels(y_true)
    y_pred = clf.predict(test_df['X'].values)

    blank_prediction_stats = calculate_blank_stats(test_df, clf)

    m_metrics = {
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'Precision': metrics.precision_score(y_true, y_pred, average='micro'),
        'Recall': metrics.recall_score(y_true, y_pred, average='micro'),
        'F1 Score': metrics.f1_score(y_true, y_pred, average='micro'),
        'Blank Predictions': blank_prediction_stats['num_blank'],
        'Blank Percentage': blank_prediction_stats['percent_blank']

    }

    print(json.dumps(m_metrics, indent=4))

    #TODO Utilize this functionality
    metrics_per_class = metrics.classification_report(y_true, y_pred, target_names=ordered_labels)

    return m_metrics


def evaluate_and_train_model(model, train_df, test_df):
    model.fit(X=train_df['X'].values, y=train_df['label'].values)

    model_metric = model_metrics(model, test_df)

    evaluation = {
        'model': model,
        'metrics': model_metric
    }

    return evaluation



def train_models(train_df, test_df=None, models=None, save_dir=None):
    if train_df is None:
        raise ValueError

    if test_df is None:
        train_df, test_df = split_train_test(train_df)

    if models is None:
        models = model_dict.keys()

    for model in models:
        print(f"Setting up {model}...")
        clf = model_dict[f'{model}']
        model = SKLearnMultiLabel(clf=clf, name=model)

        print(f"Beginning training for {model.name}...")
        clf_eval = evaluate_and_train_model(model=model, train_df=train_df, test_df=test_df)

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{model.name}.pkl")
        else:
            save_path = f"c:/users/god/data/rs_data/{model.name}.pkl"

        print(f"Done training model {model.name}...")
        print(f"Saving model to {save_path}...")
        save_pkl_file(clf_eval, save_path)


def main():
    main_dir = "c:/users/god/data/rs_data/"
    load_dir = os.path.join(main_dir, "data_sets")
    save_dir = os.path.join(main_dir, "evaluations")
    load_path = os.path.join(load_dir, "name_label.csv")

    print(f"Reading csv from {load_path}")
    df = pd.read_csv(load_path, encoding='utf-8')

    print("Entering training function...")
    train_models(train_df=df, save_dir=save_dir)



if __name__ == '__main__':
    main()