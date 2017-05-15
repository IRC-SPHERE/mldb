import pandas as pd
import numpy as np

from os.path import exists
import json

from sklearn.base import clone

from models import *
from features import load_training_data
from utils import get_or_create, insert_predictions

pd.set_option('display.width', 1000)


def get_inds(df, folds):
    return df.group.apply(lambda gg: gg in set(folds))


def train_validation_test_inds(df, train_folds, validation_folds, test_folds):
    train_inds = get_inds(df, train_folds)
    validation_inds = get_inds(df, validation_folds)
    test_inds = get_inds(df, test_folds)

    return train_inds, validation_inds, test_inds


class DBClassifier(object):
    def __init__(self, connection, task_name, partition_name, feature_union_name, force):
        self.force = force
        self.connection = connection
        self.partition = connection.partitions[task_name][partition_name]
        self.feature_union, self.task, self.df = load_training_data(
            connection=connection,
            partition_name=partition_name,
            task=connection.tasks[task_name],
            feature_union_id=feature_union_name
        )

    """
    
    """

    def fit(self, classifier, classifier_name):
        for split in self.partition.splits:
            self.fit_split(
                classifier=classifier,
                classifier_name=classifier_name,
                split=split
            )

    def fit_split(self, classifier, classifier_name, split):
        train_inds, validation_inds, test_inds = train_validation_test_inds(
            df=self.df,
            train_folds=split.train_folds,
            validation_folds=split.validation_folds,
            test_folds=split.test_folds
        )

        self.df.loc[train_inds, 'kind'] = 'train'
        self.df.loc[test_inds, 'kind'] = 'test'
        self.df.loc[validation_inds, 'kind'] = 'validation'

        # Acquire training data
        X_train, y_train = self.get_training_data(
            X=self.df.x[train_inds + validation_inds],
            y=self.df.y[train_inds + validation_inds]
        )

        # Learn the model
        classifier_ = clone(classifier)
        classifier_.fit(X_train, y_train)

        p = classifier_.predict_proba(map(list, self.df.x))
        self.df['p'] = map(json.dumps, map(list, p))
        self.df['p'] = self.df.p.apply(json.loads)
        self.save_predictions(classifier_name, split)

        del self.df['p']
        del self.df['kind']

    def save_predictions(self, classifier_name, split):
        # Save predictions to the mldb
        classifier = get_or_create(
            Classifier,
            keys=dict(
                task_id=self.task.id,
                name=classifier_name
            )
        )

        if self.force:
            print 'Deleting existing predictions...'
            PredictionConfiguration.delete().where(
                Classifier.id == classifier.id,
                FeatureUnion.id == self.feature_union.id,
                SplitDefinition.id == split.id
            ).execute()

        config = get_or_create(
            PredictionConfiguration,
            keys=dict(
                classifier_id=classifier.id,
                feature_union_id=self.feature_union.id,
                split_definition_id=split.id
            )
        )

        print 'inserting predictions...'
        insert_predictions(
            config=config,
            task=self.task,
            partition=self.partition,
            df=self.df,
        )

    """
    
    """

    def get_training_data(self, X, y):
        return X, y


def evaluate_performance(task, funcs, partition_name=None, feature_union_name=None):
    where = [Task.name == task.name]
    if partition_name is not None:
        where.append(Partition.name == partition_name)
    if feature_union_name is not None:
        where.append(FeatureUnion.name == feature_union_name)

    funcs = np.atleast_1d(funcs)
    func_names = map(lambda func: func.__name__, funcs)

    q = Partition.select(
        Task.name,
        Partition.name,
        SplitDefinition.id,
        PredictionConfiguration.id,
        Classifier.name,
        FeatureUnion.name,
        Prediction.kind,
        Labels.y,
        Prediction.p
    ).join(
        Task,
        on=Partition.task_id == Task.id
    ).join(
        SplitDefinition,
        on=SplitDefinition.partition_id == Partition.id
    ).join(
        PredictionConfiguration,
        on=PredictionConfiguration.split_definition_id == SplitDefinition.id
    ).join(
        Classifier,
        on=PredictionConfiguration.classifier_id == Classifier.id
    ).join(
        FeatureUnion,
        on=PredictionConfiguration.feature_union_id == FeatureUnion.id
    ).join(
        Prediction,
        on=Prediction.prediction_configuration_id == PredictionConfiguration.id
    ).join(
        Labels,
        on=Prediction.label_id == Labels.id
    ).join(
        GroupDefinition,
        on=(
            (GroupDefinition.partition_id == Partition.id) &
            (GroupDefinition.label_id == Labels.id)
        )
    ).where(
        *where
    ).tuples()

    rows = list(q)
    df = pd.DataFrame(
        rows,
        columns=[
            'task',
            'partition',
            'split',
            'config',
            'classifier',
            'features',
            'fold_type',
            'y',
            'p',
        ]
    )

    group_columns = ['task', 'partition', 'split', 'config', 'classifier', 'features', 'fold_type']
    groups = df.groupby(group_columns)
    rows = []
    for vals, group in groups:
        p = group.p.tolist()
        y = group.y.tolist()

        rows.append(list(vals) + [func(np.asarray(y), np.asarray(p)) for func in funcs])
    ret = pd.DataFrame(rows, columns=group_columns + func_names)

    return ret, ret.groupby(group_columns).mean()
