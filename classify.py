import pandas as pd
import numpy as np

from itertools import groupby
from os.path import exists

import json

from sklearn.pipeline import Pipeline
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
    def __init__(self, connection, partition_name, task_name=None, feature_union_name=None):
        assert isinstance(feature_union_name, (str, unicode, type(None)))
        assert isinstance(task_name, (str, unicode, type(None)))
        assert isinstance(partition_name, (str, unicode))
        
        if task_name is None:
            task_name = 'default'
        
        if feature_union_name is None:
            feature_union_name = 'identity'
        
        self.connection = connection
        self.task = connection.tasks[task_name]
        self.partition = connection.partitions[task_name][partition_name]
        self.feature_union = connection.feature_unions[task_name][feature_union_name]
        
        self.rows = load_training_data(
            task_name=task_name,
            connection=connection,
            partition_name=partition_name,
            feature_union_name=feature_union_name
        )
        
        data = []
        for sequence_name, sequence_group in groupby(self.rows, lambda rr: rr['sequence']):
            for time_value, group in groupby(sequence_group, lambda rr: rr['i']):
                elements = list(group)
                
                assert all(map(lambda ee: ee['y'] == elements[0]['y'], elements))
                assert all(map(lambda ee: ee['group'] == elements[0]['group'], elements))
                assert all(map(lambda ee: ee['label_id'] == elements[0]['label_id'], elements))
                
                data.append(dict(
                    i=time_value,
                    sequence=sequence_name,
                    group=elements[0]['group'],
                    label_id=elements[0]['label_id'],
                    x={'{}::{}'.format(element['function_name'], kk): vv
                       for element in elements
                       for kk, vv in element['x'].iteritems()},
                    y=elements[0]['y']
                ))
        
        self.df = pd.DataFrame(data)
        self.df.set_index('label_id', inplace=True)
    
    """
    
    """
    
    def fit(self, model, model_name=None):
        if model_name is None:
            if isinstance(model, Pipeline):
                model_name = '+'.join(
                    ['{}={}'.format(nn, ff.__class__.__name__) for nn, ff in model.steps]
                )

        for split in self.partition.splits:
            self._fit_split(
                split=split,
                model=model,
                model_name=model_name
            )
    
    def _fit_split(self, split, model, model_name):
        train_inds, validation_inds, test_inds = train_validation_test_inds(
            df=self.df,
            train_folds=split.train_folds,
            validation_folds=split.validation_folds,
            test_folds=split.test_folds
        )
        
        self.df.loc[train_inds, 'kind'] = 'train'
        self.df.loc[test_inds, 'kind'] = 'test'
        self.df.loc[validation_inds, 'kind'] = 'validation'
        
        learnt_model = self.fit_split(
            model=clone(model),
            train_X=self.df[train_inds].x,
            train_y=self.df[train_inds].y,
            validation_X=self.df[validation_inds].x,
            validation_y=self.df[train_inds].y,
            test_X=self.df[test_inds].x,
            test_y=self.df[train_inds].y
        )
        
        assert learnt_model is not None
        
        self.evaluate_performance(
            model=learnt_model,
            model_name=model_name,
            split=split
        )
    
    def fit_split(self, model, train_X, train_y, validation_X, validation_y, test_X, test_y):
        """

            # Learn the model
            model = <insert classifier here>
            model.fit(train_X, train_y)
            
            return model
        
        :param model:
        :param self:
        :param train_X:
        :param train_y:
        :param validation_X:
        :param validation_y:
        :param test_X:
        :param test_y:
        :return:
        """
        
        raise NotImplementedError
    
    def evaluate_performance(self, model, model_name, split):
        # Save predictions to the mldb
        classifier = get_or_create(
            Classifier,
            keys=dict(
                name=model_name
            )
        )
        
        print 'Deleting existing predictions for this configuration.'
        PredictionConfiguration.delete().where(
            Classifier.id == classifier.id,
            Task.id == self.task.id,
            FeatureUnion.id == self.feature_union.id,
            SplitDefinition.id == split.id
        ).execute()
        
        config = get_or_create(
            PredictionConfiguration,
            keys=dict(
                classifier_id=classifier.id,
                task_id=self.task.id,
                feature_union_id=self.feature_union.id,
                split_definition_id=split.id
            )
        )
        
        print 'inserting predictions...'
        insert_predictions(
            config=config,
            model=model,
            df=self.df
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
        Groups,
        on=(
            (Groups.partition_id == Partition.id) &
            (Groups.label_id == Labels.id)
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
