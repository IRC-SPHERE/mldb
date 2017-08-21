from sklearn.datasets import make_classification

import numpy as np
import pandas as pd

from importer import Importer, BaseSplit

from features import identity

from classify import DBClassifier

from utils import insert_data


class Example(Importer):
    def __init__(self):
        super(Example, self).__init__(
            connection=dict(
                database='mldb_test',
                credentials=dict(user='', password='', host='localhost', port=5432)
            ),
            meta=dict(
                dataset=dict(name='mldb_test_dataset'),
                tasks=[dict(name='default', columns=['one', 'two'], meta={})],
                views=[dict(name='default', meta=dict(columns=['a', 'b', 'c']))],
                sequences=[
                    dict(name='default', meta=dict()),
                ],
            )
        )
        
        self.data = {}
        
        self.X, self.y = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_repeated=0,
            n_classes=4,
        )
    
    def import_view(self, sequence, view):
        return [dict(zip(range(5), xx)) for xx in self.X]
        # Alternatively just return self.X
    
    def import_labels(self, sequence, task):
        return self.y


class TrainValidationTest(BaseSplit):
    def __init__(self):
        super(TrainValidationTest, self).__init__([
            dict(train_folds=[0], validation_folds=[1], test_folds=[2]),
            dict(train_folds=[1], validation_folds=[2], test_folds=[0]),
            dict(train_folds=[2], validation_folds=[0], test_folds=[1])
        ])
    
    def __call__(self, label):
        return np.mod(label.id, 3)


class ExampleClassifier(DBClassifier):
    def __init__(self):
        super(ExampleClassifier, self).__init__(
            connection=Example(),
            partition_name=TrainValidationTest.__name__
            # connection, task_name, partition_name, feature_union_name
        )
    
    def fit_split(self, model, train_X, train_y, validation_X, validation_y, test_X, test_y):
        model.fit(train_X, train_y)
        
        return model


if __name__ == '__main__':
    ## IMPORT DATA
    # from features import extract_features
    # from models import FeatureUnion
    #
    # test = Example()
    # test.import_all_data()
    # test.register_partition(TrainValidationTest())
    # func_id_1 = extract_features(connection=test, func=identity, func_name='identity_1')
    # func_id_2 = extract_features(connection=test, func=identity, func_name='identity_2')
    # FeatureUnion.create(task=test.tasks['default'], name='identity', function_ids=[func_id_1, func_id_2])
    
    ## MAKE PREDICTIONS
    # from sklearn.pipeline import Pipeline
    # from sklearn.feature_extraction import DictVectorizer
    # from sklearn.linear_model import LogisticRegressionCV
    #
    # model = Pipeline([
    #         ('vectorizer', DictVectorizer()),
    #         ('logistic_regression', LogisticRegressionCV())
    #     ])
    #
    # clf = ExampleClassifier()
    # clf.fit(model)
    
    ## EVALUATE PERFORMANCE
    from performance import evaluate_performance
    
    
    def accuracy(y, p):
        return np.mean(np.argmax(p, axis=1) == y)
    
    
    evaluate_performance(
        connection=Example(),
        func=accuracy
    )
