import types

from models import *
from utils import get_or_create, islambda


def evaluate_performance(connection, func, task_name=None, force=False):
    assert not islambda(func), 'Performance functions cannot be lambdas'
    
    if task_name is None:
        task_name = 'default'
    assert task_name in connection.tasks
    
    task = connection.tasks[task_name]
    
    metric_name = func.__name__
    performance_metric = get_or_create(
        Metrics,
        keys=dict(
            name=metric_name
        )
    )
    
    configurations = PredictionConfiguration.select().join(Task, on=task == PredictionConfiguration.task)
    for configuration in configurations:
        splits = configuration.split_definition
        
        rows = Partition.select(
            Labels.y, Prediction.p, Groups.key
        ).join(
            SplitDefinition, on=Partition.id == SplitDefinition.partition_id
        ).join(
            PredictionConfiguration, on=SplitDefinition.id == PredictionConfiguration.split_definition_id
        ).join(
            Prediction, on=PredictionConfiguration.id == Prediction.prediction_configuration_id
        ).join(
            Labels, on=Labels.id == Prediction.label_id
        ).join(
            Groups, on=Labels.id == Groups.label_id
        ).where(
            PredictionConfiguration.id == configuration.id
        ).dicts()
        
        data = list(rows)
        
        performance = dict()
        for xval_key in ('train', 'validation', 'test'):
            keys = set(getattr(splits, '{}_folds'.format(xval_key)))
            subset = filter(lambda dd: dd['key'] in keys, data)
            
            performance[xval_key] = func(
                [dd['y'] for dd in subset],
                [dd['p'] for dd in subset]
            )
        
        performance = get_or_create(
            Performance,
            keys=dict(
                metric=performance_metric,
                prediction_configuration=configuration
            )
        )
        
        performance.result = performance
        performance.save()
        
        print performance
