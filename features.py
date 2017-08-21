import pandas as pd
import numpy as np

from itertools import groupby

from utils import get_or_create
from models import *


def check_type(d):
    if isinstance(d, dict):
        for kk in d.keys():
            d[kk] = check_type(d[kk])
        
        return d
    
    elif isinstance(d, (list, tuple)):
        return d
    
    elif isinstance(d, np.ndarray) and d.ndim == 1:
        if d.dtype in (float, int, np.float, np.int):
            return d.astype(float).tolist()
        
        return d.tolist()
    
    elif isinstance(d, pd.Series):
        return check_type(d.todict())
    
    elif isinstance(d, pd.DataFrame):
        assert len(d) == 1
        
        return d.to_dict(orient='records')[0]
    
    elif isinstance(d, (int, float, np.float, np.int)):
        return float(d)
    
    elif isinstance(d, (str, unicode)):
        return d
    
    else:
        raise ValueError


# def parse_feature_type(d):
#     return check_type(d)


def _extract_features(task, view, sequence, func_name, func, left, right):
    function = get_or_create(
        Function,
        keys=dict(
            name=func_name,
        )
    )
    
    function = get_or_create(
        FunctionConfiguration,
        keys=dict(
            task=task,
            function=function,
            left=left,
            right=right
        )
    )
    
    labels = Labels.select().where(
        Labels.task == task,
        Labels.sequence == sequence
    )
    
    data = Data.select().where(
        Data.view == view,
        Data.sequence == sequence
    )
    
    df = pd.DataFrame(
        data=[dd.x for dd in data],
        index=[dd.i for dd in data],
    )
    
    rows = []
    for label in labels:
        inds = ((df.index > label.i + left) &
                (df.index <= label.i + right))
        seq = df[inds]
        
        f_of_x = func(seq)
        f_of_x_safe = check_type(f_of_x)
        
        assert np.all(np.isfinite(f_of_x)), \
            'Extracted features contain NaNs, with\ndf=\n{}\n\nf(df) = {}'.format(
                df[inds], f_of_x
            )
        
        rows.append(dict(
            function_configuration=function,
            label=label,
            x=f_of_x_safe
        ))
    
    Features.insert_many(rows).execute()


def identity(xx):
    return xx


def extract_features(connection, task=None, view=None, func=None, func_name=None, left=-1, right=0, force=False,
                     singleton_feature_union=False):
    if task is None:
        task = 'default'
    
    if view is None:
        view = 'default'
    
    if func is None:
        func = identity
    
    if func_name is None:
        func_name = func.__name__
    
    assert left < right
    
    task = connection.tasks[task]
    view = connection.views[view]
    
    base_func = get_or_create(
        Function,
        keys={
            'name': func_name,
        }
    )
    
    func_instance = get_or_create(
        FunctionConfiguration,
        keys={
            'task': task,
            'view': view,
            'function': base_func,
            'left': left,
            'right': right,
        }
    )
    
    num_features = Features.select().where(
        Features.function_configuration == func_instance
    ).count()
    
    num_labels = Labels.select().where(
        Labels.task == task
    ).count()
    
    do_computation = force or (num_features != num_labels)
    if do_computation:
        Features.delete().where(
            Features.function_configuration == func_instance
        ).execute()
        
        for _, sequence in connection.itersequences():
            print task, view, sequence, func_instance
            
            _extract_features(
                task=task,
                view=view,
                sequence=sequence,
                func_name=func_name,
                func=func,
                left=left,
                right=right
            )
    
    if singleton_feature_union:
        FeatureUnion.create(
            task=task,
            name=func_name,
            function_ids=[func_instance.id]
        )
    
    return func_instance.id


# def load_training_data_y(feature_union_id, partition_name):
#     # Load base data
#     feature_union = FeatureUnion.get(FeatureUnion.id == feature_union_id)
#     task = feature_union.task
#
#     partition = Partition.get(
#         Partition.task_id == task.id,
#         Partition.name == partition_name
#     )
#
#     # Load labels
#     query = Labels.select(
#         Labels.id,
#         Sequence.name,
#         Groups.key,
#         Labels.i,
#         Labels.y
#     ).join(
#         Groups, on=(
#             (Groups.label_id == Labels.id) &
#             (Groups.partition_id == partition.id)
#         )
#     ).join(
#         Task, on=Labels.task_id == Task.id
#     ).join(
#         Sequence, on=Labels.sequence_id == Sequence.id
#     ).where(
#         Labels.task == task
#     ).dicts()
#
#     rows = [row for row in query]
#     for row in rows:
#         row[-1] = np.asarray(row[-1])
#
#     df = pd.DataFrame(
#         [row[1:] for row in rows],
#         index=[row[0] for row in rows],
#         columns=('sequence', 'group', 'i', 'y')
#     )
#     df.index.name = 'id'
#
#     return df


# def labeload_training_data_x(feature_union_id):
#     # Load base data
#     feature_union = FeatureUnion.get(FeatureUnion.id == feature_union_id)
#     function_ids = feature_union.function_ids
#     task = feature_union.task
#
#     # The main query and its join
#     query = Labels.select(
#         Labels.id,
#         Features.function_id,
#         Features.x,
#     ).join(
#         Features,
#         on=(
#             (Features.label_id == Labels.id) &
#             (Features.function_id << function_ids)
#         )
#     ).where(
#         Labels.task == task
#     ).dicts()
#
#     data = []
#     for label_id, group in groupby(query, lambda dd: dd['id']):
#         print {'{}_{}'.format(element['function'], kk): vv
#                 for element in group for kk, vv in element['x'].iteritems()}
#         break
#
#     return df


def _load_training_data(dataset_id, partition_id, feature_union_id):
    '''
    SELECT
          s.name                      sequence_id,
          l.i                         i,
          f.function_configuration_id function_id,
          g.key                       group_key,
          f.x                         x,
          l.y                         y
        FROM datasets d
          JOIN tasks t ON t.dataset_id = d.id
          JOIN sequences s ON s.dataset_id = d.id
          JOIN labels l ON (l.task_id = t.id AND
                            l.sequence_id = s.id)
          JOIN features f ON f.label_id = l.id
          JOIN partitions p ON p.task_id = t.id
          JOIN groups g ON (g.partition_id = p.id AND
                            g.label_id = l.id)
        WHERE
          d.id = 1 AND
          p.id = 1 AND
          f.function_configuration_id IN (SELECT unnest(function_ids)
                                          FROM feature_unions
                                          WHERE id = 1)
        ORDER BY
          s.id,
          l.i,
          f.function_configuration_id;
  
    :param dataset_id:
    :param partition_id:
    :param feature_union_id:
    :return:
    '''
    feature_union = FeatureUnion.get(FeatureUnion.id == feature_union_id)
    function_ids = feature_union.function_ids
    
    query = Dataset.select(
        Labels.id.alias('label_id'),
        Sequence.name.alias('sequence'),
        Labels.i.alias('i'),
        Function.name.alias('function_name'),
        Groups.key.alias('group'),
        Features.x.alias('x'),
        Labels.y.alias('y')
    ).join(
        Task, on=Task.dataset == Dataset.id
    ).join(
        Sequence, on=Sequence.dataset == Dataset.id
    ).join(
        Labels, on=((Labels.task == Task.id) &
                    (Labels.sequence == Sequence.id))
    ).join(
        Features, on=Features.label == Labels.id
    ).join(
        Partition, on=Partition.task == Task.id
    ).join(
        Groups, on=((Groups.partition == Partition.id) &
                    (Groups.label == Labels.id))
    ).join(
        FunctionConfiguration, on=Features.function_configuration_id == FunctionConfiguration.id
    ).join(
        Function, on=FunctionConfiguration.function == Function.id
    ).where(
        Dataset.id == dataset_id,
        Partition.id == partition_id,
        (Features.function_configuration << function_ids)
    ).order_by(
        Sequence.id,
        Labels.i,
        Features.function_configuration
    ).dicts()
    
    return query


# def load_training_data(connection, task, partition_name, feature_union_id):
def load_training_data(connection, task_name, partition_name, feature_union_name):
    task = connection.tasks[task_name]
    partition = connection.partitions[task][partition_name]
    feature_union = connection.feature_unions[task_name][feature_union_name]
    
    rows = _load_training_data(
        dataset_id=connection.dataset.id,
        partition_id=partition.id,
        feature_union_id=feature_union.id
    )
    
    return rows
