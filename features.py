import pandas as pd

import numpy as np

from utils import get_or_create
from models import *


def parse_feature_type(d):
    if isinstance(d, (list, tuple)):
        return map(float, d)
    
    elif isinstance(d, np.ndarray) and d.ndim == 1:
        return d.astype(float).tolist()
    
    elif isinstance(d, pd.Series):
        return d.values.astype(float).tolist()
    
    elif isinstance(d, (int, float)):
        return [float(d)]
    
    else:
        raise ValueError(
            'Returned feature type must be from the following set:' +
            '\n\t[{}].'.format(', '.join(map(str, (list, tuple, np.ndarray, pd.Series, int, float)))) +
            '\nYou provided: ' +
            '\n\t{}'.format(type(d))
        )


def _extract_features(task, view, sequence, func_name, func, left, right, do_update):
    function = get_or_create(
        Function,
        keys=dict(
            task=task,
            name=func_name,
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
        index=[dd.t for dd in data],
        columns=view.columns
    )
    df.index.name = 't' 

    rows = []
    for label in labels:
        inds = ((df.index > label.t + left) &
                (df.index <= label.t + right))
        seq = df[inds]
        f_of_x = func(seq)
        f_of_x = parse_feature_type(f_of_x)
        
        assert np.all(np.isfinite(f_of_x)), 'Extracted features contain NaNs, with\ndf=\n{}\n\nf(df) = {}'.format(
            df[inds], f_of_x
        )
        
        if do_update:
            Features.update(
                x=f_of_x
            ).where(
                Features.label == label,
                Features.function == function
            ).execute()
        
        else:
            rows.append(dict(
                function=function,
                label=label,
                x=f_of_x
            ))
    
    if not do_update:
        Features.insert_many(rows).execute()


def extract_features(connection, task, view, func, func_name, left, right, force):
    assert left < right;
    
    task = connection.tasks[task]
    view = connection.views[view]
    
    function = get_or_create(
        Function,
        keys=dict(
            task=task,
            name=func_name,
            left=left,
            right=right
        )
    )
    
    num_features = Features.select().where(
        Features.function == function
    ).count()
    
    num_labels = Labels.select().where(
        Labels.task == task
    ).count()
    
    if not force:
        if num_features == num_labels:
            return
    
    do_update = num_features == num_labels
    if num_features != num_labels:
        Features.delete().where(
            Features.function == function
        ).execute()
    
    for _, sequence in connection.itersequences():
        print task, view, sequence, function
        
        _extract_features(
            task=task,
            view=view,
            sequence=sequence,
            func_name=func_name,
            func=func,
            left=left,
            right=right,
            do_update=do_update
        )


def load_training_data_y(feature_union_id, partition_name):
    # Load base data
    feature_union = FeatureUnion.get(FeatureUnion.id == feature_union_id)
    task = feature_union.task
    
    partition = Partition.get(
        Partition.task_id == task.id,
        Partition.name == partition_name
    )
    
    # Load labels
    query = Labels.select(
        Labels.id,
        Sequence.name,
        GroupDefinition.key,
        Labels.t,
        Labels.y
    ).join(
        GroupDefinition, on=(
            (GroupDefinition.label_id == Labels.id) &
            (GroupDefinition.partition_id == partition.id)
        )
    ).join(
        Task, on=Labels.task_id == Task.id
    ).join(
        Sequence, on=Labels.sequence_id == Sequence.id
    ).where(
        Labels.task == task
    ).tuples()
    
    rows = [list(row) for row in query]
    for row in rows:
        row[-1] = np.asarray(row[-1])
    
    df = pd.DataFrame(
        [row[1:] for row in rows],
        index=[row[0] for row in rows],
        columns=('sequence', 'group', 't', 'y')
    )
    df.index.name = 'id'
    
    return df


def load_training_data_x(feature_union_id):
    # Load base data
    feature_union = FeatureUnion.get(FeatureUnion.id == feature_union_id)
    function_ids = feature_union.function_ids
    task = feature_union.task
    
    # The main query and its join
    query = Labels.select(
        Labels.id,
        Features.x
    ).join(
        Features,
        on=(
            (Features.label_id == Labels.id) &
            (Features.function_id << function_ids)
        )
    ).where(
        Labels.task == task
    ).tuples()
    
    # Perform length test on data
    rows = [list(row) for row in query]
    num_functions = len(function_ids)
    # for fid in xrange(num_functions):
    #     assert len(set([len(row[1]) for row in rows[fid::num_functions]])) == 1
    assert len(rows) % num_functions == 0
    for ii in xrange(0, len(rows), num_functions):
        for jj in xrange(ii + 1, ii + num_functions):
            rows[ii][1].extend(rows[jj][1])
        rows[ii][1] = np.asarray(rows[ii][1])
    rows = rows[::num_functions]
    assert len(set([len(row[1]) for row in rows])) == 1
    
    # Create the data frame
    df = pd.DataFrame(
        [row[1:] for row in rows],
        index=[row[0] for row in rows],
        columns=['x']
    )
    df.index.name = 'id'
    
    return df


def load_training_data(connection, task, partition_name, feature_union_id):
    if isinstance(feature_union_id, (str, unicode)):
        feature_union = FeatureUnion.get(
            FeatureUnion.task == task,
            FeatureUnion.name == feature_union_id
        )
        feature_union_id = feature_union.id
    
    else:
        feature_union = FeatureUnion.get(
            FeatureUnion.id == feature_union_id
        )
    
    # Load the x and y dataframes
    df_y = load_training_data_y(feature_union_id, partition_name)
    df_x = load_training_data_x(feature_union_id)
    
    assert (df_x.index == df_y.index).all()
    df = pd.concat([df_x, df_y], axis=1)
    df = df[['sequence', 'group', 't', 'x', 'y']]
    
    return feature_union, task, df
