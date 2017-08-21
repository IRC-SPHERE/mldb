import pandas as pd
import numpy as np

import models


def islambda(func):
    l0 = lambda: None
    
    return isinstance(func, type(l0)) and func.__name__ == l0.__name__


def get_or_create(model, keys, values=None):
    try:
        instance = model.get(**keys)
    
    except model.DoesNotExist:
        params = {kk: vv for kk_vv in (keys, values or {}) for kk, vv in kk_vv.iteritems()}
        instance = model.create(**params)
    
    return instance


def insert_data(sequence, view, df, force):
    # Simple check based on number of records in DB and on file
    if force is False:
        num_in_db = models.Data.select().where(
            models.Data.sequence == sequence,
            models.Data.view == view
        ).count()
        
        if num_in_db == len(df):
            return
    
    # Delete existing data
    models.Data.delete().where(
        models.Data.sequence == sequence,
        models.Data.view == view
    ).execute()
    
    # Insert data
    if isinstance(df, pd.DataFrame):
        inds = df.index.tolist()
        df = df.values.tolist()
    
    else:
        inds = range(len(df))
        
        if isinstance(df, np.ndarray):
            df = df.tolist()
    
    rows = [dict(
        view=view,
        sequence=sequence,
        i=ii,
        x=rr
    ) for ii, rr in zip(inds, df)]
    
    models.Data.insert_many(rows).execute()


def insert_labels(sequence, task, df, force):
    # Simple check based on number of records in DB and on file
    if force is False:
        num_in_db = models.Labels.select().where(
            models.Labels.sequence == sequence,
            models.Labels.task == task
        ).count()
        
        if num_in_db == len(df):
            return
    
    # Delete existing data
    models.Labels.delete().where(
        models.Labels.sequence == sequence,
        models.Labels.task == task
    ).execute()
    
    if isinstance(df, pd.DataFrame):
        inds = df.index.tolist()
        df = df.values.tolist()
    
    else:
        inds = range(len(df))
        
        if isinstance(df, np.ndarray):
            df = df.tolist()
    
    # Insert data
    rows = [dict(
        task=task,
        sequence=sequence,
        i=ii,
        y=rr
    ) for ii, rr in zip(inds, df)]
    
    models.Labels.insert_many(rows).execute()


def insert_predictions(config, model, df):
    # Delete existing data
    models.Prediction.delete().where(
        models.Prediction.prediction_configuration == config
    ).execute()
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba([row.x for ri, row in df.iterrows()])
    else:
        probs = model.predict([row.x for ri, row in df.iterrows()])
    
    rows = [dict(prediction_configuration=config,
                 label=models.Labels.get(models.Labels.id == ii),
                 p=pp) for (ii, rr), pp in zip(df.iterrows(), probs)]
    models.Prediction.insert_many(rows).execute()
