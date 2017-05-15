import pandas as pd
from numpy import linspace, clip

import models


def interval_to_time(data, columns, n0, n1, oversampling_rate=10, time_key='t'):
    df = pd.DataFrame(
        {time_key: linspace(n0, n1, oversampling_rate * (n1 - n0) + 1)}
    )
    df.set_index(time_key, inplace=True)
    
    # Initialise the columns
    for col in columns:
        df.loc[:, col] = 0.0
    
    # Set the non-zero data elements
    for ii, row in data.iterrows():
        inds = (df.index >= row.start) & (df.index <= row.end)
        
        df.loc[inds, row['name']] = 1.0
    
    return df[columns]


def load_dataframe(path, columns, index_col):
    df = pd.read_csv(path)
    
    for col in columns:
        if col not in df:
            df.loc[:, col] = None
    
    df.set_index(index_col, inplace=True)
    df = df.groupby(level=0).last()  # Remove duplicates
    
    return df[columns]


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
    rows = [dict(
        view=view,
        sequence=sequence,
        t=tt,
        x=rr.tolist()
    ) for tt, rr in df.iterrows()]
    
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
    
    # Insert data
    rows = [dict(
        task=task,
        sequence=sequence,
        t=tt,
        y=rr.tolist()
    ) for tt, rr in df.iterrows()]
    
    models.Labels.insert_many(rows).execute()


def insert_predictions(config, task, partition, df):
    # Delete existing data
    models.Prediction.delete().where(
        models.Prediction.prediction_configuration == config
    ).execute()

    rows = [dict(prediction_configuration=config,
            label=tt,
            kind=rr.kind,
            p=map(lambda pp: clip(pp, 1e-12, 1.0), rr.p)) for tt, rr in df.iterrows()]
    models.Prediction.insert_many(rows).execute()
