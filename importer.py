from playhouse.postgres_ext import *
import json

from meta import Metadata
from models import *

from utils import get_or_create, insert_data, insert_labels


class BaseSplit(object):
    def __init__(self, splits):
        self.splits = splits
    
    @property
    def num_splits(self):
        return len(self.splits)
    
    def __iter__(self):
        return iter(self.splits)
    
    def __call__(self, ll):
        raise NotImplementedError


class Importer(object):
    def __init__(self, connection, meta):
        if isinstance(connection, dict):
            self.connection_params = connection
        else:
            self.connection_params = json.load(open(connection, 'r'))
        
        db.init(
            self.connection_params['database'],
            **self.connection_params['credentials']
        )
        db.create_tables(all_tables, safe=True)
        
        self.db = db
        self.meta = Metadata(meta)
        
        # Load the dataset object
        self.dataset, created = Dataset.get_or_create(
            name=self.meta.name
        )
        
        # Load the tasks
        self.tasks = {}
        for task in self.meta.tasks:
            self.tasks[task['name']] = get_or_create(
                Task,
                keys=dict(
                    dataset=self.dataset,
                    name=task['name']
                ),
                values=dict(
                    columns=task['columns'],
                    num_columns=len(task['columns'])
                )
            )
        
        # Load the views
        self.views = {}
        for view in self.meta.views:
            self.views[view['name']] = get_or_create(
                View,
                keys=dict(
                    dataset=self.dataset,
                    name=view['name']
                ),
                values=dict(meta=dict())
            )
        
        # Load the sequences
        self.sequences = {}
        for sequence in self.meta.sequences:
            self.sequences[sequence['name']] = get_or_create(
                Sequence,
                keys=dict(
                    dataset=self.dataset,
                    name=sequence['name']
                ),
                values=dict(meta=dict())
            )
        
        # Load the partitions
        self.splits = {}
        self.partitions = {}
        self.classifiers = {}
        self.feature_unions = {}
        for task_name, task in self.tasks.iteritems():
            pp = {pp.name: pp for pp in task.partitions}
            self.partitions[task_name] = pp
            self.partitions[task] = pp
            
            ss = {pp.name: list(pp.splits.execute()) for pp in task.partitions}
            self.splits[task_name] = ss
            self.splits[task] = ss
            
            # cc = {}
            # for classifier in task.classifiers:
            #     cc[classifier.name] = classifier
            # self.classifiers[task] = cc
            # self.classifiers[task_name] = cc
            
            ff = {}
            for feature_union in task.feature_unions:
                ff[feature_union.name] = feature_union
            self.feature_unions[task] = ff
            self.feature_unions[task_name] = ff
    
    """

    """
    
    def itersequences(self):
        for sequence in sorted(self.sequences.keys()):
            yield sequence, self.sequences[sequence]
    
    def itertasks(self):
        for task in sorted(self.tasks.keys()):
            yield task, self.tasks[task]
    
    def iterviews(self):
        for view in sorted(self.views.keys()):
            yield view, self.views[view]
    
    """
    
    """
    
    def import_all_data(self):
        for _, sequence in self.itersequences():
            self.import_sequence(sequence)
    
    def import_sequence(self, sequence):
        print sequence
        
        assert sequence.dataset == self.dataset
        
        print ' ', 'Tasks'
        for _, task in self.itertasks():
            print ' ', ' ', task
            
            ll = self.import_labels(task=task, sequence=sequence)
            if ll is not None:
                insert_labels(
                    sequence=sequence,
                    task=task,
                    df=ll,
                    force=False
                )
        
        print ' ', 'Views'
        for _, view in self.iterviews():
            print ' ', ' ', view
            
            dd = self.import_view(sequence=sequence, view=view)
            if dd is not None:
                insert_data(
                    sequence=sequence,
                    view=view,
                    df=dd,
                    force=False
                )
    
    """
    
    """
    
    def import_view(self, sequence, view):
        raise NotImplementedError
    
    def import_labels(self, sequence, task):
        raise NotImplementedError
    
    def register_partition(self, partition_generator, task=None):
        if task is None:
            task = Task.get(
                Task.dataset == self.dataset,
                Task.name == 'default'
            )
        
        assert isinstance(task, Task)
        
        assert isinstance(partition_generator, BaseSplit)
        partition_name = partition_generator.__class__.__name__
        partition = get_or_create(
            Partition,
            keys=dict(
                task=task,
                name=partition_name
            )
        )
        
        label_query = Labels.select().where(
            Labels.task == task
        )
        
        group_query = Groups.select().where(
            Groups.partition == partition
        )
        
        update = (group_query.count() == label_query.count()) and group_query.count() != 0
        if update:
            Groups.delete().where(
                Groups.partition == partition
            ).execute()
        
        rows = [dict(
            partition=partition,
            label=label,
            key=partition_generator(label)
        ) for label in label_query]
        Groups.insert_many(rows).execute()
        
        for ii, split in enumerate(partition_generator):
            self.register_split_definition(
                task=task,
                split_num=ii,
                partition_name=partition_name,
                split=split
            )
    
    def register_split_definition(self, task, partition_name, split_num, split, tag=None):
        assert isinstance(split, dict)
        
        tag = tag or ''
        
        partition = get_or_create(
            Partition,
            keys=dict(
                task=task,
                name=partition_name
            )
        )
        
        get_or_create(
            SplitDefinition,
            keys=dict(
                tag=tag,
                split_num=split_num,
                partition=partition
            ),
            values=split
        )
