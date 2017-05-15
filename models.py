from playhouse.postgres_ext import *

db = PostgresqlDatabase(None)


class BaseModel(Model):
    class Meta:
        database = db


"""

"""


class Dataset(BaseModel):
    id = PrimaryKeyField()
    name = CharField(unique=True, null=False, index=True)
    
    def __repr__(self):
        return "<{} name={}>".format(
            self.__class__.__name__,
            self.name
        )
    
    class Meta:
        db_table = 'datasets'
        
        order_by = (
            'name',
        )


"""

"""


class Task(BaseModel):
    id = PrimaryKeyField()
    
    dataset = ForeignKeyField(Dataset, related_name='tasks', null=False, index=True, on_delete='CASCADE')
    name = CharField(null=False, index=True)
    
    columns = ArrayField(field_class=CharField, null=False)
    num_columns = IntegerField(null=False)
    
    def __repr__(self):
        return "<{} db={} task={}>".format(
            self.__class__.__name__,
            self.dataset.name,
            self.name
        )
    
    class Meta:
        db_table = 'tasks'
        
        order_by = (
            'dataset',
            'name',
        )
        
        indexes = (
            (('dataset', 'name'), True),
        )


class Sequence(BaseModel):
    id = PrimaryKeyField()
    
    dataset = ForeignKeyField(Dataset, related_name='sequences', null=False, index=True, on_delete='CASCADE')
    name = CharField(null=False, index=True)
    
    t_min = FloatField(null=False)
    t_max = FloatField(null=False)
    
    def __repr__(self):
        return "<{} db={} sequence={}>".format(
            self.__class__.__name__,
            self.dataset.name,
            self.name
        )
    
    class Meta:
        db_table = 'sequences'
        
        order_by = (
            'dataset',
            'name',
        )
        
        indexes = (
            (('dataset', 'name'), True),
        )


class View(BaseModel):
    id = PrimaryKeyField()
    
    dataset = ForeignKeyField(Dataset, related_name='views', null=False, index=True, on_delete='CASCADE')
    name = CharField(null=False, index=True)
    
    columns = ArrayField(field_class=CharField, null=False)
    num_columns = IntegerField(null=False)
    
    def __repr__(self):
        return "<{} db={} view={}>".format(
            self.__class__.__name__,
            self.dataset.name,
            self.name
        )
    
    class Meta:
        db_table = 'views'
        
        order_by = (
            'dataset',
            'name'
        )
        
        indexes = (
            (('dataset', 'name'), True),
        )


"""

"""


class Classifier(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='classifiers', null=False, index=True, on_delete='CASCADE')
    name = CharField(null=False, index=True)
    
    def __repr__(self):
        return "<{} {} {}>".format(
            self.__class__.__name__,
            self.name,
            self.task
        )
    
    class Meta:
        db_table = 'classifiers'
        
        order_by = (
            'task',
            'name',
        )
        
        indexes = (
            (('task', 'name'), True),
        )


class Partition(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='partitions', null=False, index=True, on_delete='CASCADE')
    name = CharField(null=False, index=True)
    
    def __repr__(self):
        return "<{} task={} name={}>".format(
            self.__class__.__name__,
            self.task.name,
            self.name
        )
    
    class Meta:
        db_table = 'partitions'
        
        order_by = (
            'task',
            'name',
        )
        
        indexes = (
            (('task', 'name'), True),
        )


class Function(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='functions', null=False, index=True, on_delete='CASCADE')
    name = CharField(null=False, index=True)
    
    left = FloatField(null=False, default=-1.0)
    right = FloatField(null=False, default=0)
    
    def __repr__(self):
        return "<{} name={} task={} window=({}, {}]>".format(
            self.__class__.__name__,
            self.name,
            self.task.name,
            self.left,
            self.right
        )
    
    class Meta:
        db_table = 'functions'
        
        order_by = (
            'task',
            'name',
            'right',
            '-left'
        )
        
        indexes = (
            (('task', 'name', 'left', 'right'), True),
        )


class FeatureUnion(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='feature_unions', null=False, index=True, on_delete='CASCADE')
    name = CharField(null=False, index=True)
    
    function_ids = ArrayField(IntegerField, null=False, unique=True)
    
    def __repr__(self):
        return "<{} name={} task={} function_ids={{{}}}>".format(
            self.__class__.__name__,
            self.name,
            self.task.name,
            ','.join(map(str, self.function_ids))
        )
    
    class Meta:
        db_table = 'feature_unions'
        
        order_by = (
            'task',
            'name',
        )
        
        indexes = (
            (('task', 'name'), True),
        )


class Labels(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='labels', null=False, index=True, on_delete='CASCADE')
    sequence = ForeignKeyField(Sequence, related_name='labels', null=False, index=True, on_delete='CASCADE')
    
    t = FloatField(null=False)
    y = ArrayField(FloatField, null=False)
    
    def __repr__(self):
        return "<{} {} {} {} {}>".format(
            self.__class__.__name__,
            self.task.name,
            self.sequence.name,
            self.t,
            self.y
        )
    
    class Meta:
        db_table = 'labels'
        
        order_by = (
            'task',
            'sequence',
            't'
        )


class Data(BaseModel):
    id = PrimaryKeyField()
    
    view = ForeignKeyField(View, null=False, index=True, on_delete='CASCADE')
    sequence = ForeignKeyField(Sequence, null=False, index=True, on_delete='CASCADE')
    
    t = FloatField(null=False)
    x = ArrayField(FloatField, null=False)
    
    def __repr__(self):
        return "<{} {}={}>".format(
            self.__class__.__name__,
            self.t, self.x
        )
    
    class Meta:
        db_table = 'data'
        
        order_by = (
            'view',
            'sequence',
            't'
        )
        
        indexes = (
            (('sequence', 'view', 't'), True),
        )


"""

"""


class SplitDefinition(BaseModel):
    id = PrimaryKeyField()
    
    partition = ForeignKeyField(Partition, related_name='splits', null=False, index=True, on_delete='CASCADE')
    
    tag = TextField(null=True, index=True)
    split_num = IntegerField(null=False, index=True)
    
    train_folds = ArrayField(IntegerField, null=False)
    validation_folds = ArrayField(IntegerField, null=False)
    test_folds = ArrayField(IntegerField, null=False)
    
    def __repr__(self):
        return "<{} partition={} train={} validation={} test={}>".format(
            self.__class__.__name__,
            self.partition.name,
            self.train_folds,
            self.validation_folds,
            self.test_folds
        )
    
    class Meta:
        db_table = 'split_definitions'
        
        order_by = (
            'partition',
            'tag',
            'split_num'
        )
        
        indexes = (
            (('partition', 'tag', 'split_num'), True),
        )


class GroupDefinition(BaseModel):
    id = PrimaryKeyField()
    
    partition = ForeignKeyField(Partition, related_name='partitions', null=False, index=True, on_delete='CASCADE')
    label = ForeignKeyField(Labels, related_name='labels', null=False, index=True, on_delete='CASCADE')
    
    key = IntegerField(null=True)
    
    def __repr__(self):
        return "<{} label={} split_value={}>".format(
            self.__class__.__name__,
            self.label,
            self.fold
        )
    
    class Meta:
        db_table = 'group_definitions'
        
        order_by = (
            'partition',
            'label',
            'key'
        )
        
        indexes = (
            (('partition', 'label', 'key'), True),
        )


class Features(BaseModel):
    id = PrimaryKeyField()
    
    function = ForeignKeyField(Function, null=False, index=True, on_delete='CASCADE')
    label = ForeignKeyField(Labels, related_name='features', null=False, index=True, on_delete='CASCADE')
    
    x = ArrayField(FloatField, null=False)
    
    def __repr__(self):
        return "<{} {} {} {}>".format(
            self.__class__.__name__,
            self.function,
            self.label,
            self.x
        )
    
    class Meta:
        db_table = 'features'
        
        order_by = (
            'function',
            'label',
        )
        
        indexes = (
            (('function', 'label'), True),
        )


"""

"""


class PredictionConfiguration(BaseModel):
    id = PrimaryKeyField()
    
    classifier = ForeignKeyField(Classifier, null=False, index=True, on_delete='CASCADE')
    feature_union = ForeignKeyField(FeatureUnion, null=False, index=True, on_delete='CASCADE')
    split_definition = ForeignKeyField(SplitDefinition, related_name='predictions', null=False, index=True,
                                       on_delete='CASCADE')
    
    def __repr__(self):
        return "<{} classifier={} feature_union={} p={}>".format(
            self.__class__.__name__,
            self.classifier.name,
            self.feature_union.name,
            self.split_definition
        )
    
    class Meta:
        db_table = 'prediction_configuration'
        
        order_by = (
            'classifier',
            'feature_union',
            'split_definition',
        )
        
        indexes = (
            (('classifier', 'feature_union', 'split_definition'), True),
        )


class Prediction(BaseModel):
    id = PrimaryKeyField()
    
    prediction_configuration = ForeignKeyField(PredictionConfiguration, related_name='predictions', null=False,
                                               index=True, on_delete='CASCADE')
    label = ForeignKeyField(Labels, related_name='predictions', null=False, index=True, on_delete='CASCADE')
    
    kind = CharField(null=True, index=True)
    
    p = ArrayField(FloatField, null=False)
    
    def __repr__(self):
        return "<{} prediction_configuration={} p={}>".format(
            self.__class__.__name__,
            self.prediction_configuration,
            self.p
        )
    
    class Meta:
        db_table = 'predictions'
        
        order_by = (
            'prediction_configuration',
            'label'
        )
        
        indexes = (
            (('prediction_configuration', 'label'), True),
        )


all_tables = [
    # Depth 0
    Dataset,
    
    # Depth 1
    Task,
    Sequence,
    View,
    
    # Depth 2
    Classifier,
    Partition,
    Function,
    FeatureUnion,
    Labels,
    Data,
    
    # Depth 3
    SplitDefinition,
    GroupDefinition,
    Features,
    
    # Depth 4
    PredictionConfiguration,
    Prediction
]
