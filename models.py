from playhouse.postgres_ext import *

db = PostgresqlDatabase(None)


# db = PostgresqlExtDatabase(None)


class BaseModel(Model):
    class Meta:
        database = db


"""

"""


class Dataset(BaseModel):
    id = PrimaryKeyField()
    
    name = CharField(unique=True, null=False, index=True)
    meta = JSONField(null=True)
    
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


class Classifier(BaseModel):
    id = PrimaryKeyField()
    
    name = CharField(null=False, index=True, unique=True)
    meta = JSONField(null=True)
    
    def __repr__(self):
        return "<{} name={}>".format(
            self.__class__.__name__,
            self.name,
        )
    
    class Meta:
        db_table = 'classifiers'
        
        order_by = (
            'name',
        )


class Function(BaseModel):
    id = PrimaryKeyField()
    
    name = CharField(null=False, index=True, unique=True)
    meta = JSONField(null=True)
    
    def __repr__(self):
        return "<{} name={} meta={}>".format(
            self.__class__.__name__,
            self.name,
            self.meta
        )
    
    class Meta:
        db_table = 'functions'
        
        order_by = (
            'name',
        )


class Metrics(BaseModel):
    id = PrimaryKeyField()
    
    name = CharField(null=False, index=True, unique=True)
    meta = JSONField(null=True)
    
    def __repr__(self):
        return "<{} name={} meta={}>".format(
            self.__class__.__name__,
            self.name,
            self.meta
        )
    
    class Meta:
        db_table = 'metrics'
        
        order_by = (
            'name',
        )


"""

"""


class Task(BaseModel):
    id = PrimaryKeyField()
    
    dataset = ForeignKeyField(Dataset, related_name='tasks', null=False, index=True, on_delete='CASCADE')
    
    name = CharField(null=False, index=True)
    meta = JSONField(null=True)
    
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
    meta = JSONField(null=True)
    
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
    meta = JSONField(null=True)
    
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


class FunctionConfiguration(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, null=False, index=True, on_delete='CASCADE')
    view = ForeignKeyField(View, null=False, index=True, on_delete='CASCADE')
    function = ForeignKeyField(Function, null=False, index=True, on_delete='CASCADE')
    
    left = FloatField(null=False, index=True, default=-1)
    right = FloatField(null=False, index=True, default=0)
    
    meta = JSONField(null=True)
    
    def __repr__(self):
        return "<{} name={} task={} ({}, {}]  meta={}>".format(
            self.__class__.__name__,
            self.function.name,
            self.task.name,
            self.left,
            self.right,
            self.meta,
        )
    
    class Meta:
        db_table = 'function_configuration'
        
        order_by = (
            'task',
            'view',
            'function',
        )
        
        indexes = (
            (('task', 'function', 'view',), True),
        )


class Labels(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='labels', null=False, index=True, on_delete='CASCADE')
    sequence = ForeignKeyField(Sequence, related_name='labels', null=False, index=True, on_delete='CASCADE')
    
    i = FloatField(null=False)
    y = BinaryJSONField(null=False)
    
    def __repr__(self):
        return "<{} {} {} {} {}>".format(
            self.__class__.__name__,
            self.task.name,
            self.sequence.name,
            self.i,
            self.y
        )
    
    class Meta:
        db_table = 'labels'
        
        order_by = (
            'task',
            'sequence',
            'i'
        )
        
        indexes = (
            (('task', 'sequence', 'i'), True),
        )


class Data(BaseModel):
    id = PrimaryKeyField()
    
    view = ForeignKeyField(View, null=False, index=True, on_delete='CASCADE')
    sequence = ForeignKeyField(Sequence, null=False, index=True, on_delete='CASCADE')
    
    i = FloatField(null=False)
    x = BinaryJSONField(null=False)
    
    def __repr__(self):
        return "<{} i={} x={}>".format(
            self.__class__.__name__,
            self.i, self.x
        )
    
    class Meta:
        db_table = 'data'
        
        order_by = (
            'view',
            'sequence',
            'i'
        )
        
        indexes = (
            (('view', 'sequence',), False),
        )


class Features(BaseModel):
    id = PrimaryKeyField()
    
    function_configuration = ForeignKeyField(FunctionConfiguration, null=False, index=True, on_delete='CASCADE')
    label = ForeignKeyField(Labels, related_name='features', null=False, index=True, on_delete='CASCADE')
    
    x = BinaryJSONField(FloatField, null=False)
    
    def __repr__(self):
        return "<{} {} {} {}>".format(
            self.__class__.__name__,
            self.function_configuration,
            self.label,
            self.x
        )
    
    class Meta:
        db_table = 'features'
        
        order_by = (
            'label',
            'function_configuration',
        )
        
        indexes = (
            (('function_configuration', 'label'), True),
        )


"""

"""


class Partition(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='partitions', null=False, index=True, on_delete='CASCADE')
    
    name = CharField(null=False, index=True)
    meta = JSONField(null=True)
    
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


class FeatureUnion(BaseModel):
    id = PrimaryKeyField()
    
    task = ForeignKeyField(Task, related_name='feature_unions', null=False, index=True, on_delete='CASCADE')
    
    name = CharField(null=False, index=True)
    meta = JSONField(null=True)
    
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


class Groups(BaseModel):
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
        db_table = 'groups'
        
        order_by = (
            'partition',
            'label',
            'key'
        )
        
        indexes = (
            (('partition', 'label', 'key'), True),
        )


"""

"""


class PredictionConfiguration(BaseModel):
    id = PrimaryKeyField()
    
    classifier = ForeignKeyField(Classifier, null=False, index=True, on_delete='CASCADE')
    task = ForeignKeyField(Task, related_name='tasks', null=False, index=True, on_delete='CASCADE')
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


class Performance(BaseModel):
    id = PrimaryKeyField()
    
    metric = ForeignKeyField(Metrics, null=False, index=True, on_delete='CASCADE')
    prediction_configuration = ForeignKeyField(PredictionConfiguration, null=False, index=True, on_delete='CASCADE')
    result = JSONField(null=True)
    
    def __repr__(self):
        return "<{} metric={} result={}>".format(
            self.__class__.__name__,
            self.metric.name,
            self.result
        )
    
    class Meta:
        db_table = 'performance'
        
        indexes = (
            (('metric', 'prediction_configuration',), True),
        )


class Prediction(BaseModel):
    id = PrimaryKeyField()
    
    prediction_configuration = ForeignKeyField(PredictionConfiguration, null=False,
                                               index=True, on_delete='CASCADE')
    label = ForeignKeyField(Labels, related_name='predictions', null=False, index=True, on_delete='CASCADE')
    
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


"""

"""

"""

"""

all_tables = [
    # Depth 0
    Dataset,
    Classifier,
    Function,
    Metrics,
    
    # Depth 1
    Task,
    Sequence,
    View,
    
    # Depth 2
    Partition,
    FunctionConfiguration,
    FeatureUnion,
    Labels,
    Data,
    
    # Depth 3
    SplitDefinition,
    Groups,
    Features,
    
    # Depth 4
    PredictionConfiguration,
    Prediction,
    Performance,
]

"""
drop table performance cascade;
drop table predictions cascade;
drop table prediction_configuration  cascade;
drop table features cascade;
drop table groups cascade;
drop table split_definitions  cascade;
drop table data cascade;
drop table labels  cascade;
drop table feature_unions  cascade;
drop table function_configuration cascade;
drop table functions  cascade;
drop table partitions  cascade;
drop table tasks cascade;
drop table sequences cascade;
drop table views cascade;
drop table classifiers cascade;
drop table datasets cascade;
drop table metrics cascade;
"""
