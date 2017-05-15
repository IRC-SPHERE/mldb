from models import *
import json


def main(connection=None):
    print 'Feature union creation interface.'
    print
    
    if connection is None:
        print 'The following datasets are available:'
        for dataset in Dataset.select():
            print '  * {}'.format(dataset.name)
        
        '''
        DATASET
        '''
        dataset_name = raw_input('\nEnter the name of the dataset: ').strip()
        dataset = Dataset.get(Dataset.name == dataset_name)
        print
        
    else:
        dataset = connection.dataset
        dataset_name = dataset.name
    
    '''
    TASKS
    '''
    print 'The following tasks are available on the {} dataset'.format(dataset_name)
    for task in dataset.tasks:
        print '  * {}'.format(task.name)
    task_name = raw_input('\nEnter the task name for would you like to collate features: ').strip()
    task = Task.get(
        Task.dataset == dataset,
        Task.name == task_name
    )
    
    '''
    Feature functions
    '''
    print 'The following features have been extracted for this task:'
    all_function_ids = set()
    for function in Function.select().filter(Function.task == task).order_by(Function.id):
        print '  {}) {:>30s}({}, {}]'.format(
            function.id,
            function.name,
            function.left,
            function.right,
        )
        
        all_function_ids |= {function.id}
    
    """
    Obtain the feature union definition
    """
    fids = raw_input('\nEnter the IDs of the functions that you would like to collate (JSON format): ').strip()
    if fids[0] != '[':
        fids = '[' + fids
    if fids[-1] != '[':
        fids += ']'
    fids = sorted(set(json.loads(fids)))
    
    """
    Validate the inputs
    """
    for fid in fids:
        assert isinstance(fid, int)
        
        if fid not in all_function_ids:
            raise ValueError('ERROR: Function ID not in the set of candidates listed. ')
    
    """
    Obtain the union name, and create the instance
    """
    fu_name = raw_input('\nEnter the name for this feature union: ').strip()
    try:
        FeatureUnion.get(
            FeatureUnion.task == task,
            FeatureUnion.name == fu_name
        )
        raise ValueError("ERROR: A feature union of this name exists already.")
    
    except FeatureUnion.DoesNotExist:
        FeatureUnion.create(
            task=task,
            name=fu_name,
            function_ids=fids
        )


if __name__ == '__main__':
    main()
