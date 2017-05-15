import json


def validate_dataset(meta):
    # print json.dumps(meta['dataset'], indent=2, sort_keys=True)
    
    assert 'name' in meta['dataset']
    assert isinstance(meta['dataset']['name'], (str, unicode))
    assert len(meta['dataset']) > 0


def validate_views(meta):
    assert 'views' in meta
    assert len(meta['views']) > 0
    
    for view in meta['views']:
        # print json.dumps(view, indent=2, sort_keys=True)
        
        assert 'name' in view
        assert 'columns' in view
        
        assert isinstance(view['name'], (str, unicode))
        assert isinstance(view['columns'], list)
        
        assert len(view['columns']) > 0


def validate_sequences(meta):
    assert 'sequences' in meta
    assert len(meta['sequences']) > 0
    
    for sequence in meta['sequences']:
        # print json.dumps(sequence, indent=2, sort_keys=True)
        
        assert 'name' in sequence
        assert 't_min' in sequence
        assert 't_max' in sequence
        
        assert isinstance(sequence['name'], (str, unicode))
        assert isinstance(sequence['t_min'], (int, float))
        assert isinstance(sequence['t_max'], (int, float))


def validate_tasks(meta):
    assert 'tasks' in meta
    assert len(meta['tasks']) > 0
    
    for task in meta['tasks']:
        # print json.dumps(task, indent=2, sort_keys=True)
        
        assert 'name' in task
        assert 'columns' in task
        
        assert isinstance(task['name'], (str, unicode))
        assert isinstance(task['columns'], list)
        
        assert len(task['name']) > 0
        assert len(task['columns']) > 0


class Metadata(object):
    def __init__(self, filename):
        meta = json.load(open(filename, 'r'))
        
        validate_dataset(meta)
        validate_sequences(meta)
        validate_tasks(meta)
        validate_views(meta)
        
        self.name = meta['dataset']['name']
        
        self.dataset = meta['dataset']
        self.sequences = meta['sequences']
        self.tasks = meta['tasks']
        self.views = meta['views']
