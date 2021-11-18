from argparse import Namespace

def recursive_namespace(dd):
    '''
    converts a (possibly nested) dictionary into a namespace
    '''
    for d in dd:
        has_any_dicts = False
        if isinstance(dd[d], dict):
            dd[d] = recursive_namespace(dd[d])
            has_any_dicts = True
    return Namespace(**dd)
    