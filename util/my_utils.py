def object_to_dict(obj):
    """
    Converts an object to a dict. Ignores all methods and other callables,
    as well as private fields starting with '_'. Other objects (which excludes
    lists, sets, dicts or tuples) are recursively converted to dicts too.
    """
    d = {}
    for attr in dir(obj):
        value = getattr(obj, attr)
        if attr.startswith('_') or callable(value):
            continue
        elif hasattr(value, '__dict__'):
            d[attr] = object_to_dict(value)
        else:
            d[attr] = value
    return d
