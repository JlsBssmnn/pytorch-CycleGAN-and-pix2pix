import json

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

class Encoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class Empty:
    pass

def getattr_r_no_except(obj, attr):
    """
    Recursively get an attribute of an object. Different attributes can be separated by a period.
    This function will not throw an attribute error if the attribute doesn't exist. Instead, it
    return the attribute error.
    """
    parts = attr.split('.')
    assert len(parts) >= 1

    if not hasattr(obj, parts[0]):
        return AttributeError(f"'{obj.__class__.__name__}' object has no attribute '{parts[0]}'")
    attribute = getattr(obj, parts[0])

    for part in parts[1:]:
        if not hasattr(attribute, part):
            return AttributeError(f"'{attribute.__class__.__name__}' object has no attribute '{part}'")
        attribute = getattr(attribute, part)
    return attribute

def getattr_r(obj, attr):
    """
    Recursively get an attribute of an object. Different attributes can be separated by a period.
    """
    parts = attr.split('.')
    assert len(parts) >= 1
    attribute = getattr(obj, parts[0])

    for part in parts[1:]:
        attribute = getattr(attribute, part)
    return attribute

def setattr_r(obj, attr, value):
    """
    Recursively set an attribute of an object. Different attributes can be separated by a period.
    """
    parts = attr.split('.')
    assert len(parts) >= 1
    attribute = obj

    for part in parts[:-1]:
        if not hasattr(attribute, part):
            setattr(attribute, part, Empty())
        attribute = getattr(attribute, part)

    setattr(attribute, parts[-1], value)
