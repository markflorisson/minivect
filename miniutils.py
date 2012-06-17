import __builtin__

def any(it):
    for obj in it:
        if obj:
            return True
    return False

def all(it):
    for obj in it:
        if not obj:
            return False
    return True

def max(it, key=None):
    if key is not None:
        k, value = max((key(value), value) for value in it)
        return value
    return max(it)

def min(it, key=None):
    if key is not None:
        k, value = min((key(value), value) for value in it)
        return value
    return min(it)

class Condition(object):
    """
    This wraps a condition so that it can be shared by everyone and modified
    by whomever wants to.
    """
    def __init__(self, value):
        self.value = value

    def __nonzero__(self):
        return self.value

class ComparableObjectMixin(object):

    def __hash__(self):
        "Implement in subclasses"
        raise NotImplementedError

    def __eq__(self, other):
        "Implement in subclasses"
        return NotImplemented