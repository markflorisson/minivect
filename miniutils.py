"""
Miscellaneous utilities.
"""

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

class ComparableObjectMixin(object):
    "Make sure subclasses implement comparison and hashing methods"

    def __hash__(self):
        "Implement in subclasses"
        raise NotImplementedError

    def __eq__(self, other):
        "Implement in subclasses"
        return NotImplemented