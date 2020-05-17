import numpy as np



def lazy(fn, *args, **kwargs):
    '''Decorator that makes a method lazy-evaluated.
    # because of lazification its better to work with copies because the originals were being modified
    and these changes were being saved to the object
    '''
    attr_name = '_lazy_' + fn.__name__
    print(fn)

    def _lazy(self, *args, **kwargs):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy


class Singleton:
    """
    Metaclass to make sure that only one instance of the derived classes are created
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print('Creating the singleton domain object')
            cls._instance = super(Singleton, cls).__new__(cls)
            # Put any initialization here.
        return cls._instance







