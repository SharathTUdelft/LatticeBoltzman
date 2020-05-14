def lazy(fn):
    '''Decorator that makes a method lazy-evaluated.
    # because of lazification its better to work with copies because the originals were being modified
    and these changes were being saved to the object
    '''
    attr_name = '_lazy_' + fn.__name__
    print(fn)


    def _lazy(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy