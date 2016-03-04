__all__ = ["RegisterSubclasses"]


# From six.py
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a
    # dummy metaclass for one level of class instantiation that replaces
    # itself with the actual metaclass.
    class metaclass(type):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, '_TemporaryClass', (), {})


class RegistryMeta(type):
    """Metaclass for object type which registers subclasses"""
    def __init__(cls, name, bases, dct):
        if name in ['_TemporaryClass', 'RegisterSubclasses']:
            # these are baseclasses. Do nothing
            pass
        elif not hasattr(cls, '_method_registry'):
            # this is the base class.  Create an empty registry
            cls._method_registry = {}
        elif hasattr(cls, 'name'):
            # this is a labeled derived class.  Add cls to the registry
            cls._method_registry[cls.name] = cls

        super(RegistryMeta, cls).__init__(name, bases, dct)


class RegisterSubclasses(with_metaclass(RegistryMeta)):
    @classmethod
    def init(cls, method, *args, **kwargs):
        if method not in cls._method_registry:
            raise ValueError("method={0} not valid. Must be one of "
                             "{1}".format(method, list(cls.methods())))
        Adj = cls._method_registry[method]
        return Adj(*args, **kwargs)

    @classmethod
    def methods(cls):
        return cls._method_registry.keys()
