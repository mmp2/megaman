# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

__all__ = ["RegisterSubclasses"]


# From six.py
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # Use a dummy metaclass that replaces itself with the actual metaclass.
    class metaclass(type):
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, '_TemporaryClass', (), {})


class RegistryMeta(type):
    """Metaclass for object type which registers subclasses"""
    def __init__(cls, name, bases, dct):
        if name in ['_TemporaryClass', 'RegisterSubclasses']:
            # these are hidden baseclasses. Do nothing
            pass
        elif not hasattr(cls, '_method_registry'):
            # this is a registry class.  Create an empty registry
            cls._method_registry = {}
        elif hasattr(cls, 'name'):
            # this is a labeled derived class.  Add cls to the registry
            cls._method_registry[cls.name] = cls

        super(RegistryMeta, cls).__init__(name, bases, dct)


class RegisterSubclasses(with_metaclass(RegistryMeta)):
    @classmethod
    def get_method(cls, method):
        if method not in cls._method_registry:
            raise ValueError("method={0} not valid. Must be one of "
                             "{1}".format(method, list(cls.methods())))
        return cls._method_registry[method]

    @classmethod
    def init(cls, method, *args, **kwargs):
        Method = cls.get_method(method)
        return Method(*args, **kwargs)

    @classmethod
    def _remove_from_registry(cls, method):
        cls._method_registry.pop(method, None)

    @classmethod
    def methods(cls):
        return cls._method_registry.keys()
