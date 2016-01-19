import inspect
import sys


def find_mod_objs(modname, onlylocals=False):
    """ Returns all the public attributes of a module referenced by name.

    .. note::
        The returned list *not* include subpackages or modules of
        `modname`,nor does it include private attributes (those that
        beginwith '_' or are not in `__all__`).

    Parameters
    ----------
    modname : str
        The name of the module to search.
    onlylocals : bool
        If True, only attributes that are either members of `modname` OR one of
        its modules or subpackages will be included.

    Returns
    -------
    localnames : list of str
        A list of the names of the attributes as they are named in the
        module `modname` .
    fqnames : list of str
        A list of the full qualified names of the attributes (e.g.,
        ``astropy.utils.misc.find_mod_objs``). For attributes that are
        simple variables, this is based on the local name, but for
        functions or classes it can be different if they are actually
        defined elsewhere and just referenced in `modname`.
    objs : list of objects
        A list of the actual attributes themselves (in the same order as
        the other arguments)

    """

    __import__(modname)
    mod = sys.modules[modname]

    if hasattr(mod, '__all__'):
        pkgitems = [(k, mod.__dict__[k]) for k in mod.__all__]
    else:
        pkgitems = [(k, mod.__dict__[k]) for k in dir(mod) if k[0] != '_']

    # filter out modules and pull the names and objs out
    ismodule = inspect.ismodule
    localnames = [k for k, v in pkgitems if not ismodule(v)]
    objs = [v for k, v in pkgitems if not ismodule(v)]

    # fully qualified names can be determined from the object's module
    fqnames = []
    for obj, lnm in zip(objs, localnames):
        if hasattr(obj, '__module__') and hasattr(obj, '__name__'):
            fqnames.append(obj.__module__ + '.' + obj.__name__)
        else:
            fqnames.append(modname + '.' + lnm)

    if onlylocals:
        valids = [fqn.startswith(modname) for fqn in fqnames]
        localnames = [e for i, e in enumerate(localnames) if valids[i]]
        fqnames = [e for i, e in enumerate(fqnames) if valids[i]]
        objs = [e for i, e in enumerate(objs) if valids[i]]

    return localnames, fqnames, objs
