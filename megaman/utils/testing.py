# Author: James McQueen
#
# Edited from the original sklearn:
# Copyright (c) 2011, 2012
# Authors: Pietro Berkes,
#          Andreas Muller
#          Mathieu Blondel
#          Olivier Grisel
#          Arnaud Joly
#          Denis Engemann
#          Giorgio Patrini
# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import warnings
from functools import wraps
import sys
import numpy as np

def clean_warning_registry():
    """Safe way to reset warnings """
    warnings.resetwarnings()
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if 'six.moves' in mod_name:
            continue
        if hasattr(mod, reg):
            getattr(mod, reg).clear()


def assert_raise_message(exceptions, message, function, *args, **kwargs):
    """Helper function to test error messages in exceptions
    Parameters
    ----------
    exceptions : exception or tuple of exception
        Name of the estimator
    func : callable
        Calable object to raise error
    *args : the positional arguments to `func`.
    **kw : the keyword arguments to `func`
    """
    try:
        function(*args, **kwargs)
    except exceptions as e:
        error_message = str(e)
        if message not in error_message:
            raise AssertionError("Error message does not include the expected"
                                 " string: %r. Observed error message: %r" %
                                 (message, error_message))
    else:
        # concatenate exception names
        if isinstance(exceptions, tuple):
            names = " or ".join(e.__name__ for e in exceptions)
        else:
            names = exceptions.__name__

        raise AssertionError("%s not raised by %s" %
                             (names, function.__name__))


def assert_no_warnings(func, *args, **kw):
    # XXX: once we may depend on python >= 2.6, this can be replaced by the

    # warnings module context manager.
    # very important to avoid uncontrolled state propagation
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        result = func(*args, **kw)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        if len(w) > 0:
            raise AssertionError("Got warnings when calling %s: %s"
                                 % (func.__name__, w))
    return result


def assert_warns(warning_class, func, *args, **kw):
    """Test that a certain warning occurs.
    Parameters
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.
    func : callable
        Calable object to trigger warnings.
    *args : the positional arguments to `func`.
    **kw : the keyword arguments to `func`
    Returns
    -------
    result : the return value of `func`
    """

    # very important to avoid uncontrolled state propagation
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        result = func(*args, **kw)
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Filter out numpy-specific warnings in numpy >= 1.9
            w = [e for e in w
                 if e.category is not np.VisibleDeprecationWarning]

        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = any(warning.category is warning_class for warning in w)
        if not found:
            raise AssertionError("%s did not give warning: %s( is %s)"
                                 % (func.__name__, warning_class, w))
    return result


def ignore_warnings(obj=None):
    """ Context manager and decorator to ignore warnings
    Note. Using this (in both variants) will clear all warnings
    from all python modules loaded. In case you need to test
    cross-module-warning-logging this is not your tool of choice.
    Examples
    --------
    >>> with ignore_warnings():
    ...     warnings.warn('buhuhuhu')
    >>> def nasty_warn():
    ...    warnings.warn('buhuhuhu')
    ...    print(42)
    >>> ignore_warnings(nasty_warn)()
    42
    """
    if callable(obj):
        return _ignore_warnings(obj)
    else:
        return _IgnoreWarnings()


def _ignore_warnings(fn):
    """Decorator to catch and hide warnings without visual nesting"""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # very important to avoid uncontrolled state propagation
        clean_warning_registry()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            return fn(*args, **kwargs)
            w[:] = []

    return wrapper


class _IgnoreWarnings(object):

    """Improved and simplified Python warnings context manager
    Copied from Python 2.7.5 and modified as required.
    """

    def __init__(self):
        """
        Parameters
        ==========
        category : warning class
            The category to filter. Defaults to Warning. If None,
            all categories will be muted.
        """
        self._record = True
        self._module = sys.modules['warnings']
        self._entered = False
        self.log = []

    def __repr__(self):
        args = []
        if self._record:
            args.append("record=True")
        if self._module is not sys.modules['warnings']:
            args.append("module=%r" % self._module)
        name = type(self).__name__
        return "%s(%s)" % (name, ", ".join(args))

    def __enter__(self):
        clean_warning_registry()  # be safe and not propagate state + chaos
        warnings.simplefilter('always')
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = self._module.filters
        self._module.filters = self._filters[:]
        self._showwarning = self._module.showwarning
        if self._record:
            self.log = []

            def showwarning(*args, **kwargs):
                self.log.append(warnings.WarningMessage(*args, **kwargs))
            self._module.showwarning = showwarning
            return self.log
        else:
            return None

    def __exit__(self, *exc_info):
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        self._module.filters = self._filters
        self._module.showwarning = self._showwarning
        self.log[:] = []
        clean_warning_registry()  # be safe and not propagate state + chaos
