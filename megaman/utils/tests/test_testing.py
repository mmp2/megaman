# LICENSE: Simplified BSD https://github.com/mmp2/megaman/blob/master/LICENSE

import warnings
import sys
import unittest
from nose.tools import assert_raises, assert_equal

from megaman.utils.testing import assert_raise_message, assert_no_warnings, assert_warns

def test_assert_raise_message():
    def _raise_ValueError(message):
        raise ValueError(message)

    def _no_raise():
        pass

    assert_raise_message(ValueError, "test",
                         _raise_ValueError, "test")

    assert_raises(AssertionError,
                  assert_raise_message, ValueError, "something else",
                  _raise_ValueError, "test")

    assert_raises(ValueError,
                  assert_raise_message, TypeError, "something else",
                  _raise_ValueError, "test")

    assert_raises(AssertionError,
                  assert_raise_message, ValueError, "test",
                  _no_raise)

    # multiple exceptions in a tuple
    assert_raises(AssertionError,
                  assert_raise_message, (ValueError, AttributeError),
                  "test", _no_raise)


# This class is inspired from numpy 1.7 with an alteration to check
# the reset warning filters after calls to assert_warns.
# This assert_warns behavior is specific to scikit-learn because
#`clean_warning_registry()` is called internally by assert_warns
# and clears all previous filters.
class TestWarns(unittest.TestCase):
    def test_warn(self):
        def f():
            warnings.warn("yo")
            return 3

        # Test that assert_warns is not impacted by externally set
        # filters and is reset internally.
        # This is because `clean_warning_registry()` is called internally by
        # assert_warns and clears all previous filters.
        warnings.simplefilter("ignore", UserWarning)
        assert_equal(assert_warns(UserWarning, f), 3)

        # Test that the warning registry is empty after assert_warns
        assert_equal(sys.modules['warnings'].filters, [])

        assert_raises(AssertionError, assert_no_warnings, f)
        assert_equal(assert_no_warnings(lambda x: x, 1), 1)

    def test_warn_wrong_warning(self):
        def f():
            warnings.warn("yo", DeprecationWarning)

        failed = False
        filters = sys.modules['warnings'].filters[:]
        try:
            try:
                # Should raise an AssertionError
                assert_warns(UserWarning, f)
                failed = True
            except AssertionError:
                pass
        finally:
            sys.modules['warnings'].filters = filters

        if failed:
            raise AssertionError("wrong warning caught by assert_warn")
