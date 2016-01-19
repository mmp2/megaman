.. _documentation-example:

Documentation Example
=====================

Lorem ipsum I can't remember the rest...

Here's some IPython code; it can be automatically unit tested when written like this:

.. ipython::

    In [1]: import numpy as np

    In [2]: X = np.random.rand(100, 2)

    In [3]: X.shape
    Out[3]: (100, 2)

Here is a figure built by the matplotlib plugin

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, 1000)
    plt.plot(x, np.sin(x))
    plt.plot(x, np.cos(x))

