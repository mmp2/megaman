# How to Release

Here's a quick step-by-step for cutting a new release of megaman.

## Pre-release

1. update version in ``megaman/__init__.py`` to, e.g. "0.1"

2. update version in **two places** in ``doc/conf.py`` to the same

3. create a release tag; e.g.
   ```
   $ git tag -a v0.1 -m 'version 0.1 release'
   ```

4. push the commits and tag to github

5. confirm that CI tests pass on github

6. under "tags" on github, update the release notes


## Publishing the Release

1. push the new release to PyPI (requires jakevdp's permissions)
   ```
   $ python setup.py sdist upload
   ```

2. change directories to ``doc`` and build the documentation:
   ```
   $ cd doc/
   $ make html     # build documentation
   $ make publish  # publish to github pages
   ```

3. Publish the conda build (requires jakevdp's permissions)
   (see [conda docs](http://conda.pydata.org/docs/build_tutorials/pkgs2.html) for more information)

    1. update the version, commit tag, and build number in ``conda_recipes/megaman/meta.yml``
    2. commit this to the repository
    3. On both a Linux and OSX system, build the packages:
       ```
       $ cd conda_recipes
       $ anaconda login  # login to anaconda cloud
       $ conda config --set anaconda_upload yes  # auto upload good builds
       $ conda build --py all megaman  # build on all python versions
       ```

## Post-release

1. update version in ``megaman/__init__.py`` to next version; e.g. '0.2.dev0'

2. update version in ``doc/conf.py`` to the same (in two places)

3. push changes to github
