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

3. Publish the conda build:
   submit a PR to http://github.com/conda-forge/megaman-feedstock
   updating recipe/meta.yaml with the appropriate version. Once merged,
   then the conda install command will point to the new version.

## Post-release

1. update version in ``megaman/__init__.py`` to next version; e.g. '0.2.dev0'

2. update version in ``doc/conf.py`` to the same (in two places)

3. push changes to github
