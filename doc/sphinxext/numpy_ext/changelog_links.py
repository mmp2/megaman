# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This sphinx extension makes the issue numbers in the changelog into links to
GitHub issues.
"""

from __future__ import print_function

import re

from docutils.nodes import Text, reference

BLOCK_PATTERN = re.compile('\[#.+\]', flags=re.DOTALL)
ISSUE_PATTERN = re.compile('#[0-9]+')


def process_changelog_links(app, doctree, docname):
    for rex in app.changelog_links_rexes:
        if rex.match(docname):
            break
    else:
        # if the doc doesn't match any of the changelog regexes, don't process
        return

    app.info('[changelog_links] Adding changelog links to "{0}"'.format(docname))

    for item in doctree.traverse():

        if not isinstance(item, Text):
            continue

        # We build a new list of items to replace the current item. If
        # a link is found, we need to use a 'reference' item.
        children = []

        # First cycle through blocks of issues (delimited by []) then
        # iterate inside each one to find the individual issues.
        prev_block_end = 0
        for block in BLOCK_PATTERN.finditer(item):
            block_start, block_end = block.start(), block.end()
            children.append(Text(item[prev_block_end:block_start]))
            block = item[block_start:block_end]
            prev_end = 0
            for m in ISSUE_PATTERN.finditer(block):
                start, end = m.start(), m.end()
                children.append(Text(block[prev_end:start]))
                issue_number = block[start:end]
                refuri = app.config.github_issues_url + issue_number[1:]
                children.append(reference(text=issue_number,
                                          name=issue_number,
                                          refuri=refuri))
                prev_end = end

            prev_block_end = block_end

            # If no issues were found, this adds the whole item,
            # otherwise it adds the remaining text.
            children.append(Text(block[prev_end:block_end]))

        # If no blocks were found, this adds the whole item, otherwise
        # it adds the remaining text.
        children.append(Text(item[prev_block_end:]))

        # Replace item by the new list of items we have generated,
        # which may contain links.
        item.parent.replace(item, children)


def setup_patterns_rexes(app):
    app.changelog_links_rexes = [re.compile(pat) for pat in
                                 app.config.changelog_links_docpattern]


def setup(app):
    app.connect('doctree-resolved', process_changelog_links)
    app.connect('builder-inited', setup_patterns_rexes)
    app.add_config_value('github_issues_url', None, True)
    app.add_config_value('changelog_links_docpattern', ['.*changelog.*', 'whatsnew/.*'], True)
