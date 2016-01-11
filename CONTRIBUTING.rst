Contributing to an HDF Group Project
====================================

Pull requests are always welcome, and The HDF Group dev team appreciates any help the community can
give to help make HDF and related projects better.

For any particular improvement you want to make, you can begin a discussion by creating a issue in Github.
This is the best place discuss your proposed improvement (and its
implementation) with the core development team.

For bugs, please clearly describe the issue you are resolving, including the platforms on which
the issue is present and clear steps to reproduce.  Label the issue as BUG.

For improvements or feature requests, be sure to explain the goal or use case and the approach
your solution will take.  Label the issue as ENHANCEMENT.
 

Getting Started
---------------

- Create a `Github account`_.
- Fork the repository on Github at https://github.com/HDFGroup/hdf5-json.git.
- Run ``python setup.py install`` and ``python testall.py`` and verify that all tests pass.
- Implement your fix/feature
- Run install and test again and verify that nothing got broken
- Submit your pull request. (see https://help.github.com/articles/using-pull-requests/)
  

The Life Cycle of a Pull Request
--------------------------------

Here's what happens when you submit a pull request:

- The core development team will review your pull request to make sure you have included a
  issue # in your request and signed the contributor agreement.
- You should receive a response from one of our engineers with additional questions about your
  contributions.
- Pull requests that have been reviewed and approved will be signed off and merged into a
  development branch and the associated issue will be resolved with an expected
  fix version.


Style Guide
-----------

All commits to Python project in the GitHub repository should follow PEP8 style guide.
 
Testing
-------

Every non-trivial change to the code base should be accompanied by a relevant addition to or
modification of the test suite.  If you do not believe this is necessary, please add an explanation
in the issue ticket why no such changes are either needed or possible.

All changes must also pass the full test suite (including your test additions/changes) on your
local machine before you open a pull request.

Once the pull request is submitted, check Travis CI page (https://travis-ci.org/HDFGroup/hdf5-json) to 
verify that tests are passing for all supported Python versions.


Contributor Agreement
---------------------

A patch will only be considered for merging into the upstream codebase after you have signed the
`contributor agreement`_.  Send an email to info@hdfgroup.org requesting a copy of the agreement.
 