import pytest


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason='Not in "only_run_named_tests"')

    for item in items:
        # If the class containing this test has the attribute "only_run_named_tests"
        # then tests with names not in that collection are skipped.
        include_list = getattr(item.cls, "only_run_named_tests", None)
        if include_list and item.originalname not in include_list:
            item.add_marker(skip)
