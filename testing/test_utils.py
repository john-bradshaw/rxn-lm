
from rxn_lm import utils


def test_set_paths_relative_to_given_pth():
    # nb does not test windows paths.
    dict_in = {
        "key1": "relative/path/1",
        "key2": "relative/path/2",
        "key3": ["relative/path/3", "relative/path/4/is/here"],
        "key5": 67123
    }

    base_path = "/tmp/test"

    utils.set_paths_relative_to_given_pth(dict_in, base_path, {"key1", "key3"})

    expected_dict = {
        "key1": "/tmp/test/relative/path/1",
        "key2": "relative/path/2",
        "key3": ["/tmp/test/relative/path/3", "/tmp/test/relative/path/4/is/here"],
        "key5": 67123
    }

    assert expected_dict == dict_in


def test_check_paths_at_keys_absolute():
    dict_in = {
        "key1": "/tmp/test/relative",
        "key2": "../relative/path/2"}
    assert utils.check_paths_at_keys_absolute(dict_in, {"key1"})
    assert not utils.check_paths_at_keys_absolute(dict_in, {"key1", "key2"})
