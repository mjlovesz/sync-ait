import unittest
import pytest
import sys
import os
from unittest import mock

sys.path.insert(0, os.path.abspath("../../../")) ##保证amit入口

from debug.compare.common import utils


def test_check_shape_number():
    with pytest.raises(utils.AccuracyCompareException) as error:
        utils._check_shape_number("input:")
    assert error.value.error_info == utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR

    with pytest.raises(utils.AccuracyCompareException) as error:
        utils._check_shape_number("8-8")
    assert error.value.error_info == utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR

    utils._check_shape_number("88")


