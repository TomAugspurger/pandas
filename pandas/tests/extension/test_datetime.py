import numpy as np
import pytest

from pandas.core.dtypes.dtypes import DatetimeTZDtype

import pandas as pd
from pandas.core.arrays import DatetimeArrayMixin as DatetimeArray
from pandas.tests.extension import base


@pytest.fixture(params=[None, "US/Central"])
def dtype(request):
    return DatetimeTZDtype(unit="ns", tz=request.param)


@pytest.fixture
def data(dtype):
    data = DatetimeArray(pd.date_range("2000", periods=100, tz=dtype.tz),
                         tz=dtype.tz)
    return data


@pytest.fixture
def data_missing(dtype):
    return DatetimeArray(
        np.array(['NaT', '2000-01-01'], dtype='datetime64[ns]'),
        tz=dtype.tz
    )


@pytest.fixture
def data_for_sorting(dtype):
    a = pd.Timestamp('2000-01-01')
    b = pd.Timestamp('2000-01-02')
    c = pd.Timestamp('2000-01-03')
    return DatetimeArray(np.array([b, c, a], dtype='datetime64[ns]'),
                         tz=dtype.tz)


@pytest.fixture
def data_missing_for_sorting(dtype):
    a = pd.Timestamp('2000-01-01')
    b = pd.Timestamp('2000-01-02')
    return DatetimeArray(np.array([b, 'NaT', a], dtype='datetime64[ns]'),
                         tz=dtype.tz)


@pytest.fixture
def data_for_grouping(dtype):
    """
        Expected to be like [B, B, NA, NA, A, A, B, C]

        Where A < B < C and NA is missing
    """
    a = pd.Timestamp('2000-01-01')
    b = pd.Timestamp('2000-01-02')
    c = pd.Timestamp('2000-01-03')
    na = 'NaT'
    return DatetimeArray(np.array([b, b, na, na, a, a, b, c],
                                  dtype='datetime64[ns]'),
                         tz=dtype.tz)


@pytest.fixture
def na_cmp():
    def cmp(a, b):
        return a is pd.NaT and a is b
    return cmp


@pytest.fixture
def na_value():
    return pd.NaT


# ----------------------------------------------------------------------------
class BaseDatetimeTests(object):
    pass


# ----------------------------------------------------------------------------
# Tests
class TestDatetimeDtype(BaseDatetimeTests, base.BaseDtypeTests):
    pass


class TestConstructors(BaseDatetimeTests, base.BaseConstructorsTests):
    pass


class TestGetitem(BaseDatetimeTests, base.BaseGetitemTests):
    pass


class TestMethods(BaseDatetimeTests, base.BaseMethodsTests):
    @pytest.mark.xfail(reason='GH-22843', strict=True)
    def test_value_counts(self, all_data, dropna):
        # fails without .value_counts
        return super().test_value_counts(all_data, dropna)

    def test_apply_simple_series(self, data):
        if data.tz:
            # fails without .map
            raise pytest.xfail('GH-23179')
        super().test_apply_simple_series(data)

    def test_combine_add(self, data_repeated):
        # Timestamp.__add__(Timestamp) not defined
        pass


class TestInterface(BaseDatetimeTests, base.BaseInterfaceTests):

    @pytest.mark.xfail(reason="Figure out np.array(tz_aware)", strict=False)
    def test_array_interface(self, data):
        # override, because np.array(data)[0] != data[0]
        # since numpy datetime64ns scalars don't compare equal
        # to timestmap objects.
        result = np.array(data)
        # even this fails, since arary(data) is *not* tz aware, and
        # we don't compare tz-aware and tz-naive.
        # this could work if array(data) was object-dtype with timestamps.
        assert data[0] == result[0]


class TestArithmeticOps(BaseDatetimeTests, base.BaseArithmeticOpsTests):
    implements = {'__sub__', '__rsub__'}


class TestCasting(BaseDatetimeTests, base.BaseCastingTests):
    pass


class TestComparisonOps(BaseDatetimeTests, base.BaseComparisonOpsTests):

    def _compare_other(self, s, data, op_name, other):
        # the base test is not appropriate for us. We raise on comparison
        # with (some) integers, depending on the value.
        pass


class TestMissing(BaseDatetimeTests, base.BaseMissingTests):
    pass


class TestReshaping(BaseDatetimeTests, base.BaseReshapingTests):
    pass


class TestSetitem(BaseDatetimeTests, base.BaseSetitemTests):
    pass


class TestGroupby(BaseDatetimeTests, base.BaseGroupbyTests):
    pass
