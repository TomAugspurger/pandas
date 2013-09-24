# pylint: disable-msg=E1101,W0612

from datetime import datetime, timedelta
import operator
import unittest
import nose

import numpy as np
from numpy import nan
import pandas as pd

from pandas import (Index, Series, DataFrame, Panel,
                    isnull, notnull,date_range, _np_version_under1p7)
from pandas.core.index import Index, MultiIndex
from pandas.tseries.index import Timestamp, DatetimeIndex

import pandas.core.common as com

from pandas.compat import StringIO, lrange, range, zip, u, OrderedDict, long
from pandas import compat
from pandas.util.testing import (assert_series_equal,
                                 assert_frame_equal,
                                 assert_panel_equal,
                                 assert_almost_equal,
                                 ensure_clean)
import pandas.util.testing as tm

#------------------------------------------------------------------------------
# Generic types test cases


class Generic(object):

    _multiprocess_can_split_ = True

    def setUp(self):
        import warnings
        warnings.filterwarnings(action='ignore', category=FutureWarning)

    @property
    def _ndim(self):
        return self._typ._AXIS_LEN

    def _axes(self):
        """ return the axes for my object typ """
        return self._typ._AXIS_ORDERS

    def _construct(self, shape, value=None, **kwargs):
        """ construct an object for the given shape
            if value is specified use that if its a scalar
            if value is an array, repeat it as needed """

        if isinstance(shape,int):
            shape = tuple([shape] * self._ndim)
        if value is not None:
            if np.isscalar(value):
                if value == 'empty':
                    arr = None

                    # remove the info axis
                    kwargs.pop(self._typ._info_axis_name,None)
                else:
                    arr = np.empty(shape)
                    arr.fill(value)
            else:
                fshape = np.prod(shape)
                arr = value.ravel()
                new_shape = fshape/arr.shape[0]
                if fshape % arr.shape[0] != 0:
                    raise Exception("invalid value passed in _construct")

                arr = np.repeat(arr,new_shape).reshape(shape)
        else:
            arr = np.random.randn(*shape)
        return self._typ(arr,**kwargs)

    def _compare(self, result, expected):
        self._comparator(result,expected)

    def test_rename(self):

        # single axis
        for axis in self._axes():
            kwargs = { axis : list('ABCD') }
            obj = self._construct(4,**kwargs)

            # no values passed
            #self.assertRaises(Exception, o.rename(str.lower))

            # rename a single axis
            result = obj.rename(**{ axis : str.lower })
            expected = obj.copy()
            setattr(expected,axis,list('abcd'))
            self._compare(result, expected)

        # multiple axes at once

    def test_get_numeric_data(self):

        n = 4
        kwargs = { }
        for i in range(self._ndim):
            kwargs[self._typ._AXIS_NAMES[i]] = list(range(n))

        # get the numeric data
        o = self._construct(n,**kwargs)
        result = o._get_numeric_data()
        self._compare(result, o)

        # non-inclusion
        result = o._get_bool_data()
        expected = self._construct(n,value='empty',**kwargs)
        self._compare(result,expected)

        # get the bool data
        arr = np.array([True,True,False,True])
        o = self._construct(n,value=arr,**kwargs)
        result = o._get_numeric_data()
        self._compare(result, o)

        # _get_numeric_data is includes _get_bool_data, so can't test for non-inclusion
    def test_nonzero(self):

        # GH 4633
        # look at the boolean/nonzero behavior for objects
        obj = self._construct(shape=4)
        self.assertRaises(ValueError, lambda : bool(obj == 0))
        self.assertRaises(ValueError, lambda : bool(obj == 1))
        self.assertRaises(ValueError, lambda : bool(obj))

        obj = self._construct(shape=4,value=1)
        self.assertRaises(ValueError, lambda : bool(obj == 0))
        self.assertRaises(ValueError, lambda : bool(obj == 1))
        self.assertRaises(ValueError, lambda : bool(obj))

        obj = self._construct(shape=4,value=np.nan)
        self.assertRaises(ValueError, lambda : bool(obj == 0))
        self.assertRaises(ValueError, lambda : bool(obj == 1))
        self.assertRaises(ValueError, lambda : bool(obj))

        # empty
        obj = self._construct(shape=0)
        self.assertRaises(ValueError, lambda : bool(obj))

        # invalid behaviors

        obj1 = self._construct(shape=4,value=1)
        obj2 = self._construct(shape=4,value=1)

        def f():
            if obj1:
                print("this works and shouldn't")
        self.assertRaises(ValueError, f)
        self.assertRaises(ValueError, lambda : obj1 and obj2)
        self.assertRaises(ValueError, lambda : obj1 or obj2)
        self.assertRaises(ValueError, lambda : not obj1)


class TestSeries(unittest.TestCase, Generic):
    _typ = Series
    _comparator = lambda self, x, y: assert_series_equal(x,y)

    def setUp(self):
        self.ts = tm.makeTimeSeries()  # Was at top level in test_series
        self.ts.name = 'ts'

        self.series = tm.makeStringSeries()
        self.series.name = 'series'

    def test_rename_mi(self):
        s = Series([11,21,31],
                   index=MultiIndex.from_tuples([("A",x) for x in ["a","B","c"]]))
        result = s.rename(str.lower)

    def test_get_numeric_data_preserve_dtype(self):

        # get the numeric data
        o = Series([1,2,3])
        result = o._get_numeric_data()
        self._compare(result, o)

        o = Series([1,'2',3.])
        result = o._get_numeric_data()
        expected = Series([],dtype=object)
        self._compare(result, expected)

        o = Series([True,False,True])
        result = o._get_numeric_data()
        self._compare(result, o)

        o = Series([True,False,True])
        result = o._get_bool_data()
        self._compare(result, o)

        o = Series(date_range('20130101',periods=3))
        result = o._get_numeric_data()
        expected = Series([],dtype='M8[ns]')
        self._compare(result, expected)

    def test_nonzero_single_element(self):

        s = Series([True])
        self.assertRaises(ValueError, lambda : bool(s))

        s = Series([False])
        self.assertRaises(ValueError, lambda : bool(s))

    def test_interpolate(self):
        ts = Series(np.arange(len(self.ts), dtype=float), self.ts.index)

        ts_copy = ts.copy()
        ts_copy[5:10] = np.NaN

        linear_interp = ts_copy.interpolate(method='linear')
        self.assert_(np.array_equal(linear_interp, ts))

        ord_ts = Series([d.toordinal() for d in self.ts.index],
                        index=self.ts.index).astype(float)

        ord_ts_copy = ord_ts.copy()
        ord_ts_copy[5:10] = np.NaN

        time_interp = ord_ts_copy.interpolate(method='time')
        self.assert_(np.array_equal(time_interp, ord_ts))

        # try time interpolation on a non-TimeSeries
        self.assertRaises(Exception, self.series.interpolate, method='time')

    def test_interpolate_corners(self):
        s = Series([np.nan, np.nan])
        assert_series_equal(s.interpolate(), s)

        s = Series([]).interpolate()
        assert_series_equal(s.interpolate(), s)

    def test_interpolate_index_values(self):
        s = Series(np.nan, index=np.sort(np.random.rand(30)))
        s[::3] = np.random.randn(10)

        vals = s.index.values.astype(float)

        result = s.interpolate(method='values')

        expected = s.copy()
        bad = isnull(expected.values)
        good = -bad
        expected = Series(
            np.interp(vals[bad], vals[good], s.values[good]), index=s.index[bad])

        assert_series_equal(result[bad], expected)

    def test_timedelta_fillna(self):
        if _np_version_under1p7:
            raise nose.SkipTest("timedelta broken in np 1.6.1")

        #GH 3371
        s = Series([Timestamp('20130101'), Timestamp('20130101'),
                    Timestamp('20130102'), Timestamp('20130103 9:01:01')])
        td = s.diff()

        # reg fillna
        result = td.fillna(0)
        expected = Series([timedelta(0), timedelta(0), timedelta(1),
                           timedelta(days=1, seconds=9*3600+60+1)])
        assert_series_equal(result, expected)

        # interprested as seconds
        result = td.fillna(1)
        expected = Series([timedelta(seconds=1), timedelta(0),
                           timedelta(1), timedelta(days=1, seconds=9*3600+60+1)])
        assert_series_equal(result, expected)

        result = td.fillna(timedelta(days=1, seconds=1))
        expected = Series([timedelta(days=1, seconds=1), timedelta(0),
                           timedelta(1), timedelta(days=1, seconds=9*3600+60+1)])
        assert_series_equal(result, expected)

        result = td.fillna(np.timedelta64(int(1e9)))
        expected = Series([timedelta(seconds=1), timedelta(0), timedelta(1),
                           timedelta(days=1, seconds=9*3600+60+1)])
        assert_series_equal(result, expected)

        from pandas import tslib
        result = td.fillna(tslib.NaT)
        expected = Series([tslib.NaT, timedelta(0), timedelta(1),
                           timedelta(days=1, seconds=9*3600+60+1)], dtype='m8[ns]')
        assert_series_equal(result, expected)

        # ffill
        td[2] = np.nan
        result = td.ffill()
        expected = td.fillna(0)
        expected[0] = np.nan
        assert_series_equal(result, expected)

        # bfill
        td[2] = np.nan
        result = td.bfill()
        expected = td.fillna(0)
        expected[2] = timedelta(days=1, seconds=9*3600+60+1)
        assert_series_equal(result, expected)

    def test_datetime64_fillna(self):

        s = Series([Timestamp('20130101'), Timestamp('20130101'),
                    Timestamp('20130102'), Timestamp('20130103 9:01:01')])
        s[2] = np.nan

        # reg fillna
        result = s.fillna(Timestamp('20130104'))
        expected = Series([Timestamp('20130101'), Timestamp('20130101'),
                           Timestamp('20130104'), Timestamp('20130103 9:01:01')])
        assert_series_equal(result, expected)

        from pandas import tslib
        result = s.fillna(tslib.NaT)
        expected = s
        assert_series_equal(result, expected)

        # ffill
        result = s.ffill()
        expected = Series([Timestamp('20130101'), Timestamp('20130101'),
                           Timestamp('20130101'), Timestamp('20130103 9:01:01')])
        assert_series_equal(result, expected)

        # bfill
        result = s.bfill()
        expected = Series([Timestamp('20130101'), Timestamp('20130101'),
                           Timestamp('20130103 9:01:01'),
                           Timestamp('20130103 9:01:01')])
        assert_series_equal(result, expected)

    def test_fillna_int(self):
        s = Series(np.random.randint(-100, 100, 50))
        s.fillna(method='ffill', inplace=True)
        assert_series_equal(s.fillna(method='ffill', inplace=False), s)

    def test_fillna_raise(self):
        s = Series(np.random.randint(-100, 100, 50))
        self.assertRaises(TypeError, s.fillna, [1, 2])
        self.assertRaises(TypeError, s.fillna, (1, 2))

# TimeSeries-specific

    def test_fillna(self):
        ts = Series([0., 1., 2., 3., 4.], index=tm.makeDateIndex(5))

        self.assert_(np.array_equal(ts, ts.fillna(method='ffill')))

        ts[2] = np.NaN

        self.assert_(
            np.array_equal(ts.fillna(method='ffill'), [0., 1., 1., 3., 4.]))
        self.assert_(np.array_equal(ts.fillna(method='backfill'),
                                    [0., 1., 3., 3., 4.]))

        self.assert_(np.array_equal(ts.fillna(value=5), [0., 1., 5., 3., 4.]))

        self.assertRaises(ValueError, ts.fillna)
        self.assertRaises(ValueError, self.ts.fillna, value=0, method='ffill')

    def test_fillna_bug(self):
        x = Series([nan, 1., nan, 3., nan], ['z', 'a', 'b', 'c', 'd'])
        filled = x.fillna(method='ffill')
        expected = Series([nan, 1., 1., 3., 3.], x.index)
        assert_series_equal(filled, expected)

        filled = x.fillna(method='bfill')
        expected = Series([1., 1., 3., 3., nan], x.index)
        assert_series_equal(filled, expected)

    def test_fillna_inplace(self):
        x = Series([nan, 1., nan, 3., nan], ['z', 'a', 'b', 'c', 'd'])
        y = x.copy()

        y.fillna(value=0, inplace=True)

        expected = x.fillna(value=0)
        assert_series_equal(y, expected)

    def test_fillna_invalid_method(self):
        try:
            self.ts.fillna(method='ffil')
        except ValueError as inst:
            self.assert_('ffil' in str(inst))

    def test_ffill(self):
        ts = Series([0., 1., 2., 3., 4.], index=tm.makeDateIndex(5))
        ts[2] = np.NaN
        assert_series_equal(ts.ffill(), ts.fillna(method='ffill'))

    def test_bfill(self):
        ts = Series([0., 1., 2., 3., 4.], index=tm.makeDateIndex(5))
        ts[2] = np.NaN
        assert_series_equal(ts.bfill(), ts.fillna(method='bfill'))

    # New interpolation tests
    def test_nan_interpolate(self):
        s = Series([0, 1, np.nan, 3])
        result = s.interpolate()
        expected = Series([0, 1, 2, 3], dtype=np.float64)
        assert_series_equal(result, expected)

    def test_nan_irregular_index(self):
        s = Series([1, 2, np.nan, 4], index=[1, 3, 5, 9])
        result = s.interpolate()
        expected = Series([1., 2., 3., 4.], index=[1, 3, 5, 9])  # backwards compat :(
        assert_series_equal(result, expected)

    def test_nan_str_index(self):
        s = Series([0, 1, 2, np.nan], index=list('abcd'))
        result = s.interpolate()
        expected = Series([0., 1., 2., 2.], index=list('abcd'))  # endpoints weird
        assert_series_equal(result, expected)

    def test_interp_quad(self):
        sq = Series([1, 4, np.nan, 16], index=[1, 2, 3, 4])
        result = sq.interpolate(method='quadratic')
        expected = Series([1., 4., 9., 16.], index=[1, 2, 3, 4])
        assert_series_equal(result, expected)

    def test_interp_scipy_basic(self):
        s = Series([1, 3, np.nan, 12, np.nan, 25])
        # slinear
        expected = Series([1., 3., 7.5, 12., 18.5, 25.])
        result = s.interpolate(method='slinear')
        assert_series_equal(result, expected)
        # nearest
        expected = Series([1., 3., 3., 12., 12., 25.])
        result = s.interpolate(method='nearest')
        assert_series_equal(result, expected)
        # zero
        expected = Series([1., 3., 3., 12., 12., 25.])
        result = s.interpolate(method='zero')
        assert_series_equal(result, expected)
        # quadratic
        expected = Series([1, 3., 6.769231, 12., 18.230769, 25.])
        result = s.interpolate(method='quadratic')
        assert_series_equal(result, expected)
        # cubic
        expected = Series([1., 3., 6.8, 12., 18.2, 25.])
        result = s.interpolate(method='cubic')
        assert_series_equal(result, expected)

    def test_interp_limit(self):
        #failing. not implemented
        s = Series([1, 3, np.nan, np.nan, np.nan, 11])
        expected = Series([1., 3., 5., 7., np.nan, 11.])
        result = s.interpolate(method='linear', limit=2)
        assert_series_equal(result, expected)

        result = s.interpolate(method='quadratic', limit=2)
        assert_series_equal(result, expected)

        df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5, np.nan, 7],
                           'B': [1, 4, 9, np.nan, np.nan, np.nan, 49],
                           'C': [1, 2, 3, 5, 8, 13, 21], 'D': list('abcdefg')})


class TestDataFrame(unittest.TestCase, Generic):
    _typ = DataFrame
    _comparator = lambda self, x, y: assert_frame_equal(x,y)

    def test_rename_mi(self):
        df = DataFrame([11,21,31],
                       index=MultiIndex.from_tuples([("A",x) for x in ["a","B","c"]]))
        result = df.rename(str.lower)

    def test_get_numeric_data_preserve_dtype(self):

        # get the numeric data
        o = DataFrame({'A' : [1,'2',3.] })
        result = o._get_numeric_data()
        expected = DataFrame(index=[0,1,2],dtype=object)
        self._compare(result, expected)

    def test_interp_basic(self):
        df = DataFrame({'A': [1, 2, np.nan, 4], 'B': [1, 4, 9, np.nan],
                        'C': [1, 2, 3, 5], 'D': list('abcd')})
        expected = DataFrame({'A': [1, 2, 3, 4], 'B': [1, 4, 9, 9],
                              'C': [1, 2, 3, 5], 'D': list('abcd')})
        result = df.interpolate()
        assert_frame_equal(result, expected)

        result = df.interpolate(index='C')
        expected = DataFrame({'A': [1, 2, 2.6666667, 4], 'B': [1, 4, 9, 9],
                              'C': [1, 2, 3, 5], 'D': list('abcd')})
        assert_frame_equal(result, expected)

    def test_interp_combo(self):
        df = DataFrame({'A': [1., 2., np.nan, 4.], 'B': [1, 4, 9, np.nan],
                        'C': [1., 2., 3., 5.], 'D': list('abcd')})

        # these should return all the columns (but just interpolate on A)
        result = df.interpolate(values='A')
        expected = df.copy()
        expected['A'] = Series([1,2,3,4])
        assert_frame_equal(result, expected)

        ## is the expected right here?
        result = df.interpolate(values='A', index='C')
        expected = df.copy()
        expected['A'] = Series([1,2,3,4])
        assert_frame_equal(result, expected)

    def test_interp_leading_nan(self):
        df = DataFrame({'A': [1, 2, np.nan, 4], 'B': [np.nan, 2, 3, 4]})
        result = df.interpolate(values='A', index='B')
        expected = DataFrame({'A': [1, 2, 3, 4], 'B' : df['B']})
        assert_frame_equal(result, expected)

    def test_interp_various(self):
        df = DataFrame({'A': [1, 2, np.nan, 4, 5, np.nan, 7],
                        'C': [1, 2, 3, 5, 8, 13, 21]})
        result = df.interpolate(index='C', values='C', method=1)
        expected = df[['A', 'C']].copy()
        expected.A.loc[2] = 2.66666667
        expected.A.loc[4] = 5.76923077
        assert_frame_equal(result, expected)

        result = df.interpolate(index='C', values='C', method='cubic')
        expected.A.loc[2] = 2.81621174
        expected.A.loc[4] = 5.64146581
        assert_frame_equal(result, expected)

        result = df.interpolate(index='C', values='C', method='nearest')
        expected.A.loc[2] = 2
        expected.A.loc[4] = 5
        assert_frame_equal(result, expected)

        result = df.interpolate(index='C', values='C', method='quadratic')
        expected.A.loc[2] = 2.82533638
        expected.A.loc[4] = 6.02817974
        assert_frame_equal(result, expected)

        result = df.interpolate(index='C', values='C', method='slinear')
        expected.A.loc[2] = 2.66666667
        expected.A.loc[4] = 5.76923077
        assert_frame_equal(result, expected)

        result = df.interpolate(index='C', values='C', method='zero')
        expected.A.loc[2] = 2.
        expected.A.loc[4] = 5
        assert_frame_equal(result, expected)

class TestPanel(unittest.TestCase, Generic):
    _typ = Panel
    _comparator = lambda self, x, y: assert_panel_equal(x,y)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
