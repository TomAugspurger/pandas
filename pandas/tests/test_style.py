import copy
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.core.style import Styler
# import pandas.util.testing as tm
# import pandas as pd

class TestStyler(TestCase):

    def setUp(self):
        np.random.seed(24)
        self.s = DataFrame({'A': np.random.permutation(range(6))})
        self.df = DataFrame({'A': [0, 1], 'B': np.random.randn(2)})
        self.f = lambda x: x
        self.g = lambda x: x
        self.styler = Styler(self.df)
        self.attrs = pd.DataFrame({'A': ['color: red', 'color: blue']})

    # def test_builder(self):
    #     styler = Styler(self.df, table=self.even_red)
    #     result = styler._build_styles(styler.rows, styler.columns, styler.table)
    #     expected = pd.DataFrame()

    def test_update_ctx(self):
        self.styler.update_ctx(self.attrs)
        expected = {(0, 'A'): ['color: red'],
                    (1, 'A'): ['color: blue']}
        self.assertEqual(self.styler.ctx, expected)

    def test_update_ctx_flatten_multi(self):
        attrs = DataFrame({"A": ['color: red; foo: bar',
                                 'color: blue; foo: baz']})
        self.styler.update_ctx(attrs)
        expected = {(0, 'A'): ['color: red', ' foo: bar'],
                    (1, 'A'): ['color: blue', ' foo: baz']}
        self.assertEqual(self.styler.ctx, expected)

    def test_update_ctx_flatten_multi_traliing_semi(self):
        attrs = DataFrame({"A": ['color: red; foo: bar;',
                                 'color: blue; foo: baz;']})
        self.styler.update_ctx(attrs)
        expected = {(0, 'A'): ['color: red', ' foo: bar'],
                    (1, 'A'): ['color: blue', ' foo: baz']}
        self.assertEqual(self.styler.ctx, expected)

    def test_copy(self):
        s2 = copy.copy(self.styler)
        self.assertIsNot(self.styler, s2)

        self.styler.update_ctx(self.attrs)
        s2 = copy.copy(self.styler)
        self.assertEqual(self.styler.ctx, s2.ctx)

    def test_render(self):
        df = pd.DataFrame({"A": [0, 1]})
        style = lambda x: pd.Series(["color: red", "color: blue"], name=x.name)
        s = Styler(df, uuid='AB').apply(style)
        s.render()
        # it worked?

    def test_render_double(self):
        df = pd.DataFrame({"A": [0, 1]})
        style = lambda x: pd.Series(["color: red; border: 1px",
                                     "color: blue; border: 2px"], name=x.name)
        s = Styler(df, uuid='AB').apply(style)
        s.render()
        # it worked?kkk

