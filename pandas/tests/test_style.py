from unittest import TestCase

import numpy as np
from pandas import Series, DataFrame
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

    @staticmethod
    def color_bg_range(s, cmap='PuBu'):
        """Color background in a range."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex
        colors = [rgb2hex(x) for x in plt.cm.get_cmap(cmap)(s)]
        return Series([('backgroud-color', color) for color in colors])

    @staticmethod
    def even_red(df):
        return df.applymap(lambda x: ('background-color', 'red') if x % 2 == 0
                           else '')

    def test_format_row_single(self):
        expected = {0: [self.f], 1: [self.f]}, {}, []
        result = self.styler.format_style_args(self.df, rows=self.f)
        self.assertEqual(result, expected)

        result = self.styler.format_style_args(self.df, rows=[self.f])
        self.assertEqual(result, expected)

    def test_format_row_dict(self):
        expected = {0: [self.f]}, {}, []
        result = self.styler.format_style_args(self.df, rows={0: self.f})
        self.assertEqual(result, expected)

    def test_format_col_single(self):
        expected = {}, {'A': [self.f], 'B': [self.f]}, []
        result = self.styler.format_style_args(self.df, columns=self.f)
        self.assertEqual(result, expected)

        result = self.styler.format_style_args(self.df, columns=[self.f])
        self.assertEqual(result, expected)

    def test_format_col_dict(self):
        expected = {}, {'A': [self.f]}, []
        result = self.styler.format_style_args(self.df, columns={'A': self.f})
        self.assertEqual(result, expected)

    def test_format_row_col(self):
        expected = {0: [self.f]}, {'A': [self.g]}, []
        result = self.styler.format_style_args(self.df, rows={0: self.f},
                                               columns={'A': self.g})
        self.assertEqual(result, expected)

    def test_format_table(self):
        expected = {}, {}, [self.f]
        result = self.styler.format_style_args(self.df, table=self.f)
        self.assertEqual(result, expected)

        result = self.styler.format_style_args(self.df, table=[self.f])
        self.assertEqual(result, expected)

    def test_all(self):
        expected = ({0: [self.f]},
                    {'A': [self.f, self.g], 'B': [self.f, self.g]},
                    [self.f, self.g])

        result = self.styler.format_style_args(
            self.df, rows={0: self.f}, columns=[self.f, self.g],
            table=[self.f, self.g]
        )
        self.assertEqual(result, expected)

    def test_builder(self):
        styler = Styler(self.df, table=self.even_red)
        result = styler._build_styles(styler.rows, styler.columns, styler.table)
        expected = pd.DataFrame()
