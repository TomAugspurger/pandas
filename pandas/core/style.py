"""
Module for applying conditional formatting to
DataFrames and Series.
"""
from uuid import uuid1
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas  as pd
# import pandas.core.common as com

class Styler(object):
    """
    Helps style a DataFrame or Series according to the
    data.

    Parameters
    ==========

    data : Series or DataFrame

    Attributes
    ==========

    ctx : defaultdict
        maps (row, column) -> [styles] where each style is a
        TODO: <(attribute, value)> paire | <string with 'attribute: value'>

    """
    def __init__(self, data, uuid=None, other=''):
        self.data = data
        self.uuid = uuid
        # This will behave strangely for non-unique indicies
        self.ctx = defaultdict(list)
        self.other = other
        try:
            from jinja2 import Template
        except ImportError:
            msg = "Jinja2 is required to use DataFrame.style"
            raise ImportError(msg)

        self.t = Template("""
                <style  type="text/css" >
                #T_{{uuid}} tr {
                    border: none;
                }
                #T_{{uuid}} {
                        border: none;
                }
                #T_{{uuid}} th.blank {
                    border: none;
                }

                {% for s in style %}
                    #T_{{uuid}} {{s.selector}} {
                    {% for p,val in s.props %}
                        {{p}}: {{val}};
                    {% endfor %}
                    }
                {% endfor %}
                {% for s in cellstyle %}
                    #T_{{uuid}}{{s.selector}} {
                    {% for p,val in s.props %}
                        {{p}}: {{val}};
                    {% endfor %}
                    }
                {% endfor %}
                {% for s in other %}
                    {{s}}
                {% endfor %}
                </style>

                <table id="T_{{uuid}}">
                {% if caption %}
                    <caption>{{caption}}</caption>
                {% endif %}

                <thead>
                    {% for r in head %}
                    <tr>
                        {% for c in r %}
                        <{{c.type}} class="{{c.class}}">{{c.value}}</th>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </thead>
                <tbody>
                    {% for r in body %}
                    <tr>
                        {% for c in r %}
                        <{{c.type}} id="T_{{uuid}}{{c.id}}" class="{{c.class}}">{{c.value}}</th>
                        {% endfor %}
                    </tr>
                    {% endfor %}
                </tbody>
                </table>
                """)

    def translate(self):
        """
        Convert the DataFrame in `self.data` and the attrs from `_build_styles`
        into a dictionary of {head, body, uuid, cellstyle}
        """
        ctx = self.ctx
        ROW_HEADING_CLASS = "row_heading"
        COL_HEADING_CLASS = "col_heading"
        DATA_CLASS = "data"
        BLANK_CLASS = "blank"
        BLANK_VALUE = ""

        cell_context = dict()

        n_rlvls = self.data.index.nlevels
        n_clvls = self.data.columns.nlevels
        rlabels = self.data.index.tolist()
        clabels = self.data.columns.tolist()

        if n_rlvls == 1:
            rlabels = [[x] for x in rlabels]
        if n_clvls == 1:
            clabels = [[x] for x in clabels]
        clabels=list(zip(*clabels))

        style = []
        head = []

        for r in range(n_clvls):
            row_es = [{"type": "th", "value": BLANK_VALUE,
                       "class": " ".join([BLANK_CLASS])}] * n_rlvls
            for c in range(len(clabels[0])):
                cs = [COL_HEADING_CLASS, "level%s" % r, "col%s" % c]
                cs.extend(cell_context.get("col_headings", {}).get(r, {}).get(c, []))
                row_es.append({"type": "th", "value": clabels[r][c],
                               "class": " ".join(cs)})
            head.append(row_es)

        body = []
        for r, idx in enumerate(self.data.index):
            cs = [ROW_HEADING_CLASS, "level%s" % c, "row%s" % r]
            cs.extend(cell_context.get("row_headings", {}).get(r, {}).get(c, []))
            row_es = [{"type": "th", "value": rlabels[r][c], "class": " ".join(cs)}
                      for c in range(len(rlabels[r]))]

            for c, col in enumerate(self.data.columns):
                cs = [DATA_CLASS,"row%s" % r, "col%s" % c]
                cs.extend(cell_context.get("data", {}).get(r, {}).get(c, []))
                row_es.append({"type": "td", "value": self.data.iloc[r][c],
                               "class": " ".join(cs), "id": "_".join(cs[1:])})
                try:
                    style.append(
                        {'props': [x.split(":")
                                   for x in ctx[idx, col]],
                         'selector': "row%s_col%s" % (r, c)})
                except KeyError:
                    pass
            body.append(row_es)

        # uuid required to isolate table styling from others
        # in same notebook in ipnb
        uuid = self.uuid or str(uuid1()).replace("-", "_")
        return dict(head=head, cellstyle=style, body=body, uuid=uuid,
                    other=self.other)

    def render(self):
        return(self.t.render(**self.translate()))

    def _repr_html_(self):
        """
        return our rendered self
        """
        return self.render()

    def update_ctx(self, attrs):
        """
        update the state of the Styler. Collects a mapping
        of {index_label: '<property>: <value>'}

        attrs: Series or DataFrame
            should contain strings of '<property>: <value>'
        """
        # TODO: This breaks terribly with duplicates...
        # Split attrs like "attr: val; attr: val"

        if attrs.ndim == 1:
            for row, v in attrs.iteritems():
                for pair in v.rstrip(";").split(";"):
                    self.ctx[(row, attrs.name)].append(pair)
        else:
            # MultiIndex is (row, col)
            for k, v in attrs.stack().iteritems():
                for pair in v.rstrip(";").split(";"):
                    self.ctx[k].append(pair)

    def __copy__(self):
        styler = Styler(self.data)
        styler.ctx = deepcopy(self.ctx)
        return styler

    def apply(self, func, axis=0, subset=None, **kwargs):
        """
        func : function or list of function
        """
        subset = slice(None) if subset is None else subset
        if axis == 0:
            row_idx, col_idx = slice(None), subset
        else:
            row_idx, col_idx = subset, slice(None)
        result = self.data.loc[row_idx, col_idx].apply(func, axis=axis, **kwargs)
        self.update_ctx(result)
        return self

    def applymap(self, func, **kwargs):
        raise ValueError

    # Builtin stylers
    def highlight_null(self, null_color='red', notnull_color=None):
        if notnull_color is None:
            notnull_color = ''
        c = (self.data.where(self.data.isnull(),
                             'background-color: %s' % notnull_color)
             .fillna('background-color: %s' % null_color))
        self.update_ctx(c)
        return self

    def color_bg_range(self, cmap='PuBu'):
        self.apply(self._color_bg_range, cmap=cmap)
        return self

    @staticmethod
    def _color_bg_range(s, cmap='PuBu', low=.2, high=.8):
        """Color background in a range according to the data."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex
        normed = (s - s.min()) / (s.max() - s.min())
        colors = [rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return s.__class__(['background-color: %s' % color for color in colors], index=s.index)

    def color_font_even(s, even_color='black', odd_color='red'):
        """"""
        colors = [even_color if x % 2 else odd_color for x in s]
        return pd.Series(['font-color: %s' % color for color in colors], index=s.index)

    def color_gradient(s, cmap='YlGn'):
        "Color background by position"
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex
        colors = [rgb2hex(x) for x in plt.cm.get_cmap(cmap)(1 / np.arange(len(s)))]
        return pd.Series(['background-color: %s' % color for color in colors], index=s.index)

    def striped(s):
        colors = ["B5DEFF" if i % 2 else "" for i in range(len(s))]
        return pd.Series(['background-color: %s' % color for color in colors], index=s.index)

    def align(self, side='right'):
        """ align to the side """
        self.apply(self._align, side=side)
        return self

    @staticmethod
    def _align(s, side='right'):
        """align to the side """
        return s.__class__(['text-align: %s' % side], index=s.index)
