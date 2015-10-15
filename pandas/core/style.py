import uuid

import pandas.core.common as com

class Styler(object):
    """
    Helps style a DataFrame or Series according to the
    data.

    Parameters
    ==========

    data : Series or DataFrame
    rows : funcion, list of functions, or dictionary
        see Notes
    columns : function, list of functions or dictonary
        see Notes
    table : function or list of functions
        see Notes

    Attributes
    ==========

    Notes
    =====

    The rows and columns arguments take...
    """
    def __init__(self, data, rows=None, columns=None, table=None):
        self.data = data
        self.u = str(uuid.uuid1()).replace("-", "_")
        rows, columns, table = self.format_style_args(data, rows, columns,
                                                      table)
        self.rows = rows
        self.columns = columns
        self.table = table

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

    @staticmethod
    def format_style_args(data, rows=None, columns=None, table=None):
        """
        rows/columns:
            - function
            - list
            - dict
        dict : {row/col}: [function]
        """
        def cond(v):
            return (callable(v) or com.is_list_like(v)
                    and not isinstance(v, dict))

        if cond(rows):
            rows = {r: rows for r in data.index}
        if cond(columns):
            columns = {c: columns for c in data.columns}
        if callable(table) and not com.is_list_like(table):
            table = [table]

        rows = rows or {}
        columns = columns or {}
        table = table or []

        rows = {row: [x] if not com.is_list_like(x) else x
                for row, x in rows.items()}
        columns = {col: [x] if not com.is_list_like(x) else x
                for col, x in columns.items()}
        table = [table] if not com.is_list_like(table) else table

        return rows, columns, table

    def _build_styles(self, rows=None, columns=None, table=None):
        """
        Build a DataFrame indexed like ``self.data`` containing the CSS
        property: value pairs.

        rows/cols : {idx: [functions]}
        table : [functions]
        """
        from pandas.tools.merge import concat
        data = self.data
        final = []  # (row, col, prop, value)

        # rows, columns, table = self._format_style_args(rows, columns, table)
        # TODO: ensure index handling is correct
        for row in rows:
            for func in rows[row]:
                props, values  = zip(*func(data.loc[row]))
                final.append(data.__class__({'row': row, 'col': data.columns,
                                             'prop': props, 'value': values}))
        for col in columns:
            for func in columns[col]:
                props, values  = zip(*func(data[col]))
                final.append(data.__class__({'row': data.index, 'col': col,
                                             'prop': props, 'value': values}))

        for func in table:
            r = func(data)
            for col in r:
                try:
                    prop, values = zip(*r[col])
                except ValueError:
                    props, values = '', ''
                    final.append(data.__class__({'row': r.index, 'col': col,
                                                 'prop': props, 'value': values}))

        r = concat(final)[['row', 'col', 'prop', 'value']]
        # I don't feel great about this.
        # Could use a MultiIndex of [(row/col), (prop, value)]
        # Or a dumber structure like a dict.
        r = (r['prop'] + ':' + r['value']).groupby([r['row'], r['col']]).agg(
             lambda x: '; '.join(x))

        return r.unstack()

    def translate(self):
        """
        Convert the DataFrame in `self.data` and the attrs from `_build_styles`
        into a dictionary of {head, body, uuid, cellstyle}
        """
        attrs = self._build_styles(self.rows, self.columns, self.table)
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
                                   for x in attrs.loc[idx, col].split('; ')],
                         'selector': "row%s_col%s" % (r, c)})
                except KeyError:
                    pass
            body.append(row_es)

        # uuid required to isolate table styling from others
        # in same notebook in ipnb
        u = str(uuid.uuid1()).replace("-","_")
        return dict(head=head, cellstyle=style, body=body, uuid=u)

    def render(self):
        return(self.t.render(**self.translate()))

    def apply(func, axis=1, subset=None):
        """
        func : function or list of function
        """
        pass

    def applymap(func):
        pass

