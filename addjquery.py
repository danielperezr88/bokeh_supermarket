from bokeh.models import ColumnDataSource, LayoutDOM
from bokeh.core.properties import Any, Dict, Instance, String

class AddJQuery(LayoutDOM):

    __javascript__ = ["https://ajax.googleapis.com/ajax/libs/jquery/2.1.4/jquery.min.js"]
    __implementation__ = ""

    color = String
    source = Instance(ColumnDataSource)
    x = String
    y = String