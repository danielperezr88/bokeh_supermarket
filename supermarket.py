from bokeh.plotting import curdoc, figure
from bokeh.layouts import gridplot
from bokeh.models.widgets import Slider
from bokeh.models.callbacks import CustomJS
from bokeh.models import ColumnDataSource, WidgetBox, Div, Column

import requests as req

from string import Template
from addjquery import AddJQuery

def generate_url(host, protocol='http', port=80, dir=''):

    if isinstance(dir, list):
        dir = '/'.join(dir)

    return "%s://%s:%d/%s" % (protocol, host, port, dir)

#MYIP = '127.0.0.1'
MYIP = req.get(generate_url('jsonip.com')).json()['ip']
server_port = 50001

DIV_TEMPLATE = Template("\n".join([
    "<div class=\"outer\" style=\"display:table; position:absolute; height:100%; width:100%;\">",
    "   <div class=\"middle\" style=\"display:table-cell; vertical-align:middle;\">",
    "       <div class=\"inner\" id=\"results-container\" style=\"width:100%;text-align:center;margin-right:auto;margin-left:auto;\">",
    "           ${content}",
    "       </div>",
    "   </div>",
    "</div>"
]))

JS_CODE_TEMPLATE = Template("""

    function update_plot(data){
        ss.data = data['results'];
        ss.trigger('change');
    }

    function update_text(data){
        $('#results-container').html(data['results']);
    }

    clearTimeout(timer)
    timer = window.setTimeout(function(){

        var inputs = $('div .bk-slider-parent').find('input');
        inputs = $.map(inputs, function(a){return $(a).val()});

        $.ajax({
            url: '${url_compute}' + '/[' + inputs + ']',
            type: 'POST',
            contentType: 'application/json',
            success: update_plot
        });

        $.ajax({
            url: '${url_predict}' + '/[' + inputs + ']',
            type: 'POST',
            contentType: 'application/json',
            success: update_text
        });

    }, 2000);

""")

default_data = req.post(url=generate_url(MYIP, port=server_port, dir='defaults'), timeout=20).json()
static_source = ColumnDataSource(data=default_data['results'])

callback = CustomJS(args=dict(ss=static_source), code=JS_CODE_TEMPLATE.safe_substitute(
    url_compute=generate_url(MYIP, port=server_port, dir='compute'),
    url_predict=generate_url(MYIP, port=server_port, dir='predict')
))

def redraw():
    static_source.data = default_data['results']


field_data = req.post(url=generate_url(MYIP, port=server_port, dir='fields'), timeout=20)
print(field_data)
field_data = field_data.json()
sliders = WidgetBox(
    children=list(Slider(**dict(zip(list(f.keys())+['callback'],list(f.values())+[callback]))) for f in field_data['results']),
    width=30
)

#scatter = Scatter3d(x='x', y='y', z='z', color='color', data_source=static_source)
plot = figure(title='PCA Plot', plot_height=300, plot_width=400, responsive=True, tools="pan,reset,save,wheel_zoom")
plot.scatter(x='x', y='y', color='color', source=static_source)

def_cont = req.post(url=generate_url(MYIP, port=server_port, dir=['predict', 'default']), timeout=20).json()

text = Column(children=[
    plot,
    Div(text=DIV_TEMPLATE.substitute(content=def_cont['results']), sizing_mode='scale_both')
], width=65)

curdoc().add_root(gridplot([[text, sliders]], responsive=True))
#curdoc().add_root(gridplot([[script, plot, sliders]], responsive=True))

curdoc().add_next_tick_callback(redraw)
