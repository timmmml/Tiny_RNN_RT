"""
Ji-an's codes to play with TinyRNN models.
"""
import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div, TapTool
from bokeh.plotting import figure, show, output_file

# Example data
x = np.random.random(10)
y = np.random.random(10)
info = [f'Point ({xi:.2f}, {yi:.2f})' for xi, yi in zip(x, y)]

source = ColumnDataSource(data=dict(x=x, y=y, info=info))

hover = HoverTool(tooltips=[('Info', '@info')])

p = figure(width=600, height=400, tools=[hover], title='Interactive Scatter Plot')
renderer = p.scatter('x', 'y', size=10, source=source)

# Create a Div element to display the selected point information
info_div = Div(width=600, height=100)

# Define a JavaScript callback for when a point is clicked
callback = CustomJS(args=dict(source=source, div=info_div), code="""
    const indices = source.selected.indices;
    if (indices.length > 0) {
        const info = source.data['info'][indices[0]];
        div.text += '<p>' + info + '</p>';
        source.selected.indices = [];  // Clear selection
    }
""")

# Attach the callback to the scatter plot
renderer.js_on_event('tap', callback)

tap = TapTool(renderers=[renderer])
p.add_tools(tap)

# Create a layout with the plot and the Div element
layout = column(p, info_div)

output_file('scatter_with_click.html')
show(layout)