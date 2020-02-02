import numpy as np
import pandas as pd
import os
import glob 
import ipywidgets as widgets
from IPython.display import display

def create_interface(train_labels,label):
    ids_looking = train_labels[train_labels['label_image'] == str(label)]['id_image'].to_list()
    current_id = 0
    file = open("data/images/" + ids_looking[current_id], 'rb')
    image = file.read()

    image_widget = widgets.Image(value=image,
                                 format='png',
                                 width=300,
                                 height=300)
    file.close()

    widget_play = widgets.Play(value=0, min=0, max=len(ids_looking)-1, interval=150, step=1, disabled=False)
    widget_next = widgets.Button(description='Next', disabled=False)
    widget_prev = widgets.Button(description='Prev', disabled=True)
    widget_slider = widgets.IntSlider(value=current_id,min=0,max=len(ids_looking)-1,step=1,orientation='horizontal')
    image_number = widgets.Button(description='Image ID: ' + str(ids_looking[current_id]) , disabled=True)
    image_class = widgets.Button(description='Label: 1', disabled=True)

    image_info = widgets.HBox([image_number, image_class])
    buttons = widgets.HBox([widget_prev, widget_next])

    def load_image(image_id):
        file = open("data/images/" + image_id, 'rb')
        image_widget.value = file.read()
        file.close()
        image_number.description = 'Image ID: '+str(image_id)
        image_class.description = 'Label: ' + train_labels[train_labels["id_image"] == image_id]['label_image'].values[0]

    def load_next(args):
        global current_id
        if current_id >= len(ids_looking) - 1:
            widget_next.disabled = True
            widget_prev.disabled = False
            return
        current_id += 1
        if current_id >= len(ids_looking) - 2:
            widget_next.disabled = True
        if current_id >= 1:
            widget_prev.disabled = False
        widget_play.value = current_id
        next_image = ids_looking[current_id]
        load_image(next_image)

    def load_prev(args):
        global current_id
        current_id -= 1
        if current_id <= len(ids_looking) - 1:
            widget_next.disabled = False
        if current_id <= 1:
            widget_prev.disabled = True
        widget_play.value = current_id
        prev_image = ids_looking[current_id]
        load_image(prev_image)

    def animate(args):
        global current_id
        current_id = widget_play.value
        load_image(ids_looking[current_id])
        widget_prev.disabled = False

    widget_play.observe(animate, 'value')
    widget_next.on_click(load_next)
    widget_prev.on_click(load_prev)
    widgets.jslink((widget_play, 'value'), (widget_slider, 'value'))

    interface = widgets.VBox([image_widget, image_info, buttons, widget_slider, widget_play])
    return interface