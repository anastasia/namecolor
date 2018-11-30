import os
import time
from colourlovers import ColourLovers
from collections import namedtuple

from color import settings
from color.utils import rgb2lab


# check http://www.colourlovers.com/api
# https://github.com/elbaschid/python-colourlovers

def get_data():
    cl = ColourLovers()
    corpus_data = os.path.join(settings.DATA_DIR, "corpus.data")
    with open(corpus_data, "a+") as f:
        for i in range(1, 1000000, 100):
            time.sleep(0.5)
            col = cl.colors(num_results=100, result_offset=i)
            for c in col:
                try:
                    color = [c.title] + rgb2lab(c.rgb.red, c.rgb.green, c.rgb.blue)
                    f.write(",".join(color) + "\n")
                except Exception:
                    try:
                        title = c.title.encode('ascii', 'ignore').decode('ascii')
                        color = [title] + rgb2lab(c.rgb.red, c.rgb.green, c.rgb.blue)
                        f.write(",".join(color) + "\n")
                    except Exception as err:
                        errored_color = c
                        print("got error after decoding attempt:", errored_color, err)
                        pass


def read_data(dataset_path):
    indices = []
    names = []
    colors = []
    Name = namedtuple('Name', ['index', 'name', 'color'])

    with open(dataset_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            values = line.strip().split(",")
            # since the names in the colourlovers db sometimes have commas
            # we pop the LAB values, and anything that's left over is the name
            b_val = values.pop()
            a_val = values.pop()
            l_val = values.pop()
            name = ",".join(values)
            if len(name) > 0:
                indices.append(idx)
                names.append(name)
                colors.append((float(l_val), float(a_val), float(b_val)))

    names_ = [Name(index, name, color)
              for index, name, color in zip(indices, names, colors)]

    return names_
