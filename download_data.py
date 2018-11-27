import time
from colourlovers import ColourLovers

from utils import rgb2lab


# check http://www.colourlovers.com/api
# https://github.com/elbaschid/python-colourlovers

def get_data():
    cl = ColourLovers()
    with open("corpus_new_fixed.data", "a+") as f:
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


if __name__ == "__main__":
    get_data()
