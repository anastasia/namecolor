import time
from colourlovers import ColourLovers
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color


# check http://www.colourlovers.com/api
# https://github.com/elbaschid/python-colourlovers

def rgb2lab(r, g, b):
    newcolor = list(convert_color(
        sRGBColor(r / 255, g / 255, b / 255), LabColor).get_value_tuple())
    str_newcolor = [str(c) for c in newcolor]
    return str_newcolor


def get_data():
    cl = ColourLovers()
    data = [] 
    for i in range(1, 1000000,100):
        time.sleep(1)
        col = cl.colors(num_results=100, result_offset=i)
        data+=[ [c.title] + rgb2lab(c.rgb.red, c.rgb.blue,c.rgb.green) for c in col]
    
    print("Saving...")
    with open("/home/pablo/data/colors/corpus.data", "w") as f:
        for color in data:
            f.write(",".join(color)+"\n")

if __name__ == "__main__":
    get_data()
