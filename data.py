from collections import namedtuple

Name = namedtuple('Name', ['index', 'name', 'color'])


def read_data(dataset_path):
    indices = []
    names = []
    colors = []

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
