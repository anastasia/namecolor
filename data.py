import os
from collections import namedtuple

Name = namedtuple('Name', ['index', 'name', 'color'])

def read_data(dataset_path):
    indices = [] 
    names = [] 
    colors  = [] 

    with open(dataset_path, 'r') as f:
        for i,l in enumerate(f.readlines()):
            name, l , a ,b =l.strip().split(",")

            if len(name) > 0 :
                indices.append(i)
                names.append(name)
                colors.append(( l,a,b) ) 

    names_ = [ Name(index, name, color) 
            for index, name, color in zip(indices, names, colors) ]

    return names_
