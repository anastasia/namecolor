# namecolor

Transform a name or description into a color representation

This is the code used in [this article](https://pabloloyola.github.io/2018/04/01/color-language.html) where 
the paper [Character Sequence Models for Colorful Words](https://aclweb.org/anthology/D16-1202)
is replicated.

This code was tested on Pytorch 0.3.1

## Install
`pip install -r requirements.txt`

## In command line:

#### Data

downloading data:

`python -m color download`

#### Training model

`python -m color run`

#### Run on trained model

`python -m color trained`

## As Python module:

#### Data

downloading data:

```python
from color.download_data import get_data

get_data()
```

#### Training model

```python
from color.run import train_model

train_model()
```

#### Run on trained model

```python
from color.trained import get_color

get_color("Your color name here")
```


#### Acknowledgement

Most of the content on this repository is based on the great Pytorch tutorials by [epochx](https://github.com/epochx/pytorch-nlp-tutorial) 
