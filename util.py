import numpy as np

def a2str(a):
    """
    formatting for 1 or 2 dimensional numpy arrays of booleans
    """
    if len(a.shape) == 1:
        return "".join(map(str, a))
    elif len(a.shape) == 2:
        return "\n".join(map(lambda row: "".join(map(str, row)), a))

def pprint(a):
    """
    formatting for 1 or 2 dimensional numpy arrays of booleans
    """
    print(a2str(a))


    
