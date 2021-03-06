import numpy as np
import pandas as pd

def handle_nan(sequence):
    df = pd.DataFrame(sequence)
    c = df.interpolate() #replace nan with interpolate values
    return c.values 