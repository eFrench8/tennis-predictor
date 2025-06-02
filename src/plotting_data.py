from data_loader import load_data
import pandas as pd
import matplotlib.pyplot as plt

df = load_data()

df.plot(kind = 'scatter', x = 'Pts_1', y = 'Pts_2')

plt.show()


