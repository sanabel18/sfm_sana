import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utm

def plot_routes(geo_list):
    fig = plt.figure()
    ax = Axes3D(fig)
    color = ['r', 'darkturquoise','gold','teal','deeppink','indigo','deepskyblue','steelblue','olive','palegreen','firebrick','chocolate','peru','lightcoral','tan','limegreen','aqua','hotpink','blue','violet','r', 'r', 'darkturquoise','gold','teal','deeppink','indigo','deepskyblue','steelblue','olive','palegreen'    ,'firebrick','chocolate','peru','lightcoral','tan','limegreen','aqua','hotpink','blue','violet']
    icolor = 0
    for i, geo in enumerate(geo_list):
        with open(geo) as f:
            geodata = json.load(f)
        data = geodata['geometry']['coordinates']
        x = []
        y = []
        z = []
        x_node = []
        y_node = []
        z_node = []
        for idx_data, data_pt in enumerate(data):
            elevation = data_pt[2]
            utm_data = utm.from_latlon(data_pt[1],data_pt[0])
            utm_xy = np.array(utm_data[:2])
            x.append(utm_xy[0])
            y.append(utm_xy[1])
            z.append(elevation)
        print(len(color))
        print("i, x, y , z {} {} {} {}".format(i, len(x), len(y), len(z)))
        if i > len(color)-1 :
            icolor = 0
        print("i, icolor {} {}".format(i, icolor))
        ax.plot3D(x, y, z, color[icolor])
        icolor += 1
    plt.show()

def main():
   import glob
   plot_file_list = glob.glob("*.geojson")
   plot_routes(plot_file_list)
if __name__ == "__main__":
    main()
