import seaborn as sns
import matplotlib.pyplot as plt
import csv
import numpy as np

def main():
    
    data = np.zeros((11, 11))
    # load the csv file from the current directory
    with open("con_data_original_1.csv", "r") as f:
        reader = csv.reader(f)
        # discard the first row
        next(reader)
        while True:
            try:
                row = next(reader)
                # convert the data to floats
                row = [float(i) for i in row]
                # add the data to the numpy array
                data[int(row[1]*10), int(row[0]*10)] = abs(row[2])
            except StopIteration:
                break
    
    custom_cmap = "viridis"  # You can choose another colormap
    vmin = -1 # Minimum value for the color scale
    vmax = 0.5  # Maximum value for the color scale
    # plot the heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap=None)
    
    # add labels to the plot
    plt.xlabel("Learning Rate")
    plt.ylabel("Discount Factor")
    plt.title("Convergence Speed")
    
    plt.show()
    
if __name__ == "__main__":
    main()
    