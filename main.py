from data import add_new_columns, load_data, data_analysis
from clustering import kmeans,visualize_results,transform_data
import numpy as np

def main():
    #activating the part A of the assignment
    kVals = [2,3,5]
    path = "london.csv"
    df = load_data(path)
    df = add_new_columns(df)
    data_analysis(df)

    #reloading the data again
    features = ["cnt", "t1"]
    df = load_data(path)
    arr = transform_data(df, features)

    #activating the part B of the assignment
    print("Part B: ")
    for k in kVals:
        labels, centroids = kmeans(arr , k)
        save_path = f"visual{k}.jpg"
        visualize_results(arr, labels, centroids, save_path)
        print(f"k = {k}")
        print(np.array_str(centroids, precision=3, suppress_small=True))
        print()

if __name__ == "__main__":
    main()