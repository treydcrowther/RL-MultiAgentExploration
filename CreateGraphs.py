import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists

result = []
searched_areas = []
all_files = pd.DataFrame()
all_arrays = []
# Gather info from the csv files and create graphs about how long it took to explore the area
for i in range(100):
    file_path = f"percent_explored{i/2}.csv"
    if exists(file_path):
        # Prepend the string "explored" to beginning of each file
        # with open(file_path, "r") as file:
        # # Step 2: Read the existing contents
        #     file_contents = file.read()

        # # Step 3: Open the same file for writing (this will erase its contents)
        # with open(file_path, "w") as file:
        #     # Step 4: Write "explored" as the first line
        #     file.write("explored\n")

        #     # Step 5: Write the previously read contents back into the file
        #     file.write(file_contents)

        result.append(pd.read_csv("" + file_path))
        # add i values that match the line count of the csv file
        result[len(result) - 1].insert(0, 'step', [i for i in range(len(result[len(result) - 1]))])
    else:
        print("file doesn't exist:" + file_path)
    if(result == []):
        continue
    
added = 0
for i in range(len(result)):
    total_searched_area = result[i]["explored"].tolist()
    if(len(total_searched_area) > 300):
        continue
    added += len(total_searched_area)
    print(len(total_searched_area))
    actual = [total_searched_area[i] * 100 for i in range(len(total_searched_area))]
    plt.plot([i for i in range(len(actual))], actual)

plt.xlabel("Steps")
plt.ylabel("Percent Explored")

plt.savefig("explored_all.png")

print(added / len(result))