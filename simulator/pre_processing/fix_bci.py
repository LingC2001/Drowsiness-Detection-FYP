import pandas as pd
import csv


bci_filename = "a" + ".csv"

df = pd.read_csv(bci_filename, header=None)

with open('bci.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Sample Index", "EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3", "EXG Channel 4", "EXG Channel 5", "EXG Channel 6", 
                        "EXG Channel 7", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2", "Other", "Other", "Other", "Other",
                        "Other", "Other", "Other", "Analog Channel 0", "Analog Channel 1", "Analog Channel 2", "Timestamp", "Other"])
    for i in range(df.shape[0]):
        row_data = df.iloc[i].to_list()[0].split("\t")
        writer.writerow(row_data)

print("BCI file converted")