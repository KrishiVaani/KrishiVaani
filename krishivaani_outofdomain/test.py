import pandas as pd
import os

root_dir = '/workspace/username/krishivaani_inferences/outofdomain_filter_inferences/inferences'
csv_files = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(dirpath, file))

df = pd.read_csv("/workspace/username/krishivaani_inferences/outofdomain_filter_inferences/OutofDomain_filtered.csv")
hypos = df["filename"].tolist()

dfx = pd.read_csv("/workspace/username/krishivaani_inferences/outofdomain_inferences/OutofDomain_new.csv")
fnames = dfx['filename'].tolist()

for path in csv_files:
    df2 = pd.read_csv(path)
    df2["Filename"] = fnames
    filtered_df = df2[df2['Filename'].isin(hypos)]
    if len(filtered_df) == 3063:
        pass
        filtered_df.to_csv(path, index=False)
    else:
        print("Error in filtering", path)
        print(len(filtered_df))
