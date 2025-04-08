import pandas as pd
df = pd.read_csv("/workspace/username/krishivaani_inferences/krishivaani_unknown/broken_krishivaani_unknown.csv")
df.fillna("", inplace=True)
df2_rows = []

for i, row in df.iterrows():    
    uid = row["Original_ID"]
    uid_count = df["Original_ID"].value_counts().get(uid, 0)
    # if i+1>=len(df) or df["Original_ID"][i+1] != uid:
    if uid_count == 1:  
        row_dict = row.to_dict()
        df2_rows.append(row_dict)
    else:
        if i-1<0 or df["Original_ID"][i-1] != uid:
            x = i
            hypo = []
            row_dict = row.to_dict()
            while df["Original_ID"][x] == uid:
                hypo.append(df["Ourmodel"][x]) 
                x = x + 1
            hypo = " ".join(hypo)
            row_dict["Ourmodel"] = hypo        
            df2_rows.append(row_dict)
            cou = cou+1
            print(cou)
df2 = pd.DataFrame(df2_rows)       

duplicate_counts = df2["Original_ID"].value_counts()
duplicated_uids = duplicate_counts[duplicate_counts > 1]
print("Duplicated UIDs:", duplicated_uids)
print("Number of duplicated UIDs:", len(duplicated_uids))

df2.to_csv("/workspace/username/krishivaani_inferences/krishivaani_unknown/joint_krishivaani_unknown.csv", index=False)
print(df["Original_ID"].nunique(), df2["Original_ID"].nunique()) 
print(len(df), len(df2))