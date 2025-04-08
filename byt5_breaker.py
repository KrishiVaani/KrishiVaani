import pandas as pd

def split_hypothesis(hypothesis: str, window_size: int = 28, overlap: int = 0):
    words = hypothesis.split()
    segments = []
    start = 0
    
    while start < len(words):
        end = start + window_size
        segment = words[start:end]
        segments.append(" ".join(segment))
        if end >= len(words):
            break
        start = end - overlap      
    return segments

df = pd.read_csv("/raid/username_/pdadiga/username/001/krishivaani_inferences/outofdomain_filter_inferences/krishivaani_outofdomain_filtered.csv")
df.fillna("", inplace=True)
df2_rows = []
cou=0
for i, row in df.iterrows():    
    hypothesis = row["Ourmodel"]
    segments = split_hypothesis(hypothesis)

    if len(segments) < 2:
        row_dict = row.to_dict()
        row_dict["Original_ID"] = i
        df2_rows.append(row_dict)
    else:
        for seg_idx, segment in enumerate(segments):
            new_row = row.to_dict()  
            new_row["Ourmodel"] = segment  
            new_row["Original_ID"] = i   
            df2_rows.append(new_row)
        cou = cou+1
df2 = pd.DataFrame(df2_rows)
df2 = df2.drop(["IC","wav2vec2"], axis=1)
df2.to_csv("/raid/username_/pdadiga/username/001/krishivaani_inferences/outofdomain_filter_inferences/broken_krishivaani_outofdomain_filtered.csv", index=False)
print("Splitted rows",cou)
print(len(df), len(df2))

