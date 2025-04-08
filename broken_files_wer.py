import pandas as pd 
import os
import jiwer
import string
import re

def clean_sentence(input_string):
    if not isinstance(input_string, str):
        return ""    
    remove_chars = string.punctuation + '|'
    for char in remove_chars:
        input_string = input_string.replace(char, " ")
    input_string = input_string.replace("-", " ").replace("–", " ").replace("—", " ")
    result = input_string.replace("।", " ").strip()
    result = result.replace(':', ' ')
    result = result.strip()
    return re.sub(r'\s+', ' ', result)

directory = "/raid/username_/pdadiga/username/001/krishivaani_inferences/krishivaani_outofdomain/filter_inferences/byt5-small/"

main_df = pd.read_csv("/raid/username_/pdadiga/username/001/krishivaani_inferences/krishivaani_outofdomain/krishivaani_outofdomain.csv")
df = pd.read_csv("/raid/username_/pdadiga/username/001/krishivaani_inferences/krishivaani_outofdomain/broken_krishivaani_outofdomain.csv")

def process_file(file_path):
    comp_df = pd.read_csv(file_path)
    comp_df["Original_ID"] = df["Original_ID"]
    
    df2_rows = []
    for i, row in comp_df.iterrows():    
        uid = row["Original_ID"]
        uid_count = comp_df["Original_ID"].value_counts().get(uid, 0)
        if uid_count == 1:  
            row_dict = row.to_dict()
            df2_rows.append(row_dict)
        else:
            if i - 1 < 0 or comp_df["Original_ID"][i - 1] != uid:
                x = i
                pred = []
                hypo = []
                row_dict = row.to_dict()
                while x < len(comp_df) and comp_df["Original_ID"][x] == uid:
                    pred.append(comp_df["Predictions"][x]) 
                    hypo.append(comp_df["Ourmodel"][x])
                    x += 1
                row_dict["Predictions"] = " ".join(pred)
                row_dict["Hypothesis"] = " ".join(hypo)
                df2_rows.append(row_dict)
    
    df2 = pd.DataFrame(df2_rows)   
    assert len(main_df) == len(df2), "Mismatch error"
    df2["Ground Truth"] = main_df["ground_truth"]
    
    df2.fillna("", inplace=True)
    df2 = df2.map(clean_sentence)
    h1 = df2['Predictions'].tolist()
    g1 = df2['Ground Truth'].tolist()
    
    wer_value = jiwer.wer(g1, h1)
    print(f"{os.path.basename(file_path)}: {wer_value}")

for file_name in os.listdir(directory):
    if file_name.endswith(".csv") and "broken" in file_name:
        file_path = os.path.join(directory, file_name)
        try:
            process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
