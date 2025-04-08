import os
import jiwer
import pandas as pd
from jiwer import wer, cer
import string
import re

# combo = [['ज़','ज़'],['ऩ','ऩ'],['ऴ','ऴ'],['क़','क़'],['ख़','ख़'], ['ग़','ग़'], ['ढ़','ढ़'], ['य़','य़'],['फ़','फ़'],['ड़','ड़']] #आँ ऑ, ऎ एे, अा आ
# def normalize_combos(sen):
#     for chars in combo:
#         sen = sen.replace(chars[1], chars[0])
#     return sen

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



def process_file(file_path):
    df = pd.read_csv(file_path)
    df2 = pd.read_csv("/raid/username_/username_1/username/001/krishivaani_inferences/krishivaani_unknown/krishivaani_unknown.csv")    
    df['Predictions'] = df['Predictions'].apply(clean_sentence)
    df['Ground Truth'] = df2['ground_truth'].apply(clean_sentence)

    h1 = df['Predictions'].tolist()
    g1 = df['Ground Truth'].tolist()

    # if len(h1) == 6116:
    #     add_ground_truths(h1, g1)

    wer_value = wer(g1, h1)
    cer_value = cer(g1, h1)
    
    print(f"WER for {os.path.basename(file_path)}: {wer_value}")
    # print(f"CER for {os.path.basename(file_path)}: {cer_value}")
    # print(f"Length of {os.path.basename(file_path)}:", len(df))

folder_path = "/raid/username_/username_1/username/001/krishivaani_inferences/krishivaani_unknown/filtered_inferences/mt5-small"
for file_name in sorted(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, file_name)
    if "broken" not in file_name:
        if file_name.endswith(".csv"):    
                process_file(file_path)

