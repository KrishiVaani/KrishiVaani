import openai
import pandas as pd
import random
from openai import OpenAI

lahaja_df = pd.read_csv("/raid/username_1/username_/username/files/shotting/krishivani/Krishivani diff. Models - Combined_Krishivani.csv")
indicv_df = pd.read_csv("/raid/username_1/username_/username/indicvoice/files/results/w2v_results.csv")

client = OpenAI(
    api_key="[API-KEY]" 
)

def chat_gpt(messages, model="gpt-4o-mini"): 
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content.strip()

responses = []

print("Number of utterances:",len(lahaja_df))
for i in range(len(lahaja_df)): 
    prompt1 = "Here is an incorrect Hindi ASR transcription. Please correct any spelling, grammatical, or meaning errors and provide the corrected version without explanation."
    prompt2 = "OUTPUT: "
    # random_hypothesis, random_corr, indicv_df = select_random_hypothesis(indicv_df)
    # if indicv_df.empty:
    #     break
    # one_shot_input = prompt1 + random_hypothesis
    # one_shot_output = prompt2 + random_corr
    # One_Shot_Inp.append(one_shot_input)
    # One_Shot_Response.append(one_shot_output)
    current_input = prompt1 + lahaja_df["Hypo IndicW2V"][i]
    messages = [
        #{"role": "system", "content": prompt1},
        # {"role": "user", "content": one_shot_input},
        # {"role": "assistant", "content": one_shot_output},
        {"role": "user", "content": current_input}
    ]

    try:
        corrected_output = chat_gpt(messages, model="gpt-4o-mini")
        responses.append(corrected_output)
        print(current_input)
        print(corrected_output)
    except Exception as e:
        print(f"Error processing input: {e}")
        responses.append(str(e))

#print(responses)
ndf = pd.DataFrame({
    # "One_Shot_Input": One_Shot_Inp,  
    # "One_Shot_Response": One_Shot_Response,
    "Response": responses})

ndf.to_csv("/raid/username_1/username_/username/gpt/noshot_gpt_all_error.csv", index=False)

ndf = pd.DataFrame({
    "Hypothesis": lahaja_df["Hypo IndicW2V"], 
    "Ground Truth": lahaja_df["Ground Truth"],
    "Response": responses,
})
ndf.to_csv("/raid/username_1/username_/username/files/shotting/krishivani/noshot_gpt_all_error_w2v.csv", index=False)
