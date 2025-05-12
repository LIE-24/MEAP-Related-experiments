import os
from openai import OpenAI
import time
client = OpenAI(

   #api_key="sk-f", 
    #api_key="sk-c",
   #base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
   #base_url="https://api.deepseek.com",
   api_key="",
)

import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line: {line}, error: {e}")
    return data

file_path = './200b/xsum_results_200b_0.15_v1.jsonl'
result = read_jsonl(file_path)
count=0
score=0
# print("sss")
# print(len(result[8]['model output'].strip()))

for item in result:
    try:
        #print(item)
        print(file_path)
        print(count)
        #print(item['model output'])
        # if len(item['model output'].strip()) == 0:
        #     continue

        prompt="model output is the model's output, predicted_label is the manually annotated label, please compare model output and predicted_label, check if model output is strictly similar by comparing the two, if strictly similar return 1, otherwise return 0/n explanation needed"
       # prompt+="The criterion for judging hallucination is whether the content of model output completely exceeds the content of predicted_label. If it exceeds, it indicates hallucination, otherwise there is no hallucination. Remember: only complete exceeding counts as hallucination/n"
        prompt+="## model output: "+item['model output']+"/n"
        prompt+="## predicted_label: "+item['predicted_label']+"/n"
        prompt+="Be sure to follow the format below, be sure to follow it strictly!! \n Below is the format I provided, please follow it strictly!! \n"
        prompt+='\n score:xx \n'
        # completion = client.chat.completions.create(
        #     model="qwen-plus", # model list：https://help.aliyun.com/zh/model-studio/getting-started/models
        #     messages=[
        #         {'role': 'system', 'content': 'You are a helpful assistant.'},
        #         {'role': 'user', 'content': prompt}],
        #     )
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        print('--------------')
        print(completion.choices[0].message.content)
        score+=int(completion.choices[0].message.content.split('score:')[1].split('\n')[0])
        count+=1
        #time.sleep(1)
    except Exception as e:
        print(e)
        print(count)
        print(score)
        score+=0
        count+=1
        continue

print(score/count)  
    # print(completion.choices[0].message.content)
    # break
