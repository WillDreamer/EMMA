from glob import glob
import os
import torch
import json
import numpy as np
import re
import time
import random
import transformers
import argparse
import logging
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import yaml
import base64
from io import BytesIO
from openai import OpenAI

## 首先通过sglang进行模型推理部署，并指定port
model_to_port = {
    'Qwen/Qwen2.5-VL-7B-Instruct': 30000,
    'Qwen/Qwen2-VL-7B-Instruct': 30001,
    'meta-llama/Llama-3.2-11B-Vision-Instruct': 30002
}

model_to_key = {
    'Qwen/Qwen2.5-VL-7B-Instruct': "EMPTY",
    'Qwen/Qwen2-VL-7B-Instruct': "EMPTY",
    'meta-llama/Llama-3.2-11B-Vision-Instruct': "EMPTY"
}

llm_summary_port = 8080


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)
            return None

def create_message(sample):
    query = sample['query']
    all_contents = []
    matches = re.findall(r"<(image_\d+)>", query)
    split_text = re.split(r"<image_\d+>", query)
    for i, fragment in enumerate(split_text):
        if fragment.strip():
            all_contents.extend([
                {"type": "text", "text": fragment}
            ])
        if i < len(matches):
            if sample[matches[i]]:
                img_base64 = encode_image_to_base64(sample[matches[i]])
                all_contents.extend([
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ])
            else:
                logging.error(
                    f"The image token {matches[i]} is in the query, but there is no corresponding image provided by the data")

    messages = [
        {
            "role": "user",
            "content": all_contents
        }
    ]
    return messages

def generate_answer(messages, port = 8086, key = "EMPTY", retry_attempts = 5, temperature=0.01,max_tokens=4096):
    
    attempt = 0

    openai_api_key = key
    openai_api_base = f"http://localhost:{port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    while attempt < retry_attempts:
        try:
            models = client.models.list()
            model = models.data[0].id
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,)

            return response
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")

            if 'error' in str(e) and 'message' in str(e):
                error_message = str(e)
                if 'The server had an error processing your request.' in error_message:
                    sleep_time = 30
                    logging.error(f"Server error, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                elif 'Please try again in ' in error_message:
                    sleep_time = float(error_message.split('Please try again in ')[1].split('s.')[0])
                    logging.error(f"Rate limit exceeded, retrying in {sleep_time * 2}s...")
                    time.sleep(sleep_time * 2)
                elif 'RESOURCE_EXHAUSTED' in error_message:
                    sleep_time = 30
                    logging.error(f"Gemini rate limit, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print("Unknown error, skipping this request.")
                    break
            attempt += 1

    return completion

def build_query(sample, config, strategy):
    """Build the text query by combining the context, question and options. The <image_n> token is still there"""
    context = sample['context']
    question = sample['question']
    example = ""
    res_dict = {}
    if sample['type'].lower() == 'multiple choice':
        options = sample['options']
        start_chr = 'A'
        for option in options:
            example += f"{start_chr}: {option}\n"
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_format']
        empty_prompt = empty_prompt_sample_structure.format(context=context, question=question, options=example)
        if strategy == 'CoT':
            res_dict['query'] = empty_prompt + config['Strategy_Instruction']['CoT']
        else:
            res_dict['query'] = empty_prompt + config['Strategy_Instruction']['Directly']

        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:
        empty_prompt_sample_structure = config['open_ended_format']
        empty_prompt = empty_prompt_sample_structure.format(context=context, question=question)
        if strategy == 'CoT':
            res_dict['query'] = empty_prompt + config['Strategy_Instruction']['CoT']
        else:
            res_dict['query'] = empty_prompt + config['Strategy_Instruction']['Directly']
        res_dict['gt_content'] = sample['answer']

    # append existing key and value in data
    res_dict.update(sample)
    return res_dict

def construct_assistant_message(completion):
    content = completion.choices[0].message.content.strip()
    
    return {"role": "assistant", "content": content}

def summarize_message(agent_contexts, port=8080, key = "EMPTY"):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        # [-1]是上一轮，["content"]取出模型给出的答案
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent and explain the reasoning in each solution."
    agent_context = [{"role": "user", "content": prefix_string}]

    ## 这里是取一个llm来作为summarization
    completion = generate_answer(agent_context, port = port, key = key)
    content = completion.choices[0].message.content

    return content

def construct_message(agents, prefix, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct? Please reiterate your answer, with your final answer a single answer of the form \\boxed{{answer}} at the end of your response.".format(prefix)}

    prefix_string = "Here is are solution from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: {}".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Using each response as additional advice, can you give an updated bullet by bullet answer to {}? Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
    return {"role": "user", "content": prefix_string}

def construct_message_summary(summary, prefix, idx):
    prefix_string = "Here is a summary of solutions from several other agents: {}".format(summary)

    prefix_string = prefix_string + "\n\n Examine each these solutions as additional advice, can solve {} and give your updated answer? Explain your reasoning. \n Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
    return {"role": "user", "content": prefix_string}


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def parse_answer(input_str):
	return remove_boxed(last_boxed_only_string(input_str))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type = int, default = 3)
    parser.add_argument("--rounds", type = int, default = 3)
    parser.add_argument("--save_str", type = str, default = "/chenyang_data/mmr/EMMA/multiagent-ft/multiagent-ft/generate_dir/")
    parser.add_argument("--summarize", default=True, dest = "summarize")
    parser.add_argument("--device", type = int, dest = "device", default = 0)
    parser.add_argument("--temperature", default = 1, type = float, dest = "temperature")
    parser.add_argument("--top_p", default = 0.9, type = float, dest = "top_p")
    parser.add_argument('--dataset_name', type=str, default='luckychao/EMMA')
    parser.add_argument('--models',type=list,
                        default=['Qwen/Qwen2.5-VL-7B-Instruct','Qwen/Qwen2-VL-7B-Instruct','meta-llama/Llama-3.2-11B-Vision-Instruct'])
    parser.add_argument('--subject',type=str,nargs='+',default=['Math', 'Physics', 'Chemistry', 'Coding'],
                        help='List of subjects to load. Choose from: Chemistry, Coding, Math, Physics.')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--strategy', type=str, default='CoT', choices=['CoT', 'Direct'])
    parser.add_argument('--save_every', type=int, default=20, help='save every n problems')

    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]"
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)
    
    config = load_yaml('/chenyang_data/mmr/EMMA/multiagent-ft/multiagent-ft/gpt.yaml')
    
    # Load Dataset
    logging.info(f"Loading dataset {args.dataset_name}, subject: {args.subject}")
    sub_dataset_list = []
    for subj in args.subject:
        sub_dataset = load_dataset(args.dataset_name, subj, split=args.split)
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)

    agents = args.agents
    rounds = args.rounds
    random.seed(0)

    save_file_name = "{}{}_agents_{}_rounds.json".format(args.save_str, agents, rounds)
    if os.path.exists(save_file_name):
        logging.info("Results already exists.")
        logging.info(f"Reading {save_file_name}")
        with open(save_file_name, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    skip_pids = []
    if results:
        for pid, data in results.items():
            if 'debate_result' in data:
                skip_pids.append(pid)

        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems...")

    ports = [model_to_port[model] for model in args.models]
    keys = [model_to_key[model] for model in args.models]

    for idx, sample in enumerate(tqdm(dataset)):

        pid = sample['pid']
        if skip_pids and pid in skip_pids:
            continue
        
        # 输入数据集sample格式，返回query和gt_content
        sample = build_query(sample, config, args.strategy)
        question = sample['question']
        answer = sample['gt_content']
        
        # 去掉sample中的image
        problem: dict = sample.copy()
        for i in range(1, 6):
            problem.pop('image_' + str(i))
        # Question+Image+Task_Des
        messages = create_message(sample)
        # question = messages[0]['content'][0]['text']
        
        agent_contexts = [messages.copy() for agent in range(agents)]
        
        for round in range(rounds):
            # 第0轮，每个agent产生一个回答assistant_message
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    if args.summarize:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        random.shuffle(agent_contexts_other)
                        summary = summarize_message(agent_contexts_other[:5], port = llm_summary_port, key = keys[i])
                        message = construct_message_summary(summary, question, 2 * round - 1)
                    else:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        random.shuffle(agent_contexts_other)
                        message = construct_message(agent_contexts_other[:5], question, 2 * round - 1)
                    agent_context.append(message)

                # 生成一个vlm的回复
                completion = generate_answer(agent_context, port = ports[i], key = keys[i])
                # 提取内容构建下一个的输入
                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        results[pid] = problem
        agent_contexts_store = agent_contexts.copy()
        for i in range(len(agent_contexts_store)):
            agent_contexts[i][0]['content'][1]["image_url"] = None
        results[pid]['debate_result'] = (agent_contexts_store, answer)


        if idx == 2 or (idx % args.save_every == 0 and idx > 0) or idx == len(dataset) - 1:
            with open(save_file_name, 'w') as f:
                f.write(json.dumps(results, indent=2))
            logging.info(f"Save results to {save_file_name}")

    json.dump(results, open(save_file_name, "w"))
    pass