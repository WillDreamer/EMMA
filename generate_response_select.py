import argparse
import json
import os
import logging
from tqdm import tqdm
import time
from datasets import load_dataset, concatenate_datasets
from openai import OpenAI
from data_utils import load_yaml, verify_response, build_query

###! VLLM CoT outputs, LLM Reasons, Another LLM concludes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='luckychao/EMMA')
    parser.add_argument('--remark', type=str, default='_select_Qwen32B')
    parser.add_argument('--subject', nargs='+', type=str, default=['Math', 'Physics', 'Chemistry', 'Coding'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--strategy', type=str, default='CoT', choices=['CoT', 'Direct'])
    parser.add_argument('--config_path', type=str, default="configs/gpt.yaml")
    parser.add_argument('--output_path', type=str, default='/chenyang_data/mmr/EMMA/results/EMMA-reimplement/open-source')
    parser.add_argument('--save_every', type=int, default=20, help='save every n problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer generation')
    # Remote model
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help='remote llm engine',
                        choices=['chatgpt-4o-latest','Qwen/Qwen2.5-VL-7B-Instruct', 'grok-2-vision-latest', 'claude-3-5-sonnet-latest', 'gemini-2.0-flash-exp','gemini-2.0-flash-thinking-exp-1219','OpenGVLab/InternVL2_5-78B'])
    parser.add_argument('--api_key', type=str, default='')
    # Local model
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct', help="local model path or huggingface model name")
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.01)

    args = parser.parse_args()

    # Load Dataset
    logging.info(f"Loading dataset {args.dataset_name}, subject: {args.subject}")
    sub_dataset_list = []
    for subj in args.subject:
        sub_dataset = load_dataset(args.dataset_name, subj, split=args.split)
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)

    # Load Config
    logging.info(f"Loading config")
    config = load_yaml(args.config_path)

    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:30000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    llm_api_key = "EMPTY"
    llm_api_base = "http://localhost:30001/v1"
    llm_client = OpenAI(
        api_key=llm_api_key,
        base_url=llm_api_base,
    )
    from models import qwen_2_5_select
    model = qwen_2_5_select.GPT_Model(client, llm_client, args.model, \
            temperature=args.temperature, \
            max_tokens=args.max_tokens)

    logging.info(f"Model loaded!")

    args.output_path = os.path.join(args.output_path,args.model+args.remark+'.json')
    
    if os.path.exists(args.output_path):
        logging.info("Results already exists.")
        logging.info(f"Reading {args.output_path}")
        with open(args.output_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    skip_pids = []
    if not args.rerun and results:
        for pid, data in results.items():
            if 'response' in data and verify_response(data['response']):
                skip_pids.append(pid)

        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems...")

    logging.info(f"Starting to generate.....")
    for idx, sample in enumerate(tqdm(dataset)):
        ### sample.keys() ['query', 'gt_content', 'pid', 'question', 'options', 'answer', 'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'solution', 'subject', 'task', 'category', 'source', 'type', 'context']
        pid = sample['pid']
        if skip_pids and pid in skip_pids:
            continue

        # 输入数据集sample格式，返回query和gt_content
        sample = build_query(sample, config, args.strategy)
        problem: dict = sample.copy()
        for i in range(1, 6):
            problem.pop('image_' + str(i))
        try:
            response, vlm_response, llm_reason_response = model.get_response(sample)
            results[pid] = problem
            results[pid]['response'] = response
            results[pid]['vlm_response'] = vlm_response
            results[pid]['llm_reason_response'] = llm_reason_response
        except Exception as e:
            logging.error(f"Error in generating answer for {pid}")
            logging.error(e)
            results[pid] = problem
            results[pid]['error'] = str(e)
        
        if idx == 2 or (idx % args.save_every == 0 and idx > 0) or idx == len(dataset) - 1:
            try:
                with open(args.output_path, 'w') as f:
                    f.write(json.dumps(results, indent=2))
                logging.info(f"Save results to {args.output_path}")
            except Exception as e:
                logging.info(f"Error in saving {args.output_path}")
                logging.info(e)
    
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(results, indent=2))
    logging.info(f"Save results to {args.output_path}")

    logging.info("End Generation......")


if __name__ == "__main__":
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

    main()












