
import os
import torch
import json
import re
import time
import random
import argparse
import logging
import yaml
import base64
import numpy as np
import pandas as pd
from glob import glob
from io import BytesIO
from IPython.display import display, HTML

import deepspeed
import transformers
from trl import SFTConfig, SFTTrainer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import Qwen2VLProcessor, AutoProcessor, AutoModelForVision2Seq, AutoModelForVision2Seq, BitsAndBytesConfig

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def compare_generations(base_gen, ft_gen):
    # Create a DataFrame
    df = pd.DataFrame({
        'Base Generation': [base_gen],
        'Fine-tuned Generation': [ft_gen]
    })
    # Style the DataFrame
    styled_df = df.style.set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap',
        'border': '1px solid black',
        'padding': '10px',
        'width': '250px',  # Set width to 150px
        'overflow-wrap': 'break-word'  # Allow words to break and wrap as needed
    })
    
    # Display the styled DataFrame
    display(HTML(styled_df.to_html()))


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

def create_sft_message(sample,model):

    system_message = "You are an expert multi-modal reasoning model."
    query = sample['query']
    answer = sample['gt_content']

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
                if 'Qwen2-VL' in model:
                    all_contents.extend([
                        {
                            "type": "image",
                            "image": f"data:image;base64,{img_base64}"
                        }
                    ])
                else:
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
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": all_contents
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]
    return messages


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


def format_data(sample,config,strategy,model):
        
    # 添加query构建（Task-prompt）和gt_content
    sample = build_query(sample, config, strategy)
    # 去掉sample中的image
    problem: dict = sample.copy()
    for i in range(1, 6):
        problem.pop('image_' + str(i))
    # Question+Image+Task_Des
    messages = create_sft_message(sample,model)
    return {"messages": messages}



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

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652,151653,151655]
    else: 
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

def generate_description(message, model, processor):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--save_str", type = str, default = "/chenyang_data/mmr/EMMA/multiagent-ft/multiagent-ft/qwen2-7b-sft")
    parser.add_argument("--merged_path", type = str, default = "/chenyang_data/mmr/EMMA/multiagent-ft/multiagent-ft/qwen2-7b-sft-merged")
    parser.add_argument("--model", type = str, default = "Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--epochs", type = int, default = 3)
    parser.add_argument("--device", type = int, help = "device", default = 0)
    parser.add_argument("--temperature", default = 1, type = float, help = "temperature")
    parser.add_argument("--lr", default = 2e-4, type = float, help = "learning late")
    parser.add_argument('--dataset_name', type=str, default='luckychao/EMMA')
    parser.add_argument('--subject',type=str,nargs='+',default=['Math', 'Physics', 'Chemistry', 'Coding'],
                        help='List of subjects to load. Choose from: Chemistry, Coding, Math, Physics.')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--strategy', type=str, default='Direct', choices=['CoT', 'Direct'])
    

    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps before performing a backward/update pass")
    parser.add_argument("--gradient_checkpointing", default=True, help="Use gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="Optimizer type")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="Checkpoint saving strategy")
    parser.add_argument("--bf16", default=True, help="Use bfloat16 precision")
    parser.add_argument("--tf32", default=False, help="Use tf32 precision")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to hub")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Report metrics to tensorboard")
    parser.add_argument("--use_reentrant", default=False, help="Use reentrant checkpointing")
    parser.add_argument("--dataset_text_field", type=str, default="", help="Dummy field for collator")
    parser.add_argument("--skip_prepare_dataset", default=True, help="Skip dataset preparation for collator")

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
    
    # ========================= 1. Load Dataset =========================
    config = load_yaml('/chenyang_data/mmr/EMMA/multiagent-ft/multiagent-ft/gpt.yaml')
    logging.info(f"Loading dataset {args.dataset_name}, subject: {args.subject}")
    sub_dataset_list = []
    for subj in args.subject:
        sub_dataset = load_dataset(args.dataset_name, subj, split=args.split)
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)

    dataset_train = [format_data(sample, config, args.strategy, args.model) for sample in dataset]
    logging.info(f"Data is pre-processed!")

    # ========================= 2. Fine-tune VLM =========================
    # Hugging Face model id
    model_id = args.model 
    
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        # device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)

    logging.info(f"Model is loaded!")


    # sample
    sample = dataset_train[2]["messages"]
    text = processor.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=False)
    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    )
    model_inputs = model_inputs.to("cuda")
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    logging.info(text,'\n',output_text)
 
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", 
    )
    peft_model = get_peft_model(model, peft_config)
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    # print("LoRA applied layers:", peft_config.target_modules)

    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
    print("Trainable layers:")
    for layer in trainable_layers:
        print(layer)
    
    train_config = SFTConfig(
        output_dir=args.save_str,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        learning_rate=args.lr,
        bf16=args.bf16,
        tf32=args.tf32,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        push_to_hub=args.push_to_hub,
        report_to=args.report_to,
        gradient_checkpointing_kwargs={"use_reentrant": args.use_reentrant},
        dataset_text_field=args.dataset_text_field,
        dataset_kwargs={"skip_prepare_dataset": args.skip_prepare_dataset},
        deepspeed="zero_configs/zero2.json",  	# DeepSpeed配置文件
    )
    train_config.remove_unused_columns = False
    
    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=dataset_train,
        data_collator=collate_fn,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    logging.info(f"Start supervised fine-tuning!")
    trainer.train()
    logging.info(f"Model is saved to {args.save_str}")
    # save model 
    trainer.save_model(args.save_str)

    del model
    del trainer
    torch.cuda.empty_cache()

    # ========================= 3. Infer VLM =========================
    # Load Model base model
    model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16
            )
    processor = AutoProcessor.from_pretrained(model_id)
    # Preparation for inference

    sample = dataset_train[2]["messages"]
    base_description = generate_description(sample, model, processor)
    
    logging.info(base_description)

    model.load_adapter(args.save_str)
    ft_description = generate_description(sample, model, processor)
    logging.info(ft_description)

    compare_generations(base_description, ft_description)

    merge=False
    if merge:
        # Merge LoRA and base model and save
        peft_model = PeftModel.from_pretrained(model, args.save_str)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(args.merged_path,safe_serialization=True, max_shard_size="4GB")
        processor = AutoProcessor.from_pretrained(args.model)
        processor.save_pretrained(args.merged_path)
    


 
