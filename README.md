<p align="center">
    <img src="assets/emma-small.jpg" width="30%"> <br>
</p>

# EMMA: An Enhanced MultiModal ReAsoning Benchmark

🌟  This is the official repository for the paper "[Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark](https://www.arxiv.org/abs/2501.05444)", which contains generation and evaluation code for the **EMMA** benchmark.

[[🌐 Homepage](https://emma-benchmark.github.io/)] [[🤗EMMA](https://huggingface.co/datasets/luckychao/EMMA)] [[🤗EMMA-mini](https://huggingface.co/datasets/luckychao/EMMA-mini)] [[📖 ArXiv Paper](https://www.arxiv.org/abs/2501.05444)]

## 💥 Results


| **Method**                     | **CoT**  | **EMMA: Math (892)** | **EMMA: Phys. (156)** | **EMMA: Chem. (1,176)** | **EMMA: Coding (564)** | **EMMA: Overall (2,788)** | **EMMA-mini: Math (100)** | **EMMA-mini: Phys. (100)** | **EMMA-mini: Chem. (100)** | **EMMA-mini: Coding (100)** | **EMMA-mini: Overall (400)** |
|--------------------------------|----------|----------------------|-----------------------|-------------------------|------------------------|---------------------------|---------------------------|----------------------------|----------------------------|-----------------------------|------------------------------|
| Random choice                  | $-$      | 14.01                | 25.64                 | 16.50                   | 25.71                  | 18.08                     | 13.00                     | 23.00                      | 27.00                      | 28.00                       | 22.75                        |
| Human Expert                   | $-$      | $-$                  | $-$                   | $-$                     | $-$                    | $-$                       | 75.00                     | 64.50                      | 86.00                      | 85.50                       | 77.75                        |
| Claude 3.5 Sonnet              | ✗ | 25.34                | 33.97                 | 40.90         | 38.65                  | 35.08                     | 23.00                     | 34.00                      | {44.00}               | 35.00                       | 34.00                        |
| Gemini 2.0 Flash               | ✗ | 23.88                | 38.46                 | 36.31                   | {42.02}         | 33.61                     | 20.00                     | 40.00                      | 36.00                      | 41.00                       | 34.25                        |
| GPT-4o                         | ✗ | 27.24                | 38.46                 | 31.89                   | 40.07                  | 32.42                     | 30.00                     | 38.00                      | 33.00                      | 40.00                       | 35.25                        |
| Qwen2-VL-72B-Instruct          | ✗ | {33.07}         | 42.31                 | 32.06                   | 34.57                  | 33.46                     | {38.00}            | 40.00                      | 34.00                      | 37.00                       | 37.25                        |
| LLaVA-Onevision-72B            | ✗ | 27.69                | 35.90                 | 25.26                   | 28.72                  | 27.33                     | 25.00                     | 32.00                      | 24.00                      | 28.00                       | 27.25                        |
| InternVL2-Llama3-76B           | ✗ | 25.11                | 22.44                 | 24.06                   | 27.84                  | 25.07                     | 31.00                     | 22.00                      | 21.00                      | 28.00                       | 25.50                        |
| InternVL2.5-78B                | ✗ | 31.39                | 38.46                 | 35.20                   | 31.91                  | 33.50                     | 30.00                     | 40.00                      | 38.00                      | 33.00                       | 35.25                        |
| Claude 3.5 Sonnet (CoT)        | $\cmark$ | 29.37                | 41.03                 | {41.07}            | 40.60                  | {37.23} (<span style="color:ForestGreen;">$\uparrow 2.15$</span>) | 30.00                     | 38.00                      | {41.00}             | 39.00                       | 37.00 (<span style="color:ForestGreen;">$\uparrow 3.00$</span>) |
| Gemini 2.0 Flash (CoT)         | $\cmark$ | 25.90                | 38.46                 | 24.66                   | 40.96                  | 29.12 (<span style="color:Red;">$\downarrow 4.48$</span>)   | 24.00                     | 41.00                      | 36.00                      | {44.00}              | 36.25 (<span style="color:ForestGreen;">$\uparrow 2.00$</span>) |
| GPT-4o (CoT)                   | $\cmark$ | 25.56                | {43.59}        | 33.67                   | 39.01                  | 32.71 (<span style="color:ForestGreen;">$\uparrow 0.29$</span>) | 27.00                     | 44.00                      | 35.00                      | 38.00                       | 36.00 (<span style="color:ForestGreen;">$\uparrow 0.75$</span>) |
| Qwen2-VL-72B-Instruct (CoT)      | $\cmark$ | 27.69                | 34.62                 | 24.57                   | 29.43                  | 27.12 (<span style="color:Red;">$\downarrow 6.35$</span>)   | 35.00                     | 34.00                      | 32.00                      | 23.00                       | 31.00 (<span style="color:Red;">$\downarrow 6.25$</span>) |
| LLaVA-Onevision-72B (CoT)       | $\cmark$ | 22.42                | 15.38                 | 22.70                   | 30.67                  | 23.82 (<span style="color:Red;">$\downarrow 3.52$</span>)   | 23.00                     | 26.00                      | 23.00                      | 29.00                       | 25.25 (<span style="color:Red;">$\downarrow 2.00$</span>) |
| InternVL2-Llama3-76B (CoT)      | $\cmark$ | 22.20                | 32.05                 | 19.73                   | 30.32                  | 23.35 (<span style="color:Red;">$\downarrow 1.72$</span>)   | 27.00                     | 33.00                      | 21.00                      | 32.00                       | 28.25 (<span style="color:ForestGreen;">$\uparrow 2.75$</span>) |
| InternVL2.5-78B (CoT)           | $\cmark$ | 25.56                | 39.74                 | 27.47                   | 25.18                  | 27.08 (<span style="color:Red;">$\downarrow 6.42$</span>)   | 31.00                     | 36.00                      | 24.00                      | 19.00                       | 27.50 (<span style="color:Red;">$\downarrow 7.75$</span>) |
| Grok-2-vision-latest      | $-$      |  -       | -          | -                   | -           | -              | 32.00                     | 48.00              | 37.00              | 35.00                       | 38.00              |
| Gemini 2.0 Flash Thinking      | $-$      | {31.61}       | {56.41}          | 37.93                   | {43.44}           | {38.06}              | 35.00                     | {57.00}               | {41.00}              | 41.00                       | {43.50}              |
| o1                             | $-$      | $-$                  | $-$                   | $-$                     | $-$                    | $-$                       | {41.00}              | {49.00}             | 40.00                      | {53.00}                | {45.75}                |

## 👀 About EMMA

The ability to organically reason **over** and **with** both text and images is a pillar of human intelligence, yet the ability of Multimodal Large Language Models (MLLMs) to perform such multimodal reasoning remains under-explored.
We introduce **EMMA (Enhanced MultiModal reAsoning)**, a benchmark targeting organic multimodal reasoning across mathematics, physics, chemistry, and coding. 
EMMA tasks demand advanced cross-modal reasoning that cannot be solved by thinking separately in each modality, offering an enhanced test suite for MLLMs' reasoning capabilities. 

EMMA is composed of 2,788 problems, of which 1,796 are newly constructed, across four domains. Within each subject, we further provide fine-grained labels for each question based on the specific skills it measures. 

<p align="center">
    <img src="assets/EMMA.jpg" width="90%"> <br>
  <b>Overview of EMMA.</b> 
</p>

Our evaluation of state-of-the-art MLLMs on EMMA reveals significant limitations in handling complex multimodal and multi-step reasoning tasks, with even advanced techniques like Chain-of-Thought prompting and test-time compute scaling underperforming. 
These findings underscore the need for improved multimodal architectures and training paradigms to close the gap between human and model reasoning in multimodality. 

## 🏆 Leaderboard

The leaderboard is available [here](https://emma-benchmark.github.io/#leaderboard).

## 📖 Dataset Usage

### Data Downloading

To create a more balanced subset of EMMA, we randomly sample 400 questions (100 per subject) from the benchmark and get EMMA-mini[🤗](https://huggingface.co/datasets/luckychao/EMMA-mini).

You can download both two datasets by the following command (Taking downloading math data as an example):

```python
from datasets import load_dataset

dataset = load_dataset("luckychao/EMMA", "Math", split="test")
```

```python
from datasets import load_dataset

dataset = load_dataset("luckychao/EMMA-mini", "Math", split="test")
```

### Data Format

The dataset is provided in jsonl format and contains the following attributes:

```
{
    "pid": [string] Problem ID, e.g., “math_1”,
    "question": [string] The question text,
    "options": [list] Choice options for multiple-choice problems. For free-form problems, this could be a 'none' value,
    "answer": [string] The correct answer for the problem,
    "image_1": [image] ,
    "image_2": [image] ,
    "image_3": [image] ,
    "image_4": [image] ,
    "image_5": [image] ,
    "solution": [string] The detailed thinking steps required to solve the problem,
    "subject": [string] The subject of data, e.g., “Math”, “Physics”...,
    "task": [string] The task of the problem, e.g., “Code Choose Vis”,
    "category": [string] The category of the problem, e.g., “2D Transformation”,
    "source": [string] The original source dataset of the data, e.g., “math-vista”. For handmade data, this could be “Newly annotated” ,
    "type": [string] Types of questions, e.g., “Multiple Choice”, “Open-ended”,
    "context": [string] Background knowledge required for the question. For problems without context, this could be a 'none' value,
}
```

## 📈 Evaluation

### Responses Generation
Our repository supports the evaluation of open source models such as Qwen2-VL, InternVL, LLaVA, and closed source models such as GPT, Gemini, Claude, etc. 

The 1st step:
You can generate responses of these models by using the following commands:


Open-source Model:
```
 python generate_response.py \
 --dataset_name 'luckychao/EMMA' \
 --split 'test' \
 --subject 'Math' 'Physics' 'Chemistry' 'Coding' \
 --strategy 'CoT' \
 --config_path 'configs/gpt.yaml' \
 --model_path '/data1/whx/InternVL2_5-78B' \
 --output_path '/home/whx/MM-Reasoning/EMMA/results/EMMA-reimplement/open-source' \
 --max_tokens 4096 \
 --temperature 0.7 \
 --save_every 20
```

Close-source Model:

```
python generate_response.py --subject 'Math' 'Physics' 'Chemistry' 'Coding' --model grok-2-vision-latest --api_key 'xai-kxye8FXAGUBcT6YIpCZCNteZZrPBwp0f7yHEztNg1xpwFT0FkGzEI9KP6yPZT5Llv7BNiJlgVEUHaHO0'
```

### Answer Evaluation

Once all the model outputs have been generated, execute the `evaluate.py` function to extract the short answer text from the detailed response and evaluate the correctness of the answers.
We offer two evaluation methods: **Fast-eval** and **LLMs-eval**. The fast-eval method employs rule-based extraction for quicker processing, while the LLMs-eval method leverages advanced models like GPT-4o to enhance precision in extraction and evaluation.

Fast-extract:
```
python evaluate.py \
--results_dir '/home/whx/MM-Reasoning/EMMA/results/EMMA-mini-reimplement/closed-source/' \
--response_label 'response' \
--save_every 20
```

LLMs-eval:
```
python evaluate.py \
--results_dir 'path_to_your_results_dir' \
--response_label 'response' \
--save_every 20 \
--gpt_eval \
--api_key '' \
--model 'chatgpt-4o-latest' 
```

### Score Calculation

Finally, execute `python evaluation/calculate_acc.py` to calculate the final score based on the evaluation results. 
This step will compute overall accuracy as well as accuracy for each subject, category, and tasks.

## LeaderBoard Supplement
- grok-2-vision-latest: EMMA-mini 38.0%


## 📝Citation

If you find our benchmark useful in your research, please consider citing this BibTex:

```
@article{hao2025can,
  title={Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark},
  author={Hao, Yunzhuo and Gu, Jiawei and Wang, Huichen Will and Li, Linjie and Yang, Zhengyuan and Wang, Lijuan and Cheng, Yu},
  journal={arXiv preprint arXiv:2501.05444},
  year={2025}
}
```
