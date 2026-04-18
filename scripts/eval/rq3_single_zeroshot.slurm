#!/bin/bash
#SBATCH --job-name=rq3-single-zs
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/rq3-single-zeroshot-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobyyang7@outlook.com
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/chengzhi.ucsb/code/toby/latent-MAS
source /home/chengzhi.ucsb/code/toby/latent-MAS/.venv/bin/activate

python -c "
import sys; sys.path.insert(0, '.')
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import create_dataset
from src.utils.answer_extraction import extract_answer, math_is_equivalent, MATH_EQUIVALENT_TASKS

model_name = 'Qwen/Qwen3-4B'
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')
model.eval()

system_prompt = (
    'You are a helpful assistant. You must reason step-by-step to solve the provided question.\n'
    'Your output MUST follow this format: first show your reasoning inside <think> </think> tags, '
    'then provide the final answer inside <answer> </answer> tags.\n'
    'Example: <think> step-by-step reasoning here </think><answer> final answer here </answer>'
)

tasks = {
    'gsm8k': {'split': 'test', 'max_tokens': 8192},
    'math500': {'split': 'test', 'max_tokens': 16384},
}

all_results = {}

for task_name, cfg in tasks.items():
    print(f'\n=== {task_name} ===')
    dataset = create_dataset(task=task_name, split=cfg['split'])
    print(f'Samples: {len(dataset)}')

    correct = 0
    total = 0

    for i in range(len(dataset)):
        item = dataset[i]
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': item['question']},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(text, return_tensors='pt').to('cuda')

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg['max_tokens'], do_sample=False, pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        pred = extract_answer(response, task_type=task_name)
        gold = item['answer'] if task_name in MATH_EQUIVALENT_TASKS else extract_answer(item['answer'], task_type=task_name)

        if task_name in MATH_EQUIVALENT_TASKS:
            is_correct = math_is_equivalent(pred, gold, raw_text=response)
        else:
            is_correct = pred.strip() == gold.strip()

        if is_correct:
            correct += 1
        total += 1

        if total % 50 == 0:
            print(f'  {total}/{len(dataset)} acc={correct/total*100:.1f}%', flush=True)

    acc = correct / total * 100 if total > 0 else 0
    print(f'{task_name}: {correct}/{total} = {acc:.1f}%')
    all_results[task_name] = {'correct': correct, 'total': total, 'accuracy': acc}

print('\n=== SUMMARY ===')
for task_name, r in all_results.items():
    print(f'  {task_name:>15}: {r[\"accuracy\"]:5.1f}% ({r[\"correct\"]}/{r[\"total\"]})')

with open('outputs/rq3/single_agent_zeroshot_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('Results saved to outputs/rq3/single_agent_zeroshot_results.json')
"

echo "=== DONE ==="
