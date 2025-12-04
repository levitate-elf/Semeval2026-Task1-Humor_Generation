import os
import torch

# 使用 HF 镜像（你原来就有这一行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets
from trl import PPOConfig, PPOTrainer

# ===================== 1. Tokenizer =====================
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'

print(tokenizer)

# ===================== 2. 数据集：IMDB -> input_ids =====================
dataset = load_dataset('imdb')
# 把 train/test/unsupervised 拼一起，然后再重新切
dataset = concatenate_datasets(list(dataset.values()))

def encode_example(data):
    return {
        'input_ids': tokenizer.encode(
            data['text'],
            truncation=True,
            max_length=5
        )
    }

dataset = dataset.map(encode_example, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=2000)

print(dataset, dataset['train'][0])

# ===================== 3. 加载 actor / ref_actor / critic / ref_critic =====================
actor_path = r"E:\PythonProject1\humor_generation\5_ppo\model\actor"
critic_path = r"E:\PythonProject1\humor_generation\5_ppo\model\critic"

model_actor = AutoModelForCausalLM.from_pretrained(actor_path)
model_actor_ref = AutoModelForCausalLM.from_pretrained(actor_path)

model_critic = AutoModelForSequenceClassification.from_pretrained(
    critic_path, num_labels=1
)
model_critic_ref = AutoModelForSequenceClassification.from_pretrained(
    critic_path, num_labels=1
)

# 设备交给 TRL/accelerate 自己管，一般不用手动 .to(device)

# ===================== 4. PPOConfig & PPOTrainer =====================

ppo_config = PPOConfig(
    # 这个字段在不少版本里是可选的，加上有利于 log

    # 输出目录
    output_dir="model/ppo_trl",

    # ★ 改动 2：学习率调小一点，更新更温和
    learning_rate=5e-6,

    # batch / grad 配置：保持和你原来差不多的量级
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,

    # ★ 改动 3：只做 1 个 PPO epoch，别来回拧太多次
    num_ppo_epochs=1,

    # ★ 改动 4：适当调大 batch，减少噪声，但整体还是很轻量
    batch_size=32,
    mini_batch_size=8,

    # ★ 改动 5：总 episode 数从 200000 大幅降到 5000 左右
    #   这是个“试水”级别训练，不会把策略掰到完全变形
    total_episodes=5000,

)

# 初始化 PPOTrainer
trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,      # 就是 tokenizer
    model=model_actor,               # 你的 actor
    ref_model=model_actor_ref,       # 你的 actor_ref（只用来算 KL，不更新）
    reward_model=model_critic_ref,   # 用 critic_ref 当 reward model
    value_model=model_critic,        # 用 critic 当 value model（估值网络）
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# ===================== 5. 开始 PPO 训练 =====================
trainer.train()

# 训练结束后，把强化过的 actor 存起来
model_actor.save_pretrained("model/trl_fix")
tokenizer.save_pretrained("model/trl_fix")


import torch
import random
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(THIS_DIR)
CKPT_PATH = os.path.join(PROJECT_DIR, "model", "trl_fix")

from transformers import AutoTokenizer, AutoModelForCausalLM
# ckpt_path = "model/ppo_trl/checkpoint-500"
# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
# tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
tokenizer = AutoTokenizer.from_pretrained("5_ppo/model/trl_fix")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'

print(tokenizer)

from datasets import load_dataset, concatenate_datasets

dataset = load_dataset('imdb')
dataset = concatenate_datasets(list(dataset.values()))
dataset = dataset.remove_columns(['label'])

print(dataset, dataset[0])


from transformers import AutoModelForCausalLM

# load model from manual ppo
# model_actor = AutoModelForCausalLM.from_pretrained('model/ppo').to(device)
# load model from trl ppo
model_actor = AutoModelForCausalLM.from_pretrained(CKPT_PATH).to(device)
# model_actor = AutoModelForCausalLM.from_pretrained(ckpt_path).to(device)

print(model_actor.config)

#====question====
question = random.choices(dataset, k=12)
question = [i['text'] for i in question]

question = tokenizer(question,
                     padding=True,
                     truncation=True,
                     max_length=5,
                     return_tensors='pt').input_ids.to(device)

#====answer====
answer = model_actor.generate(input_ids=question,
                              min_length=-1,
                              max_length=50,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              top_k=0.0,
                              top_p=1.0,
                              do_sample=True)
answer = answer[:, question.shape[1]:]

for q, a in zip(question, answer):
    print(tokenizer.decode(q), '->', tokenizer.decode(a))
    print('==============')

