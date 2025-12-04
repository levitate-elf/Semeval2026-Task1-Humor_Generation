import os
import torch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

# ===== 1. Tokenizer =====
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'left'

print(tokenizer)

# ===== 2. 数据集：IMDB -> input_ids 形式 =====
dataset = load_dataset('imdb')
dataset = concatenate_datasets(list((dataset.values())))

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

# ===== 3. 加载 actor / ref_actor / critic / ref_critic =====
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

model_actor = AutoModelForCausalLM.from_pretrained('model/actor')
model_actor_ref = AutoModelForCausalLM.from_pretrained('model/actor')

model_critic = AutoModelForSequenceClassification.from_pretrained(
    'model/critic', num_labels=1
)
model_critic_ref = AutoModelForSequenceClassification.from_pretrained(
    'model/critic', num_labels=1
)

# ===== 4. PPOv2Config & PPOv2Trainer（新版导入方式） =====
# ★ 关键修改：从 trl 顶层导入，而不是 trl.trainer.ppov2_trainer
from trl import PPOConfig, PPOTrainer

policy_model_path  = "model/actor"
reward_model_path  = "model/critic"

ppo_config = PPOConfig(
    output_dir="model/ppo_trl",         # 保存目录
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,

    # PPO 相关
    num_ppo_epochs=2,
    batch_size=8,
    mini_batch_size=2,
    total_episodes=20_0000,
)

# ===== 5. 初始化 PPOTrainer（用 PPOTrainer，而不是 PPOv2Trainer）=====
trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,      # 就是 tokenizer
    model=model_actor,               # 你的 actor
    ref_model=model_actor_ref,       # 你的 actor_ref
    reward_model=model_critic_ref,   # 用 critic_ref 当 reward model
    value_model=model_critic,        # 用 critic 当 value model
    train_dataset=dataset["train"],  # 还是那份 IMDB tokenized 数据
    eval_dataset=dataset['test'],
    # eval_dataset 可以不传，或者你自己后面手动 eval
    # data_collator 默认为 None，也可以用 DataCollatorWithPadding(tokenizer)
)

trainer.train()

# 训练结束后，把强化过的 actor 存起来
model_actor.save_pretrained("model/trl")
