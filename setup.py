import os

# Download data dependencies
data_urls = {
    "training": [
        "https://raw.githubusercontent.com/vinid/safety-tuned-llamas/refs/heads/main/data/training/saferpaca_Instructions_2000.json",  # training data for safety_tuned_llama
        "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json",  # training data for secalign and struq
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",  # training data for secalign and struq
    ],
    "eval": [
        "https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json",  # evaluation data for secalign and struq
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/refs/heads/main/data/advbench/harmful_behaviors.csv",  # evaluation data for safety_tuned_llama
    ],
    "configs": [
        "https://raw.githubusercontent.com/vinid/safety-tuned-llamas/refs/heads/main/configs/alpaca.json",  # training configs for safety_tuned_llama
    ],
}

os.makedirs("data", exist_ok=True)
for data_type in data_urls:
    os.makedirs(os.path.join("data", data_type), exist_ok=True)
    for data_url in data_urls[data_type]:
        filename = data_url.split("/")[-1]
        filepath = os.path.join("data", data_type, filename)
        if not os.path.exists(filepath):
            os.system(f"wget {data_url} -O {filepath}")
            print(f"Downloaded {filename} to {filepath}")
        else:
            print(f"{filename} already exists at {filepath}")
