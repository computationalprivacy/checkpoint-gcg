# Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses
This repository contains the source code for the paper "[Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses](https://arxiv.org/abs/2505.15738)" by Xiaoxue Yang*, Bozhidar Stevanoski*, Matthieu Meeus, and Yves-Alexandre de Montjoye (* denotes equal contribution).

## Abstract
Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. 
Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. 
GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. 
In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. 
Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. 
We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. 
We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. 
Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs.
Whenever found, these universal suffixes would enable an adversary to run a range of attacks. 
Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.


## Environment setup
+ Install environment dependencies

    ```
    git clone https://github.com/computationalprivacy/checkpoint-gcg
    cd checkpoint-gcg
    conda create -n checkpoint_gcg python==3.10 
    ```

+ Install package dependencies

  + For finetuning and attacking using SecAlign and Struq (we adopted the requirements in `requirements.txt` in [SecAlign](https://github.com/facebookresearch/SecAlign/tree/main)):

    ```
    pip install -r requirements_secalign_struq.txt
    ```

  + For finetuning and attacking using Safety-Tuned LLaMAs (we used the `requirements.txt` from [Safety-Tuned LLaMAs](https://github.com/vinid/safety-tuned-llamas) and installed the listed packages with their latest available versions):

    ```
    pip install -r requirements_safety_tuned_llama.txt
    ```

+ Download data dependencies

    ```
    python setup.py
    ```


## SecAlign 
+ To finetune Llama3-8B-Instruct and Mistral-7B-Instruct using SecAlign, run the following respective commands:
  ```
    bash scripts/defense/secalign_llama3instruct.sh
    bash scripts/defense/secalign_mistralinstruct.sh
  ```

## StruQ
+ Similarly, to finetune Llama3-8B-Instruct and Mistral-7B-Instruct using StruQ, run the following respective commands:
  ```
    bash scripts/defense/struq_llama3instruct.sh
    bash scripts/defense/struq_mistralinstruct.sh
  ```


## Safety-Tuned LLaMAs
+ To finetune for Safety-Tuned LLaMAs, run the following script, which uses `data/training/saferpaca_Instructions_2000.json` formatted with `data/configs/alpaca.json` as training data. 
    ```
    python safety_llama_finetuning.py
    ```

## Test
+ To run standard GCG and Checkpoint-GCG attacks against defense(s) and model(s), run the following to automatically generate attack `.sh` scripts:

    ```
    python scripts/attack/generate_attack_scripts.py
    ```
+ Run the relevant `.sh` script(s) in `scripts/attack` to launch the desired attacks:

    + Standard GCG vs Checkpoint-GCG
        + Standard GCG shell scripts (directly attacking the final finetuned model $\theta_C$) have "direct" in the script filenames 
        + Checkpoint-GCG shell scripts have "checkpoint" in the script filenames, as well as the appropriate checkpoint selection strategy
    + Attacking individual samples vs universal attack
        + Individual-sample attack shell scripts have "individual" in the script filenames
        + Universal attack shell scripts have "universal" in the script filenames 

## Acknowledgements and Citation
This repository is licensed under the MIT License. It builds on [SecAlign](https://github.com/facebookresearch/SecAlign/tree/main) (CC-BY-NC) and [Safety-Tuned LLaMAs](https://github.com/vinid/safety-tuned-llamas) (MIT). We thank the authors for open-sourcing their work.

If you find our work useful in your research, please consider citing:
```
@misc{yang2025alignmentpressurecaseinformed,
      title={Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses}, 
      author={Xiaoxue Yang and Bozhidar Stevanoski and Matthieu Meeus and Yves-Alexandre de Montjoye},
      year={2025},
      eprint={2505.15738},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2505.15738}, 
}
```