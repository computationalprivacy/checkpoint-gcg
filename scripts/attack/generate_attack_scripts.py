from attack_scripts_constants import *
import os

def generate_attack_scripts(
    model: str,
    defense: str,
    dir_name: str,
    sample_ids: str,
    device: int=0,
    checkpoint_attack: bool=True,
    individual_sample_attack: bool=True,
    hpc: bool=False,
    additional_args: str=None,
    script_per_sample: bool=False,
    gcg_num_steps_per_checkpoint: int=1000,
    gcg_num_steps_total: int=50000,
):
    current_time = "2026-01-01_01-01-01"

    if defense in ["metasecalign", "secalign", "struq"]:
        data_path = "data/eval/davinci_003_outputs.json"
    elif defense == "safety_ft":
        data_path = "data/eval/harmful_behaviors.csv"

    if individual_sample_attack:
        description = """
        # For the individual sample attack:
        # if gcg_global_budget:
        # - each sample is given a total GCG budget represented by gcg_num_steps_total,  
        # - attacking each checkpoint is given a budget of min(gcg_num_steps_per_checkpoint, remaining global budget)
        # otherwise:
        # - attacking each checkpoint is always given a budget of gcg_num_steps_per_checkpoint steps
        # - gcg_num_steps_total is always set to be the same as gcg_num_steps_per_checkpoint * len(all_checkpoints) in test.py
        """
    else:
        description = """
        # For the universal attack:
        # if gcg_global_budget:
        # - attacking each checkpoint is given a total GCG budget represented by gcg_num_steps_total,  
        # - when attacking a checkpoint, attacking n samples is given a budget of min(gcg_num_steps_per_sample, remaining global budget)
        # otherwise:
        # - when attacking a checkpoint, attacking n samples is always given a budget of gcg_num_steps_per_sample
        # - gcg_num_steps_total is always set to be the same as gcg_num_steps_per_sample * gcg_num_train_samples in test.py
        """

    checkpoint_dir = "checkpoint_dir"
    if model == "llama":
        if defense == "metasecalign":
            model_path = "facebook/Meta-SecAlign-8B"
        elif defense == "secalign":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct_dpo__NaiveCompletion_2025"
            if individual_sample_attack:
                checkpoint_strategies = ["step", "loss", "gradnorm"]
            else:
                checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_llama_secalign
            )
        elif defense == "struq":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct_Meta-Llama-3-8B-Instruct_NaiveCompletion_2025"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_llama_struq
            )
        elif defense == "safety_ft":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct_safety-tuned-2000"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_safety_llama
            )
    elif model == "mistral":
        if defense == "secalign":
            model_path = "mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2025"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_mistral_secalign
            )
        elif defense == "struq":
            model_path = "mistralai/Mistral-7B-Instruct-v0.1_Mistral-7B-Instruct-v0.1_NaiveCompletion_2025"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_mistral_struq
            )
    elif model == "qwen":
        if defense == "secalign":
            model_path = 'Qwen/Qwen2-1.5B-Instruct_dpo_NaiveCompletion_2025'
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_qwen_secalign
            )
        elif defense == "struq":
            model_path = "Qwen/Qwen2-1.5B-Instruct_Qwen2-1.5B-Instruct_NaiveCompletion_2025"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_qwen_struq
            )

    if not os.path.exists(f"{dir_name}/{model}/{defense}"):
        os.makedirs(f"{dir_name}/{model}/{defense}")

    direct_or_checkpoint = "checkpoint" if checkpoint_attack else "direct"
    universal_or_individual = "individual" if individual_sample_attack else "universal"
    individual_sample_attack_arg_values = individual_sample_attack_args.format(
        gcg_num_steps_per_checkpoint=gcg_num_steps_per_checkpoint,
        gcg_num_steps_total=gcg_num_steps_total,
    )

    universal_or_individual_attack_args = (
        individual_sample_attack_arg_values
        if individual_sample_attack
        else universal_attack_args
    )

    # enforce script_per_sample value based on the setting
    if hpc:
        script_per_sample = True
    if not individual_sample_attack:
        script_per_sample = False

    scripts_dir = f"{dir_name}/{model}/{defense}"
    if checkpoint_attack:
        for checkpoint_choice in checkpoint_strategies:
            bash_script_name = f"{scripts_dir}/{direct_or_checkpoint}_{universal_or_individual}_{checkpoint_choice}"

            if script_per_sample:
                bash_script_names = []
                pbs_script_names = []
                for sample_id in sample_ids.split():
                    bash_script_formatted = checkpoint_attack_bash_script.format(
                        current_time=current_time,
                        description=description,
                        model_path=model_path,
                        device=device,
                        defense=defense,
                        data_path=data_path,
                        checkpoint_dir=checkpoint_dir,
                        checkpoint_choice=checkpoint_choice,
                        sample_ids=sample_id,
                        universal_or_individual_attack_args=universal_or_individual_attack_args,
                        additional_args=additional_args if additional_args else "",
                    )
                    bash_script_formatted = f"{checkpoint_strategies2checkpoint_strings[checkpoint_choice]}\n{bash_script_formatted}"

                    bash_script_name_sample = f"{bash_script_name}_{sample_id}.sh"
                    bash_script_names.append(bash_script_name_sample)

                    with open(bash_script_name_sample, "w") as f:
                        f.write(bash_script_formatted)

                    with open(f"{scripts_dir}/run_all_bash_scripts.sh", "w") as f:
                        f.write("\n".join(bash_script_names))
                        f.write("\n")

                    if hpc:
                        hpc_script = f"{hpc_header}\n{bash_script_formatted}"

                        pbs_script_name = (
                            f"{checkpoint_choice}_{sample_id}_{universal_or_individual}.pbs"
                        )

                        with open(os.path.join(scripts_dir, pbs_script_name), "w") as f:
                            f.write(hpc_script)
                        pbs_script_names.append(
                            f"qsub {scripts_dir}/{pbs_script_name}"
                        )

                        with open(f"{scripts_dir}/submit_all_jobs.sh", "w") as f:
                            f.write("\n".join(pbs_script_names))
                            f.write("\n")
            else:
                bash_script_formatted = checkpoint_attack_bash_script.format(
                    current_time=current_time,
                    description=description,
                    model_path=model_path,
                    device=device,
                    defense=defense,
                    data_path=data_path,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_choice=checkpoint_choice,
                    sample_ids=sample_ids,
                    universal_or_individual_attack_args=universal_or_individual_attack_args,
                    additional_args=additional_args if additional_args else "",
                )
                bash_script_formatted = f"{checkpoint_strategies2checkpoint_strings[checkpoint_choice]}\n{bash_script_formatted}"

                with open(f"{bash_script_name}.sh", "w") as f:
                    f.write(bash_script_formatted)
    else:
        bash_script_name = os.path.join(scripts_dir, f"{direct_or_checkpoint}_{universal_or_individual}")

        if script_per_sample:
            bash_script_names = []
            pbs_script_names = []

            for sample_id in sample_ids.split():
                bash_script_formatted = direct_attack_bash_script.format(
                    current_time=current_time,
                    description=description,
                    model_path=model_path,
                    device=device,
                    defense=defense,
                    data_path=data_path,
                    checkpoint_dir=checkpoint_dir,
                    sample_ids=sample_id,
                    universal_or_individual_attack_args=universal_or_individual_attack_args,
                    additional_args=additional_args if additional_args else "",
                )

                bash_script_name_sample = f"{bash_script_name}_{sample_id}.sh"
                bash_script_names.append(bash_script_name_sample)
                with open(bash_script_name_sample, "w") as f:
                    f.write(bash_script_formatted)

                with open(f"{scripts_dir}/run_all_bash_scripts.sh", "w") as f:
                    f.write("\n".join(bash_script_names))
                    f.write("\n")

                if hpc:
                    hpc_script = f"{hpc_header}\n{bash_script_formatted}"

                    pbs_script_name = (
                        f"direct_attack_{sample_id}_{universal_or_individual}.pbs"
                    )
                    with open(
                            f"{scripts_dir}/direct_attack_{sample_id}_{universal_or_individual}.pbs",
                            "w",
                    ) as f:
                        f.write(hpc_script)
                    pbs_script_names.append(
                        f"qsub {scripts_dir}/{pbs_script_name}"
                    )

                    with open(f"{scripts_dir}/submit_all_jobs.sh", "w") as f:
                        f.write("\n".join(pbs_script_names))
                        f.write("\n")
        else:
            bash_script_formatted = direct_attack_bash_script.format(
                current_time=current_time,
                description=description,
                model_path=model_path,
                device=device,
                defense=defense,
                data_path=data_path,
                checkpoint_dir=checkpoint_dir,
                sample_ids=sample_ids,
                universal_or_individual_attack_args=universal_or_individual_attack_args,
                additional_args=additional_args if additional_args else "",
            )

            with open(f"{bash_script_name}.sh", "w") as f:
                f.write(bash_script_formatted)


def get_valid_model_defense_combinations(
    models: list,
    defenses: list,
):
    valid_combinations = []
    for model in models:
        for defense in defenses:
            if defense in ["safety_ft", "metasecalign"] and model != "llama":
                continue
            if defense in ["secalign", "struq"] and model not in ["llama", "mistral", "qwen"]:
                continue
            valid_combinations.append((model, defense))
    return valid_combinations


def main(
    dir_name: str,
    device: int=0,
    hpc: bool=False,
    additional_args: str=None,
    models: list=None,
    defenses: list=None,
    checkpoint_attack_options: list=None,
    individual_sample_attack_options: list=None,
    script_per_sample: bool=False
):
    """Main function for generating attack scripts for specified models and defenses, with options for checkpoint attacks and individual sample attacks."""
    valid_combinations = get_valid_model_defense_combinations(
        models=models,
        defenses=defenses
    )

    for model, defense in valid_combinations:
        if defense in ["metasecalign", "secalign", "struq"]:
            sample_ids = [
                12, 80, 33, 5, 187, 83, 116, 122, 90, 154, 
                45, 156, 52, 189, 96, 86, 204, 37, 66, 18, 
                170, 15, 7, 55, 92, 134, 125, 124, 158, 184, 
                75, 149, 138, 71, 186, 145, 176, 118, 16, 135, 
                190, 22, 104, 141, 4, 74, 136, 44, 63, 108
            ]
        elif defense == "safety_ft":
            sample_ids = [
                422, 107, 253, 235, 311, 15, 134, 263, 385, 97,
                433, 411, 154, 444, 378, 78, 327, 137, 427, 298,
                37, 90, 345, 231, 354, 220, 45, 344, 200, 21,
                46, 306, 316, 175, 498, 102, 157, 481, 487, 171,
                153, 242, 342, 502, 312, 340, 264, 240, 159, 492
            ]

        sample_ids = " ".join([str(x) for x in sample_ids])

        for checkpoint_attack in checkpoint_attack_options:
            for individual_sample_attack in individual_sample_attack_options:
                generate_attack_scripts(
                    model,
                    defense,
                    dir_name,
                    sample_ids,
                    device=device,
                    checkpoint_attack=checkpoint_attack,
                    individual_sample_attack=individual_sample_attack,
                    hpc=hpc,
                    additional_args=additional_args,
                    script_per_sample=script_per_sample,
                )


def generate_attack_scripts_per_sample(
    dir_name: str,
    device: int=0,
    hpc: bool=False,
    additional_args: str=None,
    models: list=None,
    defenses: list=None,
    sample_id_budget_dict: dict=None,
):
    """Generate attack scripts for each sample with specified budgets, for individual-sample direct attack against the final checkpoint only."""

    valid_combinations = get_valid_model_defense_combinations(
        models=models,
        defenses=defenses
    )

    for model, defense in valid_combinations:
        sample_id_budgets = sample_id_budget_dict[f"{model}_{defense}"]
        for sample_id, budget in sample_id_budgets.items():
            generate_attack_scripts(
                model=model,
                defense=defense,
                dir_name=dir_name,
                sample_ids=str(sample_id),
                device=device,
                checkpoint_attack=False,
                individual_sample_attack=True,
                hpc=hpc,
                additional_args=additional_args,
                script_per_sample=True,
                gcg_num_steps_per_checkpoint=budget,
                gcg_num_steps_total=budget,
            )



if __name__ == "__main__":
    ### Specify the parameters for generating attack scripts here
    device = 0
    dir_name = "attack_scripts"
    hpc = False
    additional_args = None
    # additional_args = """--gcg_random_init_baseline --initial_suffix 'custom_init'"""
    # additional_args = """--initial_suffix '! ! ! ! ! ! ! ! ! !'"""
    # additional_args = """--initial_suffix '! ! ! ! !'"""
    # additional_args = """--initial_suffix '! ! !'"""

    models = [
        "llama",
        # "mistral",
        # "qwen"
    ]
    defenses = [
        "metasecalign",
        # "secalign",
        # "struq",
        # "safety_ft"
    ]
    checkpoint_attack_options = [False]  # True, False
    individual_sample_attack_options = [False]  # True, False
    script_per_sample = False

    ## Generate scripts for Checkpoint-GCG attacks and direct GCG attacks with default budget
    main(
        dir_name,
        device=device,
        hpc=hpc,
        additional_args=additional_args,
        models=models,
        defenses=defenses,
        checkpoint_attack_options=checkpoint_attack_options,
        individual_sample_attack_options=individual_sample_attack_options,
        script_per_sample=script_per_sample,  
    )

    # ### Generate scripts for individual-sample direct GCG attacks with specified (Checkpoint-GCG) budgets
    # sample_id_budget_dict = {
    #     "llama_struq": {
    #         83: 215,
    #         90: 180,
    #         116: 120,
    #         122: 660,
    #         154: 8190
    #     },
    # }

    # generate_attack_scripts_per_sample(
    #     dir_name=dir_name,
    #     device=device,
    #     hpc=hpc,
    #     additional_args=additional_args,
    #     models=models,
    #     defenses=defenses,
    #     sample_id_budget_dict=sample_id_budget_dict,
    # )
