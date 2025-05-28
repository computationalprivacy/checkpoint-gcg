from attack_scripts_constants import *
import os


def main(
    model,
    defense,
    dir_name,
    checkpoint_attack=True,
    individual_sample_attack=True,
    hpc=False,
    additional_args=None,
):
    current_time = "2026-01-01_01-01-01"

    if defense in ["secalign", "struq"]:
        sample_ids = [12, 80, 33, 5, 187, 83, 116, 122, 90, 154]
        data_path = "data/eval/davinci_003_outputs.json"
    elif defense == "safety_tuned_llama":
        sample_ids = [154, 444, 107, 385, 78, 134, 298, 311, 263, 15]
        data_path = "data/eval/harmful_behaviors.csv"
    sample_ids = " ".join([str(x) for x in sample_ids])

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

    if model == "llama":
        if defense == "secalign":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct_dpo__NaiveCompletion_2025-04-23-17-33-07"
            if individual_sample_attack:
                checkpoint_strategies = ["step", "loss", "gradnorm"]
            else:
                checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_llama_secalign
            )
        elif defense == "struq":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct_Meta-Llama-3-8B-Instruct_NaiveCompletion_2025-05-09-18-08-53"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_llama_struq
            )
    elif model == "mistral":
        if defense == "secalign":
            model_path = "mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_2025-04-27-15-02-43"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_mistral_secalign
            )
        elif defense == "struq":
            model_path = "mistralai/Mistral-7B-Instruct-v0.1_Mistral-7B-Instruct-v0.1_NaiveCompletion_2025-05-10-13-41-28"
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_mistral_struq
            )
    elif model == "safety_llama":
        model_path = "meta-llama/Meta-Llama-3-8B-Instruct_safety-tuned-2000"
        if defense == "safety_tuned_llama":
            checkpoint_strategies = ["gradnorm"]
            checkpoint_strategies2checkpoint_strings = (
                checkpoint_strategies2checkpoint_strings_safety_llama
            )

    if not os.path.exists(f"{dir_name}/{model}/{defense}"):
        os.makedirs(f"{dir_name}/{model}/{defense}")

    direct_or_checkpoint = "checkpoint" if checkpoint_attack else "direct"
    universal_or_individual = "individual" if individual_sample_attack else "universal"

    universal_or_individual_attack_args = (
        individual_sample_attack_args
        if individual_sample_attack
        else universal_attack_args
    )

    if checkpoint_attack:
        for checkpoint_choice in checkpoint_strategies:
            bash_script_formatted = checkpoint_attack_bash_script.format(
                current_time=current_time,
                description=description,
                model_path=model_path,
                defense=defense,
                data_path=data_path,
                checkpoint_choice=checkpoint_choice,
                sample_ids=sample_ids,
                universal_or_individual_attack_args=universal_or_individual_attack_args,
                additional_args=additional_args if additional_args else "",
            )
            bash_script_formatted = f"{checkpoint_strategies2checkpoint_strings[checkpoint_choice]}\n{bash_script_formatted}"

            with open(
                f"{dir_name}/{model}/{defense}/{direct_or_checkpoint}_{universal_or_individual}_{checkpoint_choice}.sh",
                "w",
            ) as f:
                f.write(bash_script_formatted)
    else:
        bash_script_formatted = direct_attack_bash_script.format(
            current_time=current_time,
            description=description,
            model_path=model_path,
            defense=defense,
            data_path=data_path,
            sample_ids=sample_ids,
            universal_or_individual_attack_args=universal_or_individual_attack_args,
            additional_args=additional_args if additional_args else "",
        )

        with open(
            f"{dir_name}/{model}/{defense}/{direct_or_checkpoint}_{universal_or_individual}.sh",
            "w",
        ) as f:
            f.write(bash_script_formatted)

    if hpc:
        pbs_script_names = []

        if checkpoint_attack:
            for checkpoint_choice in checkpoint_strategies:
                for sample_id in sample_ids.split():
                    bash_script_formatted = checkpoint_attack_bash_script.format(
                        current_time=current_time,
                        description=description,
                        model_path=model_path,
                        defense=defense,
                        data_path=data_path,
                        checkpoint_choice=checkpoint_choice,
                        sample_ids=sample_id,
                        universal_or_individual_attack_args=universal_or_individual_attack_args,
                        additional_args=additional_args if additional_args else "",
                    )
                    hpc_script = f"{hpc_header}\n{checkpoint_strategies2checkpoint_strings[checkpoint_choice]}\n{bash_script_formatted}"

                    pbs_script_name = (
                        f"{checkpoint_choice}_{sample_id}_{universal_or_individual}.pbs"
                    )
                    with open(
                        f"{dir_name}/{model}/{defense}/{checkpoint_choice}_{sample_id}_{universal_or_individual}.pbs",
                        "w",
                    ) as f:
                        f.write(hpc_script)
                    pbs_script_names.append(
                        f"qsub {dir_name}/{model}/{defense}/{pbs_script_name}"
                    )
        else:
            for sample_id in sample_ids.split():
                bash_script_formatted = direct_attack_bash_script.format(
                    current_time=current_time,
                    description=description,
                    model_path=model_path,
                    defense=defense,
                    data_path=data_path,
                    sample_ids=sample_id,
                    universal_or_individual_attack_args=universal_or_individual_attack_args,
                    additional_args=additional_args if additional_args else "",
                )
                hpc_script = f"{hpc_header}\n{bash_script_formatted}"

                pbs_script_name = (
                    f"direct_attack_{sample_id}_{universal_or_individual}.pbs"
                )
                with open(
                    f"{dir_name}/{model}/{defense}/direct_attack_{sample_id}_{universal_or_individual}.pbs",
                    "w",
                ) as f:
                    f.write(hpc_script)
                pbs_script_names.append(
                    f"qsub {dir_name}/{model}/{defense}/{pbs_script_name}"
                )
        with open(f"{dir_name}/{model}/{defense}/submit_all_jobs.sh", "a") as f:
            f.write("\n".join(pbs_script_names))
            f.write("\n")


if __name__ == "__main__":
    dir_name = "scripts/attack"
    hpc = False
    additional_args = None
    models = ["llama", "mistral", "safety_llama"]
    defenses = ["secalign", "struq", "safety_tuned_llama"]

    valid_combinations = []
    for model in models:
        for defense in defenses:
            if defense == "safety_tuned_llama" and model != "safety_llama":
                continue
            if defense in ["secalign", "struq"] and model not in ["llama", "mistral"]:
                continue
            valid_combinations.append((model, defense))

    for model, defense in valid_combinations:
        for checkpoint_attack in [True, False]:
            for individual_sample_attack in [True, False]:
                if (
                    defense == "safety_tuned_llama" and not individual_sample_attack
                ):  # universal attack not yet supported for jailbreak attacks
                    continue
                main(
                    model,
                    defense,
                    dir_name,
                    checkpoint_attack=checkpoint_attack,
                    individual_sample_attack=individual_sample_attack,
                    hpc=hpc,
                    additional_args=additional_args,
                )
