import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from datasets import Dataset
import numpy as np
import pandas as pd
import math
import os

def unlearn(
    model_path,
    output_dir,
    forget_dataset_path,
    retain_dataset_path,
):

    def forward_with_cache(model, inputs, module, no_grad=True):
        # define a tensor with the size of our cached activations
        cache = []
        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None 
        
        hook_handle = module.register_forward_hook(hook)

        if no_grad:
            with torch.no_grad():
                _ = model(**inputs)
        else:
            _ = model(**inputs)
            
        hook_handle.remove()

        return cache[0]

    def get_params(model, layer_ids, param_ids):
        params = []
        for layer_id in layer_ids:
            for i, p in enumerate(model.model.layers[layer_id].parameters()):
                if i in param_ids:
                    params.append(p)


        total_params = sum(p.numel() for p in params)
        print(f"Total trainable parameters: {total_params:,}")
        return params


    def load_model(model_name_or_path):
        torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        # model = model.to('cuda:0')
        print("model running on device: ", model.device)
        print("tokenizer_name_or_path", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.mask_token_id = tokenizer.eos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.cls_token_id = tokenizer.eos_token_id

        return model, tokenizer


    def get_data(forget_corpora, retain_corpora, batch_size=4):
        def get_dataset(raw_data):
            data = []
            for x in raw_data:
                data.append(str(x['text']))

            data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
            return data # batched data

        return (
            [get_dataset(c) for c in forget_corpora],
            [get_dataset(c) for c in retain_corpora]
        )
    
    def run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        layer_ids,
        param_ids,
        module_str,
        layer_id,
        lr,
        steering_coeff_list,
        alpha,
        scale
    ):

        updated_model = updated_model.train()
        params = get_params(updated_model, layer_ids, param_ids)
        
        if not params:
            raise ValueError("No parameters selected for optimization. Check layer_ids and param_ids.")
        optimizer = AdamW(params, lr=lr)
        frozen_module = eval(
            module_str.format(model_name="frozen_model", layer_id=layer_id)
        )
        updated_module = eval(
            module_str.format(model_name="updated_model", layer_id=layer_id)
        )

        task_num_batches_forget = [len(forget_data_list[topic_idx]) for topic_idx in range(len(forget_data_list))]
        ones = np.ones(math.ceil(task_num_batches_forget[0]), dtype=int)
        twos = np.ones(math.ceil(task_num_batches_forget[1]), dtype=int) * 2
        threes = np.ones(math.ceil(task_num_batches_forget[2]), dtype=int) * 3

        combined_forget = np.concatenate((ones, twos, threes))
        np.random.shuffle(combined_forget)

        task_num_batches_retain = [len(retain_data_list[topic_idx]) for topic_idx in range(len(retain_data_list))]
        ones = np.ones(math.ceil(task_num_batches_retain[0]), dtype=int)
        twos = np.ones(math.ceil(task_num_batches_retain[1]), dtype=int) * 2
        threes = np.ones(math.ceil(task_num_batches_retain[2]), dtype=int) * 3

        combined_retain = np.concatenate((ones, twos, threes))
        np.random.shuffle(combined_retain)

        combined_forget = combined_forget[:min(len(combined_forget), len(combined_retain))]
        combined_retain = combined_retain[:min(len(combined_forget), len(combined_retain))]

        control_vectors_list = []
        for i in range(len(forget_data_list)):
            random_vector = torch.rand(1,1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
            control_vec = random_vector / torch.norm(random_vector) * steering_coeff_list[i] # u * c
            control_vectors_list.append(control_vec)

        # length of control_vectors_list = number of forget corpora

        max_lengths = [256, 105, 335]
        
        truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side="right"

        for epoch in range(100):
            print(f"======= Epoch {epoch} =======")
            batch_idxs_forget = [0, 0, 0]
            batch_idxs_retain = [0, 0, 0]
            coeffs = {"0": 1.0, "1": 1.0, "2": 1.0}
            for idx in range(len(combined_forget)):
                topic_idx_forget = combined_forget[idx] - 1 # for 0 indexing
                topic_idx_retain = combined_retain[idx] - 1
                batch_idx_forget = batch_idxs_forget[topic_idx_forget]
                batch_idx_retain = batch_idxs_retain[topic_idx_retain]
                batch_idxs_forget[topic_idx_forget] += 1
                batch_idxs_retain[topic_idx_retain] += 1

                control_vec = control_vectors_list[topic_idx_forget]
                unlearn_batch = forget_data_list[topic_idx_forget][batch_idx_forget]
                retain_batch = retain_data_list[topic_idx_retain][batch_idx_retain]

                max_length_forget = max_lengths[topic_idx_forget]
                max_length_retain = max_lengths[topic_idx_retain]

                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length_forget
                ).to(updated_model.device)
                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec
                )

                # Adaptive coeff
                if batch_idx_forget == 0:
                    coeffs[str(int(topic_idx_forget))] = torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item() * scale

                print(coeffs)
                unlearn_loss = torch.nn.functional.mse_loss(
                    updated_forget_activations, control_vec * coeffs[f"{topic_idx_forget}"]
                )

                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length_retain
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                del retain_inputs

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                )
                retain_loss *= alpha[topic_idx_retain]

                loss = unlearn_loss + retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")
                
                del unlearn_inputs

            print("Proceeding to save model")

            tokenizer.truncation_side = truncation_side

            # Save model
            alpha_str = '_'.join(str(int(float(x))) for x in alpha)
            layer_ids_str = '_'.join(str(int(float(x))) for x in layer_ids)

            path = f"olmo_alpha_{alpha_str}_layer_{layer_id}_layerIds_{layer_ids_str}_scale_{int(scale)}_adaptive_rmu"
            final_path = os.path.join(output_dir, path)
            print("Trying to save model to path: ", final_path)
            updated_model.save_pretrained(final_path)
            tokenizer.save_pretrained(final_path)

            print(f"Saved model to {final_path}")

    forget_path = os.path.join(forget_dataset_path, 'forget.parquet') 
    retain_path = os.path.join(retain_dataset_path, 'retain.parquet')

    frozen_model, tokenizer = load_model(model_path)
    updated_model, tokenizer = load_model(model_path)

    num_model_layers = len(frozen_model.model.layers)
    print("num_model_layers: ", num_model_layers)
    if num_model_layers == 32:
        print("Using 7B model")
        batch_size = 12
        alpha = [900, 900, 900]
        scale = 5.0
        layer_ids = [24,25,26]
        layer_id = 26
        param_ids=[6]
    else:
        print("Using 1B model")
        batch_size = 8
        alpha = [900, 900, 900]
        scale = 3.0
        layer_ids = [12,13,14]
        layer_id = 14
        param_ids=[6]

    lr = 5e-5
    steering_coeff_list = [1, 1, 1]
    module_str = "{model_name}.model.layers[{layer_id}]"

    forget_df = pd.read_parquet(forget_path)
    retain_df = pd.read_parquet(retain_path)

    forget_df['text'] = forget_df['input'] + ' ' + forget_df['output']
    retain_df['text'] = retain_df['input'] + ' ' + retain_df['output']

    forget_df_task1 = forget_df[forget_df['task'] == 'Task1']
    forget_df_task2 = forget_df[forget_df['task'] == 'Task2']
    forget_df_task3 = forget_df[forget_df['task'] == 'Task3']

    retain_df_task1 = retain_df[retain_df['task'] == 'Task1']
    retain_df_task2 = retain_df[retain_df['task'] == 'Task2']
    retain_df_task3 = retain_df[retain_df['task'] == 'Task3']

    forget_dataset_task1 = Dataset.from_pandas(forget_df_task1)
    forget_dataset_task2 = Dataset.from_pandas(forget_df_task2)
    forget_dataset_task3 = Dataset.from_pandas(forget_df_task3)

    retain_dataset_task1 = Dataset.from_pandas(retain_df_task1)
    retain_dataset_task2 = Dataset.from_pandas(retain_df_task2)
    retain_dataset_task3 = Dataset.from_pandas(retain_df_task3)

    forget_dataset = [forget_dataset_task1, forget_dataset_task2, forget_dataset_task3]
    retain_dataset = [retain_dataset_task1, retain_dataset_task2, retain_dataset_task3]


    forget_data_list, retain_data_list = get_data(
        forget_dataset,
        retain_dataset,
        batch_size,
    )

    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        layer_ids,
        param_ids,
        module_str,
        layer_id,
        lr,
        steering_coeff_list,
        alpha,
        scale
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--forget_dataset_path", type=str, required=True)
    parser.add_argument("--retain_dataset_path", type=str, required=True)
    args = parser.parse_args()

    unlearn(
        args.model_path,
        args.output_dir,
        args.forget_dataset_path,
        args.retain_dataset_path
    )