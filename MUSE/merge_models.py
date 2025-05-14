import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import copy
from get_info import get_components
import argparse
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
import os
import json
import yaml


def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("./config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

class CustomModelForCausalLM(PreTrainedModel):
    def __init__(self, model_name_or_path, model_family='phi', pretrained_model_name_or_path=None, gamma=1.0, logsoftmax=True, model_cfg=None, cfg=None, minus_value=None, **kwargs):
        
        if os.path.exists(os.path.join(pretrained_model_name_or_path, 'config.json')):
            model_cfg_name = os.path.join(pretrained_model_name_or_path, 'config.json')
            with open(model_cfg_name, "r") as f:
                model_cfg = yaml.load(f, Loader=yaml.FullLoader)
        else:
            print(model_family)
            model_cfg = get_model_identifiers_from_yaml(model_family)
        print(model_cfg)
        print('Loading Customized Models')
        # Load configuration for the main model
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            use_flash_attention_2=model_cfg.get("flash_attention2", "false") == "true",
            trust_remote_code=True,
        )
        super().__init__(config)
        
        # Initialize attributes
        self.gamma = gamma
        self.minus_value = minus_value

        # self.gamma = -3.0
        # print('Gamma Changed')

        self.logsoftmax = logsoftmax
  
        self.model = None
        self.pretrained_model = None

        # Load main and pretrained models if paths are provided
        if model_name_or_path:
            self.model = self._load_model(model_name_or_path, model_cfg, **kwargs)
        if pretrained_model_name_or_path:
            self.pretrained_model = self._load_model(pretrained_model_name_or_path, model_cfg, **kwargs)

    def _load_model(self, model_path, model_cfg, **kwargs):
        """
        Helper function to load a model from a given path with the provided configuration.
        """
        config = AutoConfig.from_pretrained(
            model_path,
            use_flash_attention_2=model_cfg.get("flash_attention2", "false") == "true",
            trust_remote_code=True,
        )
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        kwargs.setdefault("trust_remote_code", True)
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            use_flash_attention_2=model_cfg.get("flash_attention2", "false") == "true",
            **kwargs
        )

    def save_pretrained(self, save_directory):
        """
        Save the custom model with its components in a directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save main model and pretrained model to separate subdirectories
        model_dir = os.path.join(save_directory, "model")
        pretrained_model_dir = os.path.join(save_directory, "pretrained_model")

        if self.model:
            self.model.save_pretrained(model_dir)
        if self.pretrained_model:
            self.pretrained_model.save_pretrained(pretrained_model_dir)

        # Save additional configuration
        custom_config = {
            "gamma": self.gamma,
            "logsoftmax": self.logsoftmax,
            "model_dir": "model",
            "pretrained_model_dir": "pretrained_model"
        }
        with open(os.path.join(save_directory, "custom_config.json"), "w") as f:
            json.dump(custom_config, f, indent=4)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load the custom model from a saved directory, including main and pretrained components.
        """
        print('Loading Customized Models')
        # Load the base configuration
        # config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Load the custom configuration file for additional settings
        custom_config_path = os.path.join(pretrained_model_name_or_path, "custom_config.json")
        with open(custom_config_path, "r") as f:
            custom_config = json.load(f)

        # Retrieve gamma and logsoftmax, and load subdirectory paths for the models
        gamma = custom_config.get("gamma", 1.0)
        logsoftmax = custom_config.get("logsoftmax", True)
        minus_value = custom_config.get("minus_value", None)

        model_dir = os.path.join(pretrained_model_name_or_path, custom_config["model_dir"])
        pretrained_model_dir = os.path.join(pretrained_model_name_or_path, custom_config["pretrained_model_dir"])
        # Initialize the custom model instance with loaded configuration
        model = cls(model_name_or_path=model_dir, pretrained_model_name_or_path=pretrained_model_dir, gamma=gamma, logsoftmax=logsoftmax, minus_value=minus_value, **kwargs)
        return model

    def forward(self, *args, **kwargs):
        if self.pretrained_model is None:
            raise ValueError("The pretrained model is not loaded or is set to None.")
        
        output = self.model(*args, **kwargs)
        
        if self.pretrained_model is not None:
            outputs1 = self.pretrained_model(*args, **kwargs)
            
        else:
            outputs1 = output

        with torch.no_grad():
            logits0 = output.logits[:, :, :].float()
            logits1 = outputs1.logits[:, :, :].float()
            if self.gamma < 0:
                logits0, logits1 = logits1, logits0
            if self.logsoftmax:
                logits0 = F.log_softmax(logits0, dim=-1)
                logits1 = F.log_softmax(logits1, dim=-1)
                logits = (1 - abs(self.gamma)) * logits1 + abs(self.gamma) * logits0
                logits = torch.exp(logits)
            else:
                logits = (1 - abs(self.gamma)) * logits1 + abs(self.gamma) * logits0
            output.logits[:, :, :] = logits
        return output


    def contrasting_generation(self, input_ids, attention_mask=None, max_length=0, max_new_tokens=None, do_sample=True, use_cache=True, pad_token_id=None):
        gen_num = input_ids.shape[0]
        print('Sample', do_sample)
        max_token = max_length

        params, model_kwargs = get_components(self.model,
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id
        )
        if self.pretrained_model is not None:
            params1, model_kwargs1 = get_components(self.pretrained_model,
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id
            )

        logits_processor, stopping_criteria, generation_config, synced_gpus, streamer = params
        update_flag = torch.ones(gen_num, dtype=torch.bool).cuda()
        input_ids = copy.deepcopy(input_ids)
        model_kwargs = self.model._get_initial_cache_position(input_ids, model_kwargs)
        if self.pretrained_model is not None:
            model_kwargs1 = self.pretrained_model._get_initial_cache_position(input_ids, model_kwargs1)
        torch.manual_seed(0)
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        max_length = generation_config.max_length
        output_scores = generation_config.output_scores
        return_dict_in_generate = generation_config.return_dict_in_generate
        scores = () if (return_dict_in_generate and output_scores) else None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        while self.model._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )
            if self.pretrained_model is not None:
                model_inputs_pre = self.pretrained_model.prepare_inputs_for_generation(
                    input_ids, **model_kwargs1
                )

            outputs0 = self.model(**model_inputs, return_dict=True)
            logits0 = outputs0.logits[:, -1, :].float()
            if self.pretrained_model is not None:
                outputs1 = self.pretrained_model(**model_inputs_pre, return_dict=True)
                logits1 = outputs1.logits[:, -1, :].float()
            else:
                outputs1 = outputs0
                logits1 = outputs1.logits[:, -1, :].float()
            if self.gamma < 0:
                logits0, logits1 = logits1, logits0
            if self.logsoftmax:
                logits0 = F.log_softmax(logits0, dim=-1)
                logits1 = F.log_softmax(logits1, dim=-1)
                logits = (1 - abs(self.gamma)) * logits1 + abs(self.gamma) * logits0
                logits = torch.exp(logits)
            else:
                logits = (1 - abs(self.gamma)) * logits1 + abs(self.gamma) * logits0

            next_token_logits = logits

            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                if self.logsoftmax:
                    probs = next_token_scores
                else:
                    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                if self.minus_value is None:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                else:
                    max_logits0 = torch.max(logits0, dim=-1)[0]
                    mask = logits0 > (max_logits0.unsqueeze(-1)-self.minus_value)
                    logits_masked = logits.masked_fill(~mask, float('-inf'))  # 使用 -inf 填充不符合条件的位置
                    next_tokens = torch.argmax(logits_masked, dim=-1).cuda()
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs0,
                model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
            )
            if self.pretrained_model is not None:
                model_kwargs1 = self.pretrained_model._update_model_kwargs_for_generation(
                    outputs1,
                    model_kwargs1,
                    is_encoder_decoder=self.pretrained_model.config.is_encoder_decoder,
                )
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs0, outputs1

        return input_ids

    def generate(self, input_ids, attention_mask=None, max_length=None, max_new_tokens=0, do_sample=False, use_cache=True, pad_token_id=None, **kwargs):
        return self.contrasting_generation(input_ids, attention_mask, max_length, max_new_tokens, do_sample, use_cache, pad_token_id, **kwargs)


# def main():
#     parser = argparse.ArgumentParser(description="Generate custom model with contrasting generation settings")
#     parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the main model")
#     parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to the pretrained model to compare with")
#     parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value for contrasting generation")
#     parser.add_argument("--logsoftmax", action="store_true", help="Use log softmax for logits if flag is present")
#     parser.add_argument("--output_path", type=str, required=True, help="Directory to save the custom model")

#     args = parser.parse_args()


#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#     custom_model = CustomModelForCausalLM(
#         model_name_or_path=args.model_name_or_path,
#         pretrained_model_name_or_path=args.pretrained_model_name_or_path,
#         gamma=args.gamma,
#         logsoftmax=args.logsoftmax
#     )


#     # Define the subdirectory name using gamma and logsoftmax values
#     gamma_prefix = "neg" if args.gamma < 0 else "pos"
#     gamma_value = str(abs(args.gamma)).replace(".", "_") 
#     logsoftmax_str = "logsoftmax" if args.logsoftmax else "no_logsoftmax"

#     subdirectory_name = f"{gamma_prefix}_gamma_{gamma_value}_{logsoftmax_str}"[:96].strip("-.")
#     subdirectory = os.path.join(args.output_path, subdirectory_name)


#     os.makedirs(subdirectory, exist_ok=True)

#     # Save the model and tokenizer in the specified subdirectory
#     custom_model.save_pretrained(subdirectory)
#     tokenizer.save_pretrained(subdirectory)

#     # Save the arguments to a JSON file in the same subdirectory
#     args_dict = vars(args)  # Convert args to a dictionary
#     with open(os.path.join(subdirectory, "merging_config.json"), "w") as f:
#         json.dump(args_dict, f, indent=4)


# if __name__ == "__main__":
#     main()
 
