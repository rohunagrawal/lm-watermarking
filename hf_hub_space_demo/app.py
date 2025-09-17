# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models" 
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
args = Namespace()

arg_dict = {
    'run_gradio': True, 
    'demo_public': False,  
    # 'model_name_or_path': 'facebook/opt-1.3b', # historical
    # 'model_name_or_path': 'facebook/opt-2.7b', # historical
    # 'model_name_or_path': 'facebook/opt-6.7b', # historical
    # 'model_name_or_path': 'meta-llama/Llama-2-7b-hf', # historical
    'model_name_or_path': 'meta-llama/Llama-3.2-3B',
    'all_models':[
        # "meta-llama/Llama-3.1-8B", # too big for the A10G 24GB
        "meta-llama/Llama-3.2-3B",
        # "meta-llama/Llama-3.2-1B",
        # "Qwen/Qwen3-8B", # too big for the A10G 24GB
        # "Qwen/Qwen3-4B",
        # "Qwen/Qwen3-1.7B",
        # "Qwen/Qwen3-0.6B",
        # "Qwen/Qwen3-4B-Instruct-2507",
        # "Qwen/Qwen3-4B-Thinking-2507",
    ],
    # 'load_fp16' : True,
    'load_fp16' : False,
    'load_bf16' : True,
    'prompt_max_length': None, 
    'max_new_tokens': 200, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.5,
    'delta': 2.0, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
}

args.__dict__.update(arg_dict)

from demo_watermark import main

main(args)
