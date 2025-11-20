import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from utils_hh.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask
# Add more algo
from utils_hh.modify_llama_oursk import LlamaAttention_heavy_hitter_our_sketch

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
    "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}

TAGET_MODULE = {
    "llama": [LlamaAttention_heavy_hitter, LlamaAttention_heavy_hitter_our_sketch],
    "opt": OPTAttention_Mask,
    "gpt_neox": GPTNeoXAttention_Mask,
}

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length
    elif length < 0:
        length = MAX_LENGTH
    return length


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument('--model_arch', type=str, default='opt')
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint/")

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)
    parser.add_argument('--enable_small_cache', action='store_true')

    parser.add_argument('--enable_numerical_stability', action='store_true')
    parser.add_argument('--epsilon_times', type=float, default=1.0)

    parser.add_argument("--sketch_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--sketch_prefill_similarity_threshold", type=float, default=0.8)

    parser.add_argument("--pyramidinfer_recent_ratio", type=float, default=0.2)
    parser.add_argument("--pyramidinfer_decay_strategy", type=str, default='cosine')
    parser.add_argument("--pyramidinfer_decay_ratio", type=float, default=0.8)
    parser.add_argument("--pyramidinfer_distance_weight", type=float, default=1.2)
    parser.add_argument("--pyramidinfer_sink_len", type=int, default=4)
    parser.add_argument("--hh_method", type=str, default='h2o')

    parser.add_argument("--sample_num", type=int, default=1000)

    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)

    model_name = args.model_name
    input_path = args.input_path
    output_path = args.output_path 

    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir)

    if args.enable_small_cache:
        print('Enable Small Cache Size')
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio

        config.enable_numerical_stability  = args.enable_numerical_stability
        config.epsilon_times = args.epsilon_times
        config.sketch_similarity_threshold = args.sketch_similarity_threshold
        config.sketch_prefill_similarity_threshold = args.sketch_prefill_similarity_threshold

        config.pyramidinfer_recent_ratio = args.pyramidinfer_recent_ratio 
        config.pyramidinfer_decay_strategy = args.pyramidinfer_decay_strategy 
        config.pyramidinfer_decay_ratio = args.pyramidinfer_decay_ratio 
        config.pyramidinfer_distance_weight = args.pyramidinfer_distance_weight 
        config.pyramidinfer_sink_len = args.pyramidinfer_sink_len 

        config.hh_method = args.hh_method
        checkpoint = copy.deepcopy(model.state_dict())
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_arch](model, config)
        model.load_state_dict(checkpoint)

    model.half().eval().cuda()
    logger.info(args)

    requests = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip() != '':
                requests.append(json.loads(line))

    print(len(requests))
    if args.sample_num < len(requests):
        print('Sample {} Examples'.format(args.sample_num))
    requests = requests[:args.sample_num]

    results = []
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            request = request['request']
            result = {'request': request, 'result': {}}
            prompt = request['prompt']
            temperature = request['temperature']
            stop = request['stop']

            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                temperature=temperature,
                top_k=args.k,
                top_p=request['top_p'],
                do_sample=True,
                num_return_sequences=request['n'],
                return_dict_in_generate=True, output_scores=True,
            )

            for name, m in model.named_modules():
                for m_inst in TAGET_MODULE[args.model_arch]:
                    if isinstance(m, m_inst):
                        m._reset_masks()

            tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
            logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
            top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

            generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
            generate_text = generate_text[: generate_text.find(stop[0])]

            result['result'] = {
                "choices": [
                    {
                        "text": generate_text,
                        "logprobs": {
                            "tokens": tokens, 
                            "token_logprobs": logprobs, 
                            "top_logprobs": top_logprobs, 
                            "text_offset": []
                        }, 
                        "finish_reason": "length"
                    }
                ], 
                "request_time": {
                    "batch_time": 0, 
                    "batch_size": 1}
            }
            
            results.append(result)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == "__main__":
    main()

