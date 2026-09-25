import ast
from pathlib import Path

def named_definitions(path, names, namespace):
    """Load selected function definitions from a Python source file.

    Optionally remove function decorators before compilation.
    """
    parsed = ast.parse(Path(path).read_text())
    nodes = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'missing official definitions: {path}')
    for node in nodes:
        node.decorator_list = [d for d in node.decorator_list
                               if not ('register_module' in ast.unparse(d))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def sdar_official(source):
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache
    from torch.nn import functional as F
    module = source/'evaluation/opencompass/opencompass/models/huggingface_bd3.py'
    names = ['add_gumbel_noise', 'top_k_logits', 'top_p_logits',
             'sample_with_temperature_topk_topp', 'get_num_transfer_tokens', 'block_diffusion_generate']
    return named_definitions(module, names, {'torch': torch, 'np': np, 'F': F, 'DynamicCache': DynamicCache})
