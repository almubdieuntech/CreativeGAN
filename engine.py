import argparse
import os
import typing
from typing import Union, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from baseline import generate, MAXIMUM_RES
from modules.models import CGRModel



def config_model(model_path: Union[str, os.PathLike],
                 device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu',
                 nsfw_allowed: Optional[bool] = True):

    data_type = torch.float32
    model = CGRModel.load_static_model(model_path, torch_dtype=data_type).to(device)
    if nsfw_allowed:
        model.safety_checker.to('cpu')

    return model


def main(model: Optional[CGRModel], prompts: Union[str, List[str], os.PathLike], size: Optional[typing.Tuple],
         using_step: Optional[bool] = True, nsfw_allowed: Optional[bool] = True, out_dir: Optional[str] = 'out',
         step: Optional[int] = 2):
    kwargs = dict(use_version=True, version='v4', use_realistic=False, size=size, nsfw_allowed=nsfw_allowed,
                  out_dir=out_dir)

    if isinstance(prompts, str) and not prompts.endswith('.txt'):

        _ = [f for f in generate(prompt=prompts, model=model, **kwargs)]

    elif isinstance(prompts, list) or prompts.endswith('.txt'):

        if prompts.endswith('.txt'):
            us_a: bool = True
            prompts = open(prompts, 'r', encoding='utf8').readlines()
        else:
            us_a: bool = False
        if using_step:
            prompts = np.array(prompts)
            bl = len(prompts) % step if len(prompts) > step else step - len(prompts)
            if us_a:
                bl += 2
            if len(prompts) % step != 0:
                v = np.array(['<BREAK>' for _ in range(bl)])
                prompts = np.concatenate((prompts, v), axis=0)
            prompts = prompts.reshape((len(prompts) // step if len(prompts) // step != 0 else 1, step))
            for i, prp in tqdm(enumerate(prompts)):
                # print(prp)
                if '<BREAK>' in prp:
                    fx = 0
                    for ix, p in enumerate(prp):
                        if p == '<BREAK>':
                            fx = ix
                            break
                    prp = prp[:fx]
                _ = [f for f in generate(prompt=prp.tolist(), model=model, **kwargs)]
        else:
            _ = [f for f in generate(prompt=prompts, model=model, **kwargs)]
