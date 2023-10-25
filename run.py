import argparse
import os
import math

import torch
from syngen_diffusion_pipeline import SynGenDiffusionPipeline
from diffusers import StableDiffusionPipeline
from typing import Optional, Union, Tuple, List, Callable, Dict

from eval_data import data
from data_CC import CC
from tqdm import tqdm
import utils.vis_utils as vis_utils
import abc
import ptp_utils
import seq_aligner
import torch.nn.functional as nnf

LOW_RESOURCE = False
MY_TOKEN = 'S'
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

# import sys
# # insert it to the first place in the path list
# sys.path.insert(0, '/home/yasmin/projects/Syntax-Guided-Generation')
class LocalBlend:
    pass


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t): # What's the point of this function?
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        if self.cur_step == 51:
            self.cur_step = 0
            self.attention_store = {}
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:] # two or more prompts
            if is_cross:
        
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], tokenizer = None, device = None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
   

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer = None, device = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer, device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

# class AttentionRefine(AttentionControlEdit):

#     def replace_cross_attention(self, attn_base, att_replace):
#         attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
#         attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
#         return attn_replace

#     def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
#                  local_blend: Optional[LocalBlend] = None, tokenizer = None, device = None):
#         super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
#         self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
#         self.mapper, alphas = self.mapper.to(device), alphas.to(device)
#         self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


# class AttentionReweight(AttentionControlEdit):

#     def replace_cross_attention(self, attn_base, att_replace):
#         if self.prev_controller is not None:
#             attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
#         attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
#         return attn_replace

#     def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
#                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
#         super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
#         self.equalizer = equalizer.to(device)
#         self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer




def main(prompts, seeds, output_directory, model_path, step_size, attn_res, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, model2, dataset2):
    pipe = load_model(model_path, gpu)
    pipe.print_volumn = print_volumn
    pipe.excite = excite
    pipe.lambda_excite = lambda_excite
    pipe.sum_attn = sum_attn
    pipe.lambda_sum_attn = lambda_sum_attn
    pipe.dist = dist
    pipe.skip = False
    if model2:
        ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token='S').to(f'cuda:{gpu}')
        extra_set_kwargs = {}
        ldm_stable.scheduler.set_timesteps(50, **extra_set_kwargs)
        tokenizer = ldm_stable.tokenizer
        controller = AttentionReplace(['a green balloon', 'a red balloon'], NUM_DIFFUSION_STEPS, cross_replace_steps=.8, self_replace_steps=0.4, tokenizer=tokenizer, device=f'cuda:{gpu}')
        ptp_utils.register_attention_control(ldm_stable, controller)
        pipe.model2 = ldm_stable

    else:
        pipe.model2 = None
        

    for prompt in tqdm(prompts):
        images = []
        pipe.prompt2 = dataset2[0] # TODO
        for seed in seeds:
            print(f'Running on: "{prompt}"')
            seed = seed.item()
            image, image2 = generate(pipe, prompt, seed, step_size, attn_res)
            save_image(image, prompt, seed, output_directory+f'/{number}/'+prompt)
            save_image(image2, prompt, seed, output_directory+f'/{number}/'+dataset2[0])
            images.append(image)

        joined_image = vis_utils.get_image_grid(images)
        joined_image.save(output_directory+f'/{number}/'+f'{prompt}.png')



def load_model(model_path, device=0):
    device = torch.device(f'cuda:{device}') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SynGenDiffusionPipeline.from_pretrained(model_path).to(device)

    return pipe


def generate(pipe, prompt, seed, step_size, attn_res):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    generator = torch.Generator(device.type).manual_seed(seed)
    result, result2 = pipe(prompt=prompt, generator=generator, syngen_step_size=step_size, attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))))
    return result['images'][0], result2['images'][0]


def save_image(image, prompt, seed, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{seed}.png"
    image.save(file_name)


def save_parameters_to_txt(seeds, dataset, reverse, gpu, number, print_volume, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, step_size, file_name="parameters.txt"):
    # Create a dictionary to store the parameters
    parameters = {
        "seeds": seeds,
        "dataset": dataset,
        "reverse": reverse,
        "gpu": gpu,
        "number": number,
        "print_volume": print_volume,
        "excite": excite,
        "lambda_excite": lambda_excite,
        "sum_attn": sum_attn,
        "lambda_sum_attn": lambda_sum_attn,
        "dist": dist,
        'step_size': step_size,
    }

    # create the output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    # Open the file in write mode and write the parameters
    with open(file_name, 'w') as file:
        for key, value in parameters.items():
            if isinstance(value, bool):
                value = str(value).lower()  # Convert bool to lowercase string
            if isinstance(value, list):
                # only store the value's length if it's a list
                value = len(value)
            line = f"{key}: {value}\n"
            file.write(line)

    print(f"Parameters saved to {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a checkered bowl on a red and blue table"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=12345
    )

    parser.add_argument(
        '--output_directory',
        type=str,
        default='./projects/Syntax-Guided-Generation/output'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='The path to the model (this will download the model if the path doesn\'t exist)'
    )

    parser.add_argument(
        '--step_size',
        type=float,
        default=20.0,
        help='The SynGen step size'
    )

    parser.add_argument(
        '--attn_res',
        type=int,
        default=256,
        help='The attention resolution (use 256 for SD 1.4, 576 for SD 2.1)'
    )


    args = parser.parse_args()

    seed_number = 12345
    torch.manual_seed(12345)
    seeds = torch.randint(0, 100000, (4,))
    dataset = data['objects']
    reverse = True
    gpu = 2
    number = 'test_ptp'
    print_volumn = False
    excite = False
    lambda_excite = 0.5 if excite else 0.0      
    sum_attn = False
    lambda_sum_attn = 0.5 if sum_attn else 0.0
    dist = 'kl'
    args.step_size = 20.0
    model2 = True



    save_parameters_to_txt(seed_number, dataset, reverse, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, args.step_size , file_name=f"{args.output_directory}/{number}/parameters.txt")



    sentences = [
    "a pink sunflower and a yellow flamingo",
    "a checkered bowl in a cluttered room",
    "a horned lion and a spotted monkey",
    "a brown brush glides through beautiful blue hair",
    "a blue and white dog sleeps in front of a black door",
    "a white fire hydrant sitting in a field next to a red building",
    "a wooden crown and a furry baby rabbit",
    "a red chair and a purple camera and a baby lion",
    "a spiky bowl and a green cat"
    ]

    dataset = ['a green balloon']
    dataset2 = ['a red balloon']

    

    main(dataset[::-1 if reverse else 1], seeds, args.output_directory, args.model_path, args.step_size, args.attn_res, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, model2, dataset2)
