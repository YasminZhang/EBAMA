import argparse
import os
import math

import torch
from syngen_diffusion_pipeline import SynGenDiffusionPipeline

from eval_data import data
from data_CC import CC
from tqdm import tqdm
import utils.vis_utils as vis_utils

def main(prompts, seeds, output_directory, model_path, step_size, attn_res, gpu, number, print_volumn, excite, lambda_excite):
    pipe = load_model(model_path, gpu)
    pipe.print_volumn = print_volumn
    pipe.excite = excite
    pipe.lambda_excite = lambda_excite
    for prompt in tqdm(prompts):
        images = []
        for seed in seeds:
            print(f'Running on: "{prompt}"')
            seed = seed.item()
            image = generate(pipe, prompt, seed, step_size, attn_res)
            save_image(image, prompt, seed, output_directory+f'/{number}/'+prompt)
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
    result = pipe(prompt=prompt, generator=generator, syngen_step_size=step_size, attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))))
    return result['images'][0]


def save_image(image, prompt, seed, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{seed}.png"
    image.save(file_name)


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

    torch.manual_seed(12345)
    seeds = torch.randint(0, 100000, (4,))
    reverse = False
    gpu = 1
    number = 12
    print_volumn = False
    excite = True
    lambda_excite = 0.1

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

    

    main(sentences[::-1 if reverse else 1], seeds, args.output_directory, args.model_path, args.step_size, args.attn_res, gpu, number, print_volumn, excite, lambda_excite)
