import argparse
import os
import math

import torch
from syngen_diffusion_pipeline import SynGenDiffusionPipeline

from eval_data import data
from data_CC import CC
from tqdm import tqdm
import utils.vis_utils as vis_utils
from data_abc import abc

import pandas as pd

def main(prompts, seeds, output_directory, model_path, step_size, attn_res, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist):
    pipe = load_model(model_path, gpu)
    pipe.print_volumn = print_volumn
    pipe.excite = excite
    pipe.lambda_excite = lambda_excite
    pipe.sum_attn = sum_attn
    pipe.lambda_sum_attn = lambda_sum_attn
    pipe.dist = dist
    pipe.model2 = None
    pipe.skip = False
    pipe.ours = False
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
    result,_ = pipe(prompt=prompt, generator=generator, syngen_step_size=step_size, attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))))
    return result['images'][0]


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
        default='./projects/Syntax-Guided-Generation/output/stable-diffusion'
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
    seeds = torch.randint(0, 100000, (64,))


    
    print_volumn = False
    excite = False
    lambda_excite = 0.5 if excite else 0.0      
    sum_attn = False
    lambda_sum_attn = 0.5 if sum_attn else 0.0
    dist = 'kl'


    reverse = False
    gpu = 2


    # df = pd.read_csv('destination.csv')
    # dataset = df['prompt'].tolist()
    number = 'pipeline_sd'
    dataset = ['a purple crown and a blue suitcase']
    seeds = seeds[[2,3,4,9]]





    save_parameters_to_txt(seed_number, dataset, reverse, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, args.step_size , file_name=f"{args.output_directory}/{number}/parameters.txt")


    main(dataset[::-1 if reverse else 1], seeds, args.output_directory, args.model_path, args.step_size, args.attn_res, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist)


    # from PARTI dataset
    # dataset = pd.read_csv('destination.csv')
    # dataset = dataset['prompt'].tolist()
    

    
