import argparse
import os
import math

import torch
from syngen_diffusion_pipeline import SynGenDiffusionPipeline

from eval_data import data
from data_CC import CC
from tqdm import tqdm
import utils.vis_utils as vis_utils

import pandas as pd

def main(prompts, seeds, output_directory, model_path, step_size, attn_res, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, ours, lambda_ours, skip, sd):
    pipe = load_model(model_path, gpu)
    pipe.print_volumn = print_volumn
    pipe.excite = excite
    pipe.lambda_excite = lambda_excite
    pipe.sum_attn = sum_attn
    pipe.lambda_sum_attn = lambda_sum_attn
    pipe.dist = dist
    pipe.model2 = None
    pipe.ours = ours
    pipe.lambda_ours = lambda_ours
    pipe.skip = skip
    pipe.sd = sd
    
    if print_volumn:
        pipe.max_attn_value = []
        pipe.sum_volumn = []
    for prompt in tqdm(prompts):
        images = []
        for seed in seeds:
            print(f'Running on: "{prompt}"')
            seed = seed.item()
            image = generate(pipe, prompt, seed, step_size, attn_res)
            save_image(image, prompt, seed, output_directory+f'/{number}/'+prompt)
            images.append(image)

            if print_volumn:
                
                df = pd.DataFrame(pipe.max_attn_value)
                df.to_csv(output_directory+f'/{number}/'+prompt+'/max_attn_value.csv', index=False)
                # make a plot, and save it
                import matplotlib.pyplot as plt
                # only plot the first 7
                df = df.iloc[:, [3,8,12]]
                plt.figure()
                plt.plot(df)
                # add legend using pipe.labels
                plt.legend('camera dog tomato'.split(' '),loc='lower left', fontsize=20)
                plt.xlabel('Step', fontsize=15)
                plt.savefig(output_directory+f'/{number}/'+prompt+'/max_attn_value.png')
                plt.close()

                # plot the sum volumn
                # df = pd.DataFrame(pipe.sum_volumn)
                # df.to_csv(output_directory+f'/{number}/'+prompt+'/sum_volumn.csv', index=False)
                # # make a plot, and save it
                # import matplotlib.pyplot as plt
                # # only plot the first 7
                # df = df.iloc[:, [2,6]]
                # plt.figure()
                # plt.plot(df)
                # # add legend using pipe.labels
                # plt.legend('crown suitcase'.split(' '),loc='lower left')
                # plt.savefig(output_directory+f'/{number}/'+prompt+'/sum_volumn.png')

                vis_utils.show_cross_attention(prompt, pipe.attention_store, pipe.tokenizer, [7,9], attn_res, ['up', 'mid', 'down'], select=0, orig_image=image, save_path=output_directory+f'/{number}/'+prompt+'/cross_attention.png')

            
                        
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


def save_parameters_to_txt(seeds, dataset, reverse, gpu, number, print_volume, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, step_size, lambda_ours, file_name="parameters.txt"):
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
        'lambda_ours': lambda_ours
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
        default='./projects/Syntax-Guided-Generation/ours/'
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

    
    
    print_volumn = True
    sum_attn = False
    lambda_sum_attn = 0.5 if sum_attn else 0.0
    excite = False
    lambda_excite = 0.5 if excite else 0.0    
    syngen = False
    lambda_syngen = 0.5 if syngen else 0.0
    dist = 'cos'
    args.step_size = 20.0
    reverse = False

 
    


    

    for mode in ['ours','sd', 'sg']:

        # mode = 'ours'

        base_number = f'key_obs/add/{mode}/'
        
        number = f'{base_number}'


        seed_number = 12345
        torch.manual_seed(seed_number)
        seeds = torch.randint(0, 100000, (64,))[[0,3]]
         
        print(seeds)
        

        dataset = ['a blue zebra and a spotted crown']

    

        if mode == 'ours':
            ours = True
            lambda_ours = 0.5
            dist = 'cos'
            skip = True
            sd = False
        if mode == 'sg':
            dist = 'kl'
            ours = False
            skip = True
            sd = False
            lambda_ours = 0.0
        if mode == 'sd':
            ours = False
            lambda_ours = 0.0
            skip = True
            sd = True

        



        gpu = 1
        



        save_parameters_to_txt(seed_number, dataset, reverse, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, args.step_size , lambda_ours, file_name=f"{args.output_directory}/{number}/parameters.txt")

        
        
        main(dataset, seeds, args.output_directory, args.model_path, args.step_size, args.attn_res, gpu, number, print_volumn, excite, lambda_excite, sum_attn, lambda_sum_attn, dist, ours, lambda_ours, skip, sd)
  