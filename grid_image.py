import argparse
import os
import math

import torch
from syngen_diffusion_pipeline import SynGenDiffusionPipeline
from eval_data import data
from tqdm import tqdm
import utils.vis_utils as vis_utils

# import Image
from PIL import Image

from prompt_name import SEEDS, PROMPTS


# prompts = data['animals_objects']

# Objects
output_directory1 = "/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/output/7"
output_directory2 = '/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/ours/lambda0.5/cos'
output_directory3 = '/home/yasmin/projects/Excite/outputs/' 
output_directory4 = '/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/output/stable-diffusion/pipeline_sd/'


# # ABC
output_directory2 = "/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/output/syn_abc_sample/"
output_directory1 = '/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/ours/abc_sample/lambda0.5'
output_directory3 = '/home/yasmin/projects/Excite/outputs/abc_sample/' 
output_directory4 = '/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/output/stable-diffusion/pipeline_sd/abc_sample'
output_directorys = [output_directory1, output_directory2, output_directory3, output_directory4]

# # DVMP
# output_directory2 = "/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/output/syn_dvmp_sample/"
# output_directory1 = '/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/ours/dvmp_sample/lambda0.5'
# output_directory4 = '/home/yasmin/projects/Excite/outputs/dvmp_sample/' 
# output_directory3 = '/home/yasmin/projects/Syntax-Guided-Generation/projects/Syntax-Guided-Generation/output/stable-diffusion/pipeline_sd/dvmp_samples'
output_directorys = [ output_directory4]


prompts = PROMPTS


seed_number = 12345
torch.manual_seed(seed_number)
seeds = torch.randint(0, 100000, (64,))[SEEDS]  

temp = [seed.item() for seed in seeds]

temp = ['27469', '75121']
# numbers = [3,4,5,7]
# temp = [seeds[number-1].item() for number in numbers]
# temp = ['27469', '74267', '67032', '64555'] # for 
# # temp = ['5494', '98725', '83553', '71155'] # 


    
methods = ['ours', 'sg', 'excite', 'sd']
for kk, output_directory in enumerate(output_directorys):
     
    for prompt in tqdm(prompts):
        images = []
        image_paths = [output_directory + '/' + prompt + '/' + f'{t}.png' for t in temp]
        for image_path in image_paths:
            images.append(Image.open(image_path))

        joined_image = vis_utils.get_image_grid(images)
        joined_image.save('add_qual/'+methods[kk]+f'_{prompt}.png')

