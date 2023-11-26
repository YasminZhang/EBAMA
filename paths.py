

methods = ['ours', 'lambda0', 'sg', 'excite', 'sd']

datasets = ['AnE', 'abc', 'dvmp']

BIG_FOLDERS = {dataset: [f'/home/yasmin/projects/Syntax-Guided-Generation/final_data/{dataset}/{method}/' for method in methods] for dataset in datasets}



from eval_data import data

ane_dataset = data['animals_objects'] + data['objects']

# select 100 prompts from ane_dataset
import random
random.seed(12345)
random.shuffle(ane_dataset)
ane_dataset = ane_dataset[:100]

from data_abc import abc

abc_dataset = abc[:100]

# read dvmp1.csv
import pandas as pd
df = pd.read_csv('dvmp1.csv')
dvmp_dataset1 = df['prompt'].tolist()[-33:]

# read dvmp2.csv
df = pd.read_csv('dvmp2.csv')
dvmp_dataset2 = df['prompt'].tolist()[-33:]

# read dvmp3.csv
df = pd.read_csv('dvmp3.csv')
dvmp_dataset3 = df['prompt'].tolist()[-34:]


dvmp_dataset = dvmp_dataset1 + dvmp_dataset2 + dvmp_dataset3

BIG_PROMPTS = {'AnE': ane_dataset, 'abc': abc_dataset, 'dvmp': dvmp_dataset}

if __name__ == '__main__':
    # combine all the images in dvmp foler into one image
    from PIL import Image
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from utils.vis_utils import get_image_grid

    for name in [ 'dvmp']:
        p = BIG_PROMPTS[name]
        f = BIG_FOLDERS[name]
        for k, p_ in enumerate(p):
            print(k+1, p_)
             
        # for idx, method in enumerate(f):
        #     images = []
        #     for prompt in p:
        #         image = Image.open(os.path.join(method, prompt+'/70690.png'))
        #         images.append(image)
        #     images = get_image_grid(images)
        #     if not os.path.exists(f'./images_all/{name}'):
        #         os.makedirs(f'./images_all/{name}', exist_ok=True)
        #     images.save(f'./images_all/{name}/{methods[idx]}.png')
             