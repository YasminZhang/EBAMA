import gradio as gr
import os
import random
import numpy as np
from PIL import Image
from eval_data import data
from paths import BIG_FOLDERS, BIG_PROMPTS 

import numpy as np


# Function to load images from each folder
def load_images(prompt):
    images = []
    # Ensure each folder has at least num_images images
    for folder in folders[problem_id-1]:
        # img_path = folder + prompt + '/' +  random.choice(os.listdir(folder+prompt)) 
        img_path = folder + prompt + '/' +  '70690.png'
        img = Image.open(img_path)  # Open the image file
        img_array = np.array(img)
        images.append(img_array)

    return images

# Function to create a multiple choice problem
def create_problem():

   
    prompt = prompts[problem_id-1]

    images = load_images(prompt)
    

    change_order = random.choice([True, False])
    if change_order:
        images = images[::-1]
        d = {'Image 1': method_list[problem_id-1], 'Image 2': 0}
    else:
        d = {'Image 1': 0, 'Image 2': method_list[problem_id-1]}

                 
    
    
    def record_choice(choice1, choice2):
         
    
        if not os.path.exists(f'./record/{user_name}'):
            os.makedirs(f'./record/{user_name}', exist_ok=True)

        global problem_id

        if problem_id == 1:
            with open(f'./record/{user_name}/{user_name}.txt', 'w') as f:
                f.write('choice1,choice2\n')
                f.write(f"{d[choice1]},{d[choice2]}\n")
        else:
            with open(f'./record/{user_name}/{user_name}.txt', 'a') as f:
                f.write(f"{d[choice1]},{d[choice2]}\n")


        return f"You selected image: {choice1, choice2}"
    
    def make_result_image():
         
        import pandas as pd
        import matplotlib.pyplot as plt
        result = pd.read_table(f'./record/{user_name}/{user_name}.txt', header=0, sep=',')
        # result.columns = ['choice1', 'choice2']
        total_problems = result.shape[0]
        fig = plt.figure(figsize=(10, 10))
        # make a bar plot using choice1
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.bar(0, (result['choice1']==0).sum()/total_problems)
        for i in range(1, 5):
                ax1.bar(i, (result['choice1']==i).sum()/(np.array(method_list)==i).sum())
        # add a y = 0.5 line
        ax1.plot([-1, 5], [0.5, 0.5], 'k--', alpha=0.5)
        ax1.set_title('Which image matches better the given description?')
        ax1.set_xticks([0, 1, 2, 3, 4])
        ax1.set_xticklabels(['ours', 'ours(lambda=0)', 'SG', 'AnE', 'SD'])
        ax1.set_ylim([0, 1])

        
        # make a bar plot using choice2
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.bar(0, (result['choice2']==0).sum()/total_problems)
        for i in range(1, 5):
                ax2.bar(i, (result['choice2']==i).sum()/(np.array(method_list)==i).sum())
        ax2.plot([-1, 5], [0.5, 0.5], 'k--', alpha=0.5)
        ax2.set_title('Which image looks overall better or more natural?')
        ax2.set_xticks([0, 1, 2, 3, 4])
        ax2.set_xticklabels(['ours', 'ours(lambda=0)', 'SG', 'AnE', 'SD'])
        ax2.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'./record/{user_name}/{user_name}.png')
        

    # Create the Gradio interface
    def create_interface():

        def on_submit(choice1, choice2):

           

            

            if (choice1 not in ['Image 1', 'Image 2']) or (choice2 not in ['Image 1', 'Image 2']):

                return {error_box: gr.Markdown(value="##  <span style=\"color:red\">{Error: Please select an image!}</span> ", visible=True) }
            

            else:

                global problem_id

                if problem_id <= total_problems:

                    
                    
                    record_choice(choice1, choice2)
                    
                
                if problem_id < total_problems:
                    problem_id += 1

                    prompt = prompts[problem_id-1]
                    images = load_images(prompt)

                    change_order = random.choice([True, False])
                    nonlocal d
                    if change_order:
                        images = images[::-1]
                        d = {'Image 1': method_list[problem_id-1], 'Image 2': 0}
                    else:
                        d = {'Image 1': 0, 'Image 2': method_list[problem_id-1]}

                    

    

                    # Update the interface with the new images
                    return {image1: images[0], image2: images[1], text1: f"### Description: <span style=\"color:red\">{prompt}</span>", text2: f"### Progress:  {problem_id}/{total_problems}", error_box: gr.Markdown(value="## \textcolor{Error}", visible=False), radio1: None, radio2: None}
                
                elif problem_id == total_problems:
                    problem_id += 1

                    prompt = 'None'
                    images = np.zeros((4, 200, 200, 3))

                    make_result_image()

                    
                    # result_image = Image.open(f'./record/{user_name}.png')
                    # result_image = np.array(result_image)
                    # w, h = result_image.shape[0], result_image.shape[1]
                

                    return {image1: images[0], image2: images[1],  text1: f"### Description: <span style=\"color:red\">{prompt}</span>", \
                            text2: f"### Progress:  Finished", ending: "### Thank you for participating in our user study! Please close this window to exit.", \
                            button: 'Thank you!', error_box: gr.Markdown(value="## \textcolor{Error}", visible=False), \
                            }
                else:
                    problem_id += 1
                    return {button: 'Thank you!'}
                


            

        

        with gr.Blocks(theme=gr.themes.Soft()) as demo:

            gr.Markdown(f"# User Study")

            gr.Markdown(f"## Your unqiue user ID is: {user_name}")

            

            text2 = gr.Markdown(f"### Progress:  {problem_id}/{total_problems}")

            ending = gr.Markdown(f"### Please complete the two questions below and click the 'Next' button to continue to the next problem.")

    

            # make images gr.Examples
            with gr.Row():
                image1 = gr.Image(value=images[0], label=f"Image {1}", width=300, height=300, show_download_button=False, show_share_button=False)  
                image2 = gr.Image(value=images[1], label=f"Image {2}", width=300, height=300, show_download_button=False, show_share_button=False)
            
            

            text1 = gr.Markdown(f"### Description: <span style=\"color:red\">{prompt}</span>")
            gr.Markdown(f"### Which image matches better the given description?")
            radio1 = gr.Radio(["Image 1", "Image 2", ], label="")

            gr.Markdown("### Which image looks overall better or more natural?")
            radio2 = gr.Radio(["Image 1", "Image 2", ], label="")


            
            error_box = gr.Markdown("## \textcolor{red}{Error}", visible=False)

            # Record the user's choice when the submit button is clicked
            button = gr.Button('Next')

            #result = gr.Image(value=np.zeros((200, 200, 3)), label="Result", width=200, height=200, show_download_button=False, show_share_button=False, visible=False)




            button.click(fn=on_submit, inputs=[radio1, radio2], outputs=[image1, image2,  text1, text2, ending, button, error_box, radio1, radio2 ])
            
        return demo
    
    # present the images in a random order

    iface = create_interface()

    return iface


current_user = 6


number = 25


random.seed(12345)
user_IDs = list(range(100, 999))
random.shuffle(user_IDs)
user_name = user_IDs[current_user]
random.seed(user_name)
 
total_problems = number * 3
problem_id = 1
methods = ['lambda0', 'sg', 'excite', 'sd']

method_list = []
prompts = []
folders = []
for name in ['AnE', 'abc', 'dvmp']:
    for kk in range(number):
        # randomly choose a method
        method_number = random.choice([1, 2, 3, 4])
        method_list.append(method_number)
        method = methods[method_number-1]
        # randomly choose a prompt
        prompt = random.choice(BIG_PROMPTS[name])
        prompts.append(prompt)  
        folders.append([f'../final_data/{name}/ours/',f'../final_data/{name}/{method}/']) 
        
 
if not os.path.exists(f'./record/{user_name}'):
    os.makedirs(f'./record/{user_name}', exist_ok=True)
    # save method_list to this folder
with open(f'./record/{user_name}/method_list.txt', 'w') as f:
    # make method_list a string
    method_list_ = [str(method) for method in method_list]
    method_list_ = ','.join(method_list_)
    f.write(f"{method_list_}\n")
# save prompts
with open(f'./record/{user_name}/prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(f"{prompt}\n")


create_problem().launch(share=True)
