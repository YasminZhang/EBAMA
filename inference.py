import argparse
import os
import math
import torch
from diffusion_pipeline import EbamaDiffusionPipeline
from tqdm import tqdm
import utils.vis_utils as vis_utils




def main(
    prompts,
    seeds,
    output_directory,
    model_path,
    step_sizes,
    attn_res,
    gpu,
    dist,
    lambda_ours,
):
    pipe = load_model(model_path, gpu)
    pipe.dist = dist
    pipe.lambda_ours = lambda_ours

    for prompt in tqdm(prompts):
        images = []
        for step_size in tqdm(step_sizes):
            for seed in seeds:
                print(f'Running on: "{prompt}"')
                seed = seed.item()
                image = generate(pipe, prompt, seed, step_size, attn_res)
                save_image(image, prompt, seed, output_directory + prompt)
                images.append(image)

        joined_image = vis_utils.get_image_grid(images)
        joined_image.save(output_directory + f"{prompt}.png")


def load_model(model_path, device=0):
    device = (
        torch.device(f"cuda:{device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    pipe = EbamaDiffusionPipeline.from_pretrained(model_path).to(device)

    return pipe


def generate(pipe, prompt, seed, step_size, attn_res):
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    generator = torch.Generator(device.type).manual_seed(seed)
    result, _ = pipe(
        prompt=prompt,
        generator=generator,
        syngen_step_size=step_size,
        attn_res=(int(math.sqrt(attn_res)), int(math.sqrt(attn_res))),
    )
    return result["images"][0]


def save_image(image, prompt, seed, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    file_name = f"{output_directory}/{seed}.png"
    image.save(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, default="a checkered bowl on a red and blue table"
    )

    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--output_directory", type=str, default="./output")

    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",  # or runwayml/stable-diffusion-v1-5
        help="The path to the model (this will download the model if the path doesn't exist)",
    )

    parser.add_argument("--step_size", type=float, default=20.0, help="the step size")

    parser.add_argument(
        "--attn_res",
        type=int,
        default=256,
        help="The attention resolution (use 256 for SD 1.4, 576 for SD 2.1)",
    )

    parser.add_argument(
        "--gpu", type=int, default=0, help="The GPU to run the model on"
    )

    parser.add_argument(
        "--dist", type=str, default="cos", help="The distance loss"  # could be 'kl'
    )

    parser.add_argument(
        "--lambda_ours", type=float, default=0.5, help="The lambda for the ours loss"
    )

    args = parser.parse_args()

    main(
        [args.prompt],
        [args.seed],
        args.output_directory,
        args.model_path,
        [args.step_size],
        args.attn_res,
        args.gpu,
        args.dist,
        args.lambda_ours,
    )
