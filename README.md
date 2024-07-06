# Object-Conditioned Energy-Based Attention Map Alignment in Text-to-Image Diffusion Models

This repository hosts the code and resources associated with our [![Static Badge](https://img.shields.io/badge/ECCV_2024_paper-arxiv_link-blue)
](https://arxiv.org/abs/2404.07389) paper on multiple-object generation and attribute binding in text-to-image generation models like Stable Diffusion.

## Abstract
 Text-to-image diffusion models have shown great success in generating high-quality text-guided images. Yet, these models may still fail to semantically align generated images with the provided text prompts, leading to problems like incorrect attribute binding and/or catastrophic object neglect. Given the pervasive object-oriented structure underlying text prompts, we introduce a novel object-conditioned Energy-Based Attention Map Alignment (EBAMA) method to  address the aforementioned problems. We show that an object-centric attribute binding loss naturally emerges by approximately maximizing the log-likelihood of a $z$-parameterized energy-based model with the help of the negative sampling technique. We further propose an object-centric intensity regularizer to prevent excessive shifts of objects attention towards their attributes. Extensive qualitative and quantitative experiments, including human evaluation, on several challenging benchmarks demonstrate the superior performance of our method over previous strong counterparts. With better aligned attention maps, our approach shows great promise in further enhancing the text-controlled image editing ability of diffusion models.

## Envirioment Setup
Clone this repository and create a conda environment:
```
conda env create -f environment.yaml
conda activate ebama
```

If you rather use an existing environment, just run:
```
pip install -r requirements.txt
```

Finally, run:
```
python -m spacy download en_core_web_trf
```

## Inference
```
python run.py --prompt "a horned lion and a spotted monkey" --seed 1269
```

Note that this will download the stable diffusion model `CompVis/stable-diffusion-v1-4`. If you rather use an existing copy of the model, provide the absolute path using `--model_path`.

## DVMP Prompt Generation
```
python dvmp.py --num_samples 500 --dest_path destination.csv
```

### Requirements for Inputs
**num_samples**: Number of prompts to generate. Default: 200.

**dest_path**: Destination CSV file path. Default: destination.csv.


## Automatic Evaluation
```
python automatic_evaluation.py --captions_and_labels <path/to/csv/file> --images_dir <path/to/image/directory>
```

### Requirements for Inputs
**captions_and_labels**: This should be a CSV file with columns named 'caption' and 'human_annotation' (optional).

**images_dir**: This directory should have subdirectories, each named after a specific prompt given to the text-to-image model. Within each subdirectory, you should have the generated images from all the models being evaluated, following the naming convention **'{model_name}_{seed}.jpg'**.

 

## Citation

If you use this code or our results in your research, please cite as:

```bibtex
@article{zhang2024object,
  title={Object-conditioned energy-based attention map alignment in text-to-image diffusion models},
  author={Zhang, Yasi and Yu, Peiyu and Wu, Ying Nian},
  journal={arXiv preprint arXiv:2404.07389},
  year={2024}
}
```



