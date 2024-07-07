# [ECCV 2024] Object-Conditioned Energy-Based Attention Map Alignment in Text-to-Image Diffusion Models

This repository hosts the code and resources associated with our [![Static Badge](https://img.shields.io/badge/ECCV_2024_paper-arxiv_link-blue)
](https://arxiv.org/abs/2404.07389)  on multiple-object generation and attribute binding in text-to-image generation models like Stable Diffusion.

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
to install the transformer-based spaCy NLP parser. 

 

## Datasets
In this work, we use the following datasets:
- AnE dataset from [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite). We provide the AnE dataset  `ane_data.py`  in the `data` folder.
- DVMP dataset from [SynGen](https://github.com/tdspora/syngen). Please follow the repo to randomly generate the DVMP dataset.
- ABC-6K dataset from [StrDiffusion](https://github.com/weixi-feng/Structured-Diffusion-Guidance). We provide the full ABC-6K dataset `ABC-6K.txt` in the `data` folder and a subset of the dataset in `data_abc.py`.


## EBAMA (our method)
To test our method on a specific prompt, run:
```
python ebama.py --prompt "a horned lion and a spotted monkey" --seed 1269
```
Note that this will download the stable diffusion model `CompVis/stable-diffusion-v1-4`. If you rather use an existing copy of the model, provide the absolute path using `--model_path`. 


## Metrics
We mainly use the following metrics to evaluate the generated images:
- Text-Image Full Similarity
- Text-Image Min Similarity 
- Text-Caption Similarity

```
python automatic_evaluation.py --captions_and_labels <path/to/csv/file> --images_dir <path/to/image/directory>
```



 

## Citation

If you find this code or our results useful, please cite as:

```bibtex
@article{zhang2024object,
  title={Object-conditioned energy-based attention map alignment in text-to-image diffusion models},
  author={Zhang, Yasi and Yu, Peiyu and Wu, Ying Nian},
  journal={arXiv preprint arXiv:2404.07389},
  year={2024}
}
```



