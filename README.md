# Official code repository for the paper: Free Hunch: Denoiser Covariance Estimation for Diffusion Models Without Extra Costs

This repository includes code for the Free Hunch paper https://arxiv.org/abs/2410.11149. The main entry point is `generate_conditional.py` which supports multiple conditioning mechanisms used as baselines in the paper. 

## Installing Dependencies

We used Python 3.10.12. The core dependencies can be installed with:

```bash
pip install tqdm==4.66.1
pip install numpy==1.26.2
pip install scipy==1.11.4
pip install matplotlib==3.9.2
pip install scikit-image==1.3.2
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install lpips==0.1.4
pip install hydra-core==1.3.2
pip install click==8.1.7
pip install PyWavelets==1.7.0
pip install hdf5storage==0.1.19
pip install torch-dct==0.1.6
```

## Preparing the Datasets

You can download the FFHQ dataset with the instructions from https://github.com/NVlabs/ffhq-dataset. For quicker experimentation, you can download a subset. After downloading the 1024x1024 version, you can use dataset_tool.py to rescale to 256x256:
```bash
python dataset_tool.py convert --source "download_dir" --dest "data/ffhq" --resolution 256x256 --transform center-crop-dhariwal
```

For Imagenet, go to https://image-net.org/challenges/LSVRC/2012/2012-downloads.php and download the validation set (2.6GB). Then also process to 256x256 resolution with
```bash
python dataset_tool.py convert --source "download_dir" --dest "data/imagenet" --resolution 256x256 --transform center-crop-dhariwal
```

We provide a set of 10 Imagenet images for quick experimentation. 

## Downloading models
You get the FFHQ model from DPS https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing get the FFHQ model and download it to models/. The text file models/ffhq_10m_setup.txt contains the architecture setup needed for initialising the model.

For the Imagenet model, you can get it from the OpenAI guided diffusion repository link: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt. Place it in models/. The architecture setup is found in models/256x256_diffusion_uncond_setup.txt. 

When generating samples for the inverse problem with generate_conditional.py, you need to set openai_state_dict_path to models/MODEL_NAME.pt and openai_setup_path to models/256x256_diffusion_uncond_setup.txt or models/ffhq_10m_setup.txt respectively for Imagenet and FFHQ. 

## Getting the variances of frequencies in the DCT basis
To get the DCT variances of the dataset for the Free Hunch model, we have already included them in data/imagenet/dct_variance.pt. They are extracted from the Imagenet validation data, and we used the same variances for the FFHQ experiments. You can also run `do_frequency_analysis.py` to extract the variances from the data (after downloading the data). 

## Running experiments

Experiments are launched with `generate_conditional.py`. Below are example commands for the different methods. Replace `gaussian_blur` with `motion_blur`, `inpainting`, or `super_resolution` to tackle other inverse problems. Adjust `noise_sigma` to adjust the Gaussian noise standard deviation to match your experimental setup. Adjust `num_steps` to look at different step counts, and change `solver` to `euler` for the standard Euler solver. Set `total_images` to 3000 to match the results from the paper. The implementation for the "time" and "space" updates is found in conditioning_utils/online_update_bfgs.py, both for the memory-efficient representation and the dense matrix version. 

For FFHQ, you need the choices:
```bash
openai_state_dict_path=models/ffhq_10m.pt
openai_setup_path=models/ffhq_10m_setup.txt
dataset=ffhq
dataset_path=data/ffhq
```

For Imagenet, you need:
```bash
openai_state_dict_path=models/256x256_diffusion_uncond.pt
openai_setup_path=models/256x256_diffusion_uncond_setup.txt
dataset=imagenet
dataset_path=data/imagenet
```

The results folder is chosen with the `outdir` argument. Below are some example choices.

### DPS baseline

```bash
python3 generate_conditional.py \
    --conditioning_mechanism=dps \
    --cond_scaling=50 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=gaussian_blur \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --clip_x0_mean=true \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/dps_gaussian_blur
```

### PiGDM baseline

```bash
python3 generate_conditional.py \
    --conditioning_mechanism=pigdm \
    --cond_scaling=1 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=gaussian_blur \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --pigdm_posthoc_scaling=true \
    --clip_x0_mean=true \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/pigdm_gaussian_blur
```

### Online covariance model

Without space updates:
```bash
python3 generate_conditional.py \
    --conditioning_mechanism=online_covariance \
    --do_space_updates=false \
    --use_analytical_score_time_update=true \
    --project_to_diagonal=false \
    --image_base_covariance=dct_diagonal \
    --scale_factor=4.0 \
    --cond_scaling=1 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=gaussian_blur \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --pigdm_posthoc_scaling=false \
    --clip_x0_mean=false \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/free-hunch-nospace_gaussian_blur
```

With space updates:
```bash
python3 generate_conditional.py \
    --conditioning_mechanism=online_covariance \
    --do_space_updates=true \
    --use_analytical_score_time_update=true \
    --project_to_diagonal=false \
    --image_base_covariance=dct_diagonal \
    --space_step_update_lower_threshold=1000.0 \
    --space_step_update_threshold=5.0 \
    --scale_factor=4.0 \
    --cond_scaling=1 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=gaussian_blur \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --pigdm_posthoc_scaling=false \
    --clip_x0_mean=false \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/free-hunch-spaceupdate_gaussian_blur
```

### TMPD

```bash
python3 generate_conditional.py \
    --conditioning_mechanism=tmpd \
    --cond_scaling=1 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=gaussian_blur \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --pigdm_posthoc_scaling=false \
    --clip_x0_mean=true \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/tmpd_gaussian_blur
```

### Peng (Convert)

```bash
python3 generate_conditional.py \
    --conditioning_mechanism=peng_convert \
    --cond_scaling=1 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=gaussian_blur \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --pigdm_posthoc_scaling=false \
    --clip_x0_mean=true \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/peng_convert_gaussian_blur
```

### Peng (Analytic)

```bash
python3 generate_conditional.py \
    --conditioning_mechanism=peng_analytic \
    --cond_scaling=1 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=gaussian_blur \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --pigdm_posthoc_scaling=false \
    --clip_x0_mean=true \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/peng_analytic_gaussian_blur
```

### DDNM

Here we show the super resolution task as an example, as DDNM is not designed to work well with the Gaussian Blur task we use as a baseline:
```bash
python3 generate_conditional.py \
    --conditioning_mechanism=ddnm \
    --cond_scaling=1 \
    --S_churn=0 \
    --num_steps=30 \
    --solver=heun \
    --max_batch_size=1 \
    --total_images=10 \
    --save_other_images=true \
    --operator_name=super_resolution \
    --noise_sigma=0.1 \
    --num_other_images_to_save=5 \
    --pigdm_posthoc_scaling=false \
    --clip_x0_mean=true \
    --scale_factor=4 \
    --inpainting_type=random \
    --inpainting_prob_lower=0.6 \
    --inpainting_prob_upper=0.8 \
    --dataset=imagenet \
    --dataset_path=data/imagenet \
    --openai_state_dict_path=models/256x256_diffusion_uncond.pt \
    --openai_setup_path=models/256x256_diffusion_uncond_setup.txt \
    --outdir=results/ddnm_super_resolution
```

## Toy data examples
We include some toy code for running experiments with low-dimensional Gaussian mixture data, without training models. These are useful for examining just the properties of the guidance methods themselves, excluding the training process. notebooks/figure_example.ipynb showcases a basic example of how to use the code, and notebooks/figure_2.ipynb reproduces Fig.2. of the paper. 

## Acknowledgements
The codebase is built on code from multiple other repositories, including https://github.com/NVlabs/edm2 for the EDM diffusion model formulation and some utility code, https://github.com/DPS2022/diffusion-posterior-sampling and https://github.com/yuanzhi-zhu/DiffPIR for measurement operators, https://github.com/xypeng9903/k-diffusion-inverse-problems and https://github.com/wyhuai/DDNM for the implementation of the different diffusion models. The analytic_variance folder contents come from https://github.com/xypeng9903/k-diffusion-inverse-problems. 

## Citation
If you want to cite the paper, you can use the following bibtext entry:
```bash
@article{rissanen2024free,
  title={Free Hunch: Denoiser Covariance Estimation for Diffusion Models Without Extra Costs},
  author={Rissanen, Severi and Heinonen, Markus and Solin, Arno},
  journal={International Conference for Learning Representations},
  year={2025}
}
```
