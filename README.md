# Distribution-Decouple Learning Network: An Innovative Approach for Single Image Dehazing with Spatial and Frequency Decoupling

Yabo Wu, Wenting Li, Ziyang Chen, Hui Wen, Zhongwei Cui, and Yongjun Zhang*

>Image dehazing methods face challenges in addressing the high coupling between haze and object feature distributions in the spatial and frequency domains. This coupling often results in oversharpening, color distortion, and blurring of details during the dehazing process. To address these issues, we introduce the Distribution-Decouple Module (DDM) and Dual-Frequency Attention Mechanism (DFAM). The DDM works effectively in the spatial domain, decoupling haze and object features through a Feature Decoupler, and then uses a Two-stream Modulator to further reduce the negative impact of haze on the distribution of object features. Simultaneously, the DFAM focuses on decoupling information in the frequency domain, separating high and low-frequency information and applying attention to different frequency components for frequency calibration. Finally, we introduce a novel dehazing network, the Distribution-Decouple Learning Network for Single Image Dehazing with Spatial and Frequency Decoupling (DDLNet). This network integrates DDM and DFAM, effectively addressing the issue of coupled feature distributions in both spatial and frequency domains, thereby enhancing the clarity and fidelity of the dehazed images. Extensive experiments indicate the outperforms of our DDLNet when compared to the state-of-the-art (SOTA) methods, achieving a 1.50 dB increase in PSNR on the SOTS-indoor dataset. Concomitantly, it indicates a 1.26 dB boost on the SOTS-outdoor dataset. Additionally, our method performs significantly well on the nighttime dehazing dataset NHR, achieving a 0.91 dB improvement. Code and trained models are available at https://github.com/aoe-wyb/DDLNet.

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Download the Datasets
- [reside-indoor](https://drive.google.com/drive/folders/1pbtfTp29j7Ip-mRzDpMpyopCfXd-ZJhC)
- [reside-outdoor](https://drive.google.com/drive/folders/1eL4Qs-WNj7PzsKwDRsgUEzmysdjkRs22)
## Training and Evaluation
### Training on ITS:
~~~
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --data_dir your_dataset_path/reside-indoor --train_data ITS-train --valid_data ITS-test
~~~
### Training on OTS:
~~~
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --data_dir your_dataset_path/reside-outdoor --train_data OTS-train --valid_data OTS-test
~~~
### Testing on SOTS-Indoor:
~~~
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_dir your_dataset_path/reside-indoor --test_data ITS-test --test_model save_model_path
~~~
### Testing on SOTS-Indoor:
~~~
CUDA_VISIBLE_DEVICES=0 python main.py --mode test --data_dir your_dataset_path/reside-outdoor --test_data OTS-test --test_model save_model_path
~~~
## Results
|Task|Dataset|PSNR|SSIM|
|----|------|-----|----|
|**Image Dehazing**|SOTS-Indoor|42.32dB (42.51dB)|0.996|
||SOTS-Outdoor|39.27dB|0.996|
||Dense-Haze|17.29dB|0.630|
||NH-HAZE|20.33dB|0.810|
||NHR|26.44dB|0.972|

- The reside-indoor weights were lost in the paper, and the PSNR for the weights provided in this repository was 42.51 dB.

## Citation
~~~
@article{wu2024distribution,
  title={Distribution-decouple learning network: an innovative approach for single image dehazing with spatial and frequency decoupling},
  author={Wu, Yabo and Li, Wenting and Chen, Ziyang and Wen, Hui and Cui, Zhongwei and Zhang, Yongjun},
  journal={The Visual Computer},
  pages={1--16},
  year={2024},
  publisher={Springer}
}
~~~

## Contact
Please contact Yabo Wu (1394884511@qq.com)if you have any questions.

**Acknowledgment:** This code is based on the [MIMO-UNet](https://github.com/chosj95/MIMO-UNet/tree/main?tab=readme-ov-file#gpu-syncronization-issue-on-measuring-inference-time) and [SFNet](https://github.com/c-yn/SFNet).
