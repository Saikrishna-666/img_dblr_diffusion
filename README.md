# MRDNet 
## Image deblurring method based on self-attention and residual wavelet transform
## Contents
The directory for this readme document is as follows：  
1.Dependencies and Installation  
2.Datasets Preparation  
3.Model  
4.Train Ways  
5.Test Ways  

--------------------------------------------------------------------
### **1.Dependencies and Installation** 
* Ubuntu 16.04
* Python 3.8  
* pytorch1.7  
* cuda11.0  
* Scikit-image  
* opencv-python  
* Tensorboard  

### **2.Datasets Preparation**  
* Download GoPro datasets from Baidu Netdisk https://pan.baidu.com/s/1wtR5gIZBoW-aYSx447ZPRQ  (the password is 8k2q)
* Copy the downloaded dataset to the `dataset` folder

### **3.Model**  
* Download pretrained model from Baidu Netdisk https://pan.baidu.com/s/1sueSRt1MjSpImP-zVBX4gA  (the password is whui)
* Load the pretrained model into the project folder and place the downloaded model in the `weights` folder

### **Architecture Note (MRDNet + DBRS)**
* The implementation uses **DBRS after MRDNet**, as a final refinement stage.
* `MRDNet` produces three outputs (1/4, 1/2, and full resolution), and DBRS takes the **last/full-resolution output** as its main input.
* DBRS is conditioned on all MRDNet outputs (`cond=mrd_outs`) and predicts a residual, producing refined output as `x + residual`.
* This matches a **post-backbone refinement** design (not an intermediate skip-branch insertion).
* `--dbrs_steps` is currently a placeholder argument for future diffusion scheduling and does not run an iterative sampler loop yet.

### **4.Train Ways**  
`python main.py --model_name MRDNet --mode train --data_dir dataset/GOPRO`

Train with DBRS refinement enabled:
`python main.py --model_name MRDNet --mode train --data_dir dataset/GOPRO --use_dbrs`

### **5.Test Ways**  
` python main.py --model_name MRDNet --mode test --data_dir dataset/GOPRO --test_model weights/MRDNet.pkl`

Test with DBRS refinement enabled:
`python main.py --model_name MRDNet --mode test --data_dir dataset/GOPRO --test_model weights/MRDNet.pkl --use_dbrs`










