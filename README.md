# MedicalNeRF
## Introduction
### NeRF Datasets of Highly Reflective Medical Surgical Instruments.

Currently, we provide two formats of datasets for use: NeRF format and Instant-NGP format. The two datasets differ slightly in the camera pose files, ` transforms_train.json ` and ` transforms_test.json `, as Instant-NGP requires an additional parameter,` AABB_Scale`, for training. Aside from this, the training and testing images in both datasets remain identical.

#### Example datasets：
![image](https://github.com/user-attachments/assets/27ad64f5-709e-4fb0-bec1-febadd0e9b4f)

            MedicalNeRF aims to provide a NeRF dataset focused on medical surgical instruments. 
            This is an often-overlooked area in previous research, despite some studies utilizing methods like ray reflection 
            equations to optimize NeRF's performance on reflective surfaces. However, there has been minimal focus on the highly 
            reflective specific applications within medical devices. Our research highlights include a series of NeRF datasets 
            featuring high-reflectivity medical instruments, enabling NeRF researchers to improve rendering performance under complex
            lighting conditions. Another significant aspect is our aim to extend this research into robotic grasping. The dataset
            provided by MedicalNeRF is particularly important in robotics, as traditional research has struggled with how to grasp
            highly reflective and transparent objects, with no effective methods established yet. Grasping the more finely detailed
            high-reflectivity medical instruments presents even greater challenges. We hope that the MedicalNeRF dataset will
            facilitate further advancements in these studies. 

## Citation

Our code is based on the following code:
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```


If you find this work helpful, please consider to cite:
```
@misc{MedicalNerf,
    title={MedicalNeRF: NeRF Datasets of Highly Reflective Medical Surgical Instruments},
    author={Lifei, Xiao},
    year={2024},
    month = Sept,
    address = {Nanjing University of Information Science & Technology, Nanjing, CHINA},
    publisher = {Github},
    primaryClass={cs.CV}
    
}

```

Copyright © 2024, HCI Lab, Nanjing University of Information Science & Technology. All rights reserved.
