<div align="center">

# CPMNetv2: A Simpler and Stronger 3D Object Detection Benchmark in Medical Image

# Installation
## Create conda env  
```bash
source build_env.sh
```

# Train  
```bash
bash train_xxx.sh
```
***Note:*** args.num_sam depend on the average number of instance (lesion) in per sample (N), suggest you set to 2N. The real batch size is (args.batch_size * args.num_sam), be careful with your GPU memory.

**If you use CPMNetv2, please cite our papers:**
    
    {@inproceedings{song2020cpm,
    title={CPM-Net: A 3D Center-Points Matching Network for Pulmonary Nodule Detection in CT Scans},
    author={Song, Tao and Chen, Jieneng and Luo, Xiangde and Huang, Yechong and Liu, Xinglong and Huang, Ning and Chen, Yinan and Ye, Zhaoxiang and Sheng, Huaqiang and Zhang, Shaoting and others},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={550--559},
    year={2020},
    organization={Springer}
    }
    
    @article{luo2021scpmnet,
    title={SCPM-Net: An anchor-free 3D lung nodule detection network using sphere representation and center points matching},
    author={Luo, Xiangde and Song, Tao and Wang, Guotai and Chen, Jieneng and Chen, Yinan and Li, Kang and Metaxas, Dimitris N and Zhang, Shaoting},
    journal={Medical Image Analysis},
    volume={75},
    pages={102287},
    year={2022},
    publisher={Elsevier}
    }



