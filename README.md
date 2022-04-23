# A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images
by Zhiming Cui, Yu Fang, Lanzhuju Mei, Bojun Zhang, Bo Yu, Jiameng Liu, Caiwen Jiang, Yuhang Sun, Lei Ma, Jiawei Huang, Yang Liu, Yue Zhao, Chunfeng Lian, Zhongxiang Ding, Min Zhu, and Dinggang Shen.


### Introduction

This repository is for our Nature Communications 2022 paper 'A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images'. 


### Installation
This repository is based on PyTorch 1.4.0.

### Training
The training and data preparation code will be released within two weeks.

### Inference
1. Write the full path of the CBCT data (.nii.gz) in the file.list, and set up in the test.py.
2. Run the model: Python test.py.

### Data
We have released the partial CBCT data (about 50 scans) with GT annotation. The link is: https://pan.baidu.com/s/1GabHtrd04g8DiMJwZO2Ksg pw: w1hz

### Note
Due to the commercial issue, we cannot release the trained model trained on the large dataset. If you want to get the results on your own CBCT, please contact Zhiming Cui 'cuizm.neu.edu@gmail.com'. He will run the results within ONE week.

### Citation

If the code or data is useful for your research, please consider citing:

    @inproceedings{cui2019toothnet,
         title={ToothNet: automatic tooth instance segmentation and identification from cone beam CT images},
         author={Zhiming Cui, Changjian Li, and Wenping Wang},
         booktitle = {CVPR},
         year = {2022}}
         
    @inproceedings{cui2021hierarchical,
         title={Hierarchical Morphology-Guided Tooth Instance Segmentation from CBCT Images},
         author={Cui, Zhiming and Zhang, Bojun and Lian, Chunfeng and Li, Changjian and Yang, Lei and Wang, Wenping and Zhu, Min and Shen, Dinggang},
         booktitle={International Conference on Information Processing in Medical Imaging},
         pages={150--162},
         year={2021},
         organization={Springer}} 
         
    @article{cui2022fully,
        title={A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images},
        author={Cui, Zhiming and Fang, Yu and Mei, Lanzhuju and Zhang, Bojun and Yu, Bo and Liu, Jiameng and Jiang, Caiwen and Sun, Yuhang and Ma, Lei and Huang, Jiawei         and others},
        journal={Nature Communications},
        volume={13},
        number={1},
        pages={1--11},
        year={2022},
        publisher={Nature Publishing Group}}

### Questions
Please contact 'cuizm.neu.edu@gmail.com'
