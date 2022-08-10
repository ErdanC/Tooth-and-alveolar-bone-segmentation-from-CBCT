# A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images
by Zhiming Cui, Yu Fang, Lanzhuju Mei, Bojun Zhang, Bo Yu, Jiameng Liu, Caiwen Jiang, Yuhang Sun, Lei Ma, Jiawei Huang, Yang Liu, Yue Zhao, Chunfeng Lian, Zhongxiang Ding, Min Zhu, and Dinggang Shen.


### Introduction

This repository is for our Nature Communications 2022 paper 'A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images'. 


### Installation
This repository is based on PyTorch 1.4.0.

### Training
The training and data preparation codes of the first and second stages have been released.

### Inference
1. Write the full path of the CBCT data (.nii.gz) in the file.list, and set up in the test.py.
2. Run the model: Python test.py.

### Data registration
We have released the partial CBCT data (about 150 scans) with GT annotation. If you want to apply for these data, please complete the registration form in following link (https://drive.google.com/file/d/1mI5GVoT5FDq4uLXfxJmQKwWpMO6J4oUY/view?usp=sharing), and then send to Zhiming Cui ('zmcui94@gmail.com'). He will send you the download link when recieve the data registration form.

### AI inference application
Due to the commercial issue, we cannot release the trained model trained on the large dataset. If you want to get the results on your own CBCT, please complete the AI inference application form in the following link (https://drive.google.com/file/d/1xzFugXfz_coKepT_2PrGH5OwtR6mbUGX/view?usp=sharing), and then send to Zhiming Cui ('zmcui94@gmail.com'). He will send you the data upload link and feedback the results in one week, when recieving the application form.

### Citation

If the code or data is useful for your research, please consider citing:

    @inproceedings{cui2019toothnet,
         title={ToothNet: automatic tooth instance segmentation and identification from cone beam CT images},
         author={Cui, Zhiming and Li, Changjian and Wang, Wenping Wang},
         booktitle = {CVPR},
         year = {2019}}
         
    @inproceedings{cui2021hierarchical,
         title={Hierarchical Morphology-Guided Tooth Instance Segmentation from CBCT Images},
         author={Cui, Zhiming and Zhang, Bojun and Lian, Chunfeng and Li, Changjian and Yang, Lei and Wang, Wenping and Zhu, Min and Shen, Dinggang},
         booktitle={International Conference on Information Processing in Medical Imaging},
         pages={150--162},
         year={2021},
         organization={Springer}} 
         
    @article{cui2022fully,
        title={A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images},
        author={Cui, Zhiming and Fang, Yu and Mei, Lanzhuju and Zhang, Bojun and Yu, Bo and Liu, Jiameng and Jiang, Caiwen and Sun, Yuhang and Ma, Lei and Huang, Jiawei and others},
        journal={Nature Communications},
        volume={13},
        number={1},
        pages={1--11},
        year={2022},
        publisher={Nature Publishing Group}}

### Questions
Please contact zmcui94@gmail.com'.
