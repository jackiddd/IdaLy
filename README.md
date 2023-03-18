# IdaLy
IdaLy provides a python library and a software that support various algorithms for industrial data augmentation, which can promote the effiency of your industrial machine learning projects(such as fault diagnosis and soft sensor).You can use any algorithms in your script files by importing the python library.Certainly, if you are not familiar with coding of python or just want to finish augmentation tasks conveniently, you can use our software directly.  
## Augmentations
Because of the sampling difficulty and data privacy of industrial data, data we can acquire sometimes is insufficient, thus causing problems such as overfitting and class-imblance often occur while training intelligent agencies. Nowadays, more and more data augmentation algorithms are applied in the field of industrial big data. We collect some algorithms often used and classify them as follows:  
- **Noise Injection**
  - GNI:  Gaussian Nosie Injection
- **Interpolation**
  - SMOTE, Synthetic Minority Over-sampling Technique. Reference: [SMOTE](https://www.jair.org/index.php/jair/article/view/10302/24590), [SMOTE in industrial data](https://ieeexplore.ieee.org/abstract/document/9858365).
  - LLE, Locally Linear Embedding. Reference: [LLE](https://www.science.org/doi/abs/10.1126/science.290.5500.2323).
  - MTD, Mega Trend Diffusion. Reference: [MTD](https://www.sciencedirect.com/science/article/pii/S0305054805001693).
  - KNNMTD, k-Nearest Neighbor Mega-Trend Diffusion. Reference: [KNNMTD](https://www.sciencedirect.com/science/article/pii/S0950705121009473).
- **Probability Model**
  - GMM, Gaussian Mixture Model. Reference: [GMM](http://leap.ee.iisc.ac.in/sriram/teaching/MLSP_16/refs/GMM_Tutorial_Reynolds.pdf), [GMM in industrial data](https://www.sciencedirect.com/science/article/pii/S002002552100935X). 
- **Deep Learning**
  - GAN, Generative Adversarial Network. Reference: [GAN](https://dl.acm.org/doi/pdf/10.1145/3422622), [GAN in industrial data](https://dl.acm.org/doi/pdf/10.1145/3422622X).
 
 We collect these algorithms in a python library.You can download it in [augmentation](https://github.com/3uchen/IdaLy/blob/master/src/augmentation.py) directly. Or you can download the library by pip instrustion: `pip install idaly`. It should be noted that the library is Python-based and requires at least Python 3.6, and the envrionment should satify [requirement.txt](https://github.com/3uchen/IdaLy/blob/master/requirements.txt).  
 Here we give an example about how to use the library.  
```python
 import idaly.augmentation as aug
 
 ori_data_path = "your_ori_industrial_data_path.npy"
 out_data_path = "your_output_path.npy"
 ori_data = np.load(ori_data_path)
 
 # aug_method = aug.method(parameter)
 # data_gen = aug_method.fit(ori_data)
 aug_smote = aug.Smote(N=the numer generated, k=15)
 gen_data = aug_smote.fit(ori_data)
 
 # visualization()
 np.save(np.concentrate((ori_data, gen_data), axis=0),out_data_path)
 ```
 ## Industrial Data Augmentation Platform
 To fininsh industrial data augmentation tasks more conveniently, we intergrate algorithms above into a software developed by pyqt5. You can download the software [here](https://drive.google.com/file/d/1muqsfoieiJoRcCWeEK9OmyYlBWDwvyO4/view?usp=sharing) directly and see the source code here.
 ![example_1](https://github.com/3uchen/IdaLy/blob/master/example_1.png)  
Idap mainly consists of senve modules:data import, algorithm select, algorithm description, parameter setting, PCA visualization, simulation test and save.
- **Data import**
    
- **Algorithm select**
- **Algorithm descripition**
- **Parameter setting**
- **PCA visualization**
- **Simulation test**
- **Save**

 
