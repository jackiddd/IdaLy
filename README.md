# IdaLy
IdaLy provides a python library and a software that support various algorithms for industrial data augmentation, which can promote the effiency of your industrial machine learning projects(such as fault diagnosis and soft sensor).You can use any algorithms in your script files by importing the python library.Certainly, if you are not familiar with coding of python or just want to finish augmentation tasks conveniently, you can use our software directly.  
## Augmentations
Because of the sampling difficulty and data privacy of industrial data, data we can acquire from real industry process sometimes is insufficient, thus causing problems such as overfitting and class-imblance while training intelligent agencies. Nowadays, more and more data augmentation algorithms are applied in the field of industrial big data. We collect some algorithms often used and classify them as follows:  
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
 
 We collect these algorithms in a python library.You can download it in [idaly](https://github.com/3uchen/IdaLy/tree/master/src/idaly) directly. Or you can download the library by pip instrustion: `pip install idaly`. It should be noted that the library is Python-based and requires at least Python 3.6, and the envrionment should satify [requirement.txt](https://github.com/3uchen/IdaLy/blob/master/requirements.txt).  
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
 ![example_1](https://github.com/3uchen/IdaLy/blob/master/example_1.png)  
 To fininsh industrial data augmentation tasks more conveniently, we intergrate algorithms mentioned above into a software developed by pyqt5. You can download the software [here](https://drive.google.com/file/d/1muqsfoieiJoRcCWeEK9OmyYlBWDwvyO4/view?usp=sharing) directly, or clone Idaly and run our python file.  
 ```
 git clone git@github.com:3uchen/IdaLy.git
 cd src\idap
 python idap_v1.py
 ```  
Idap mainly consists of seven modules:data import, algorithm select, algorithm description, parameter setting, PCA visualization, simulation test and save.
- **Data import**  
You can click the label button of "open file" to import the original data, which only supports numpy file currently. The original data can either carry task label or not. The task can be industrial regression (such as soft sensor) and classification (such as fault diagnosis), which can be identified by Idap automatically. When it's regression, Idap will see continuous output as a feature. And when it's classfication, Idap will recombine original data according to the label and then generate data respectively.
- **Algorithm select**  
You can select any intergrated algorithm by clicking related buttons. You can also choose mode of single algorithms or combinational algorithms(developing).
- **Algorithm descripition**  
This module will display the brief description and reference of algorithm you have selected.
- **Parameter setting**  
This module will display the parameter setting interface of algorithm you have selected. You can preset parameters or adjust them in real time according to visualization output.
- **PCA visualization**  
You can start data augmentaition procedure by clicking "START" buttoon. After finishing it, Idap will use PCA to reduce the dimensionality of original data and generated data. And the scatter figure of bidimensional data will display on the interface, where you can indentify the level of similarity between original data and generated data.
- **Simulation test**  
Idap embed simulation test module to further identify the effect of data augmention. We currently provide two test scenes:soft sensor and fault diagnosis. After data augmentation procedure, Idap will input the original data and new data (original data and generated data) to train model of test scenes respectively and display test results about the trained models. We will embed more test scenes and models in the future.
- **Save**  
You can save generated data, generative model (if any) and test result by click the related buttons respectively. Or you can save all results by click the label button of "Save all".  
## Citation  


 
