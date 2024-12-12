![image](https://github.com/user-attachments/assets/5b61590d-7899-4f18-8c71-848231e6cf84)# IdaLy
This repository contains code for the paper, Advancing Industrial Data Augmentation: A Thorough Survey from Foundations to Frontier Applications (submitted). IdaLy provides a python library and a software that support various algorithms for industrial data augmentation, which can promote the effiency of your industrial machine learning projects (such as fault diagnosis and soft sensor). You can use any algorithms in your script files by importing the python library. Certainly, if you are not familiar with coding of python or just want to finish augmentation tasks conveniently, you can use our software directly.  
## Augmentations
Because of the sampling difficulty and data privacy of industrial data, data we can acquire from real industry process sometimes is insufficient, thus causing problems such as overfitting and class-imblance while training intelligent agencies. Nowadays, more and more data augmentation algorithms are applied in the field of industrial big data. We collect some algorithms often used and classify them as follows:  
 ![methods](https://github.com/3uchen/IdaLy/blob/master/methods.png)  
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
 ![example](https://github.com/3uchen/IdaLy/blob/master/example.png)  
 To fininsh industrial data augmentation tasks more conveniently, we intergrate algorithms mentioned above into a software developed by pyqt5. You can download the software [here](https://drive.google.com/file/d/1muqsfoieiJoRcCWeEK9OmyYlBWDwvyO4/view?usp=sharing) directly, or clone Idaly and run our python file.  
 ```
 git clone git@github.com:3uchen/IdaLy.git
 cd src\idap
 python idap_v1.py
 ```  
Idap mainly consists of seven modules: data import, algorithm select, algorithm description, parameter setting, PCA visualization, simulation test and save.
- **Data import**  
The data import module, accessed by clicking the ‘open file’ blue label in the menu bar, is mainly used for the importation of raw data and identification of industrial context. At present, the IDAS supports importing data in NumPy, Excel, and Text file formats, which includes features and labels. The IDAS will automatically recognize the industrial task type according to the label of data after importation. In a regression task scenario (e.g., KPI predictions), the IDAS regards the continuous output label as feature, while in a classification task scenario (e.g., fault diagnosis), the IDAS reorganizes the raw data based on labels and sequentially generates virtual data for each class.
- **Methods Selection and Description Module**  
As of now, the IDAS has integrated twenty IDA methods, which are categorized into three classed: transformation, interpolation and distribution estimation. Users can opt for either single algorithm mode or combined mode during the algorithm selection, after which the algorithm description module will present a brief overview of the selected along with links to references.
- **Parameter Configuration Module**  
This module allows users to preconfigure parameters, while also enabling users to dynamically adjust parameters based on results of data generation and industrial intelligence tasks assessment to achieve optimal performance.
- **PCA Visualization Module**  
After the process of data generation, PCA visualization module leverages PCA to reduce the dimensionality of the newly merged data from both the original and synthetic datasets, and subsequently plots them on the same two-dimensional scatterplot respectively, through which users can have an initial assessment of the effectiveness of data augmentation from the perspective of data diversity and fidelity of synthetic samples. It is worth mentioning that, in a classification task scenario, by clicking on the corresponding legend of a specific category on the scatterplot, the IDAS can display the visualization effect of that category.
- **Simulation test**  
IDAS embed simulation test module to further identify the effect of data augmentation. We currently provide two test scenes: soft sensor and fault diagnosis. After data augmentation procedure, Idap will input the original data and new data (original data and generated data) to train model of test scenes respectively and display test results about the trained models. We will embed more test scenes and models in the future.
- **Save**  
You can save generated data, generative model (if any) and test result by click the related buttons respectively. Or you can save all results by click the label button of "Save all".  

 # visualization()
Based on Tennessee Eastman (TE) process dataset, We conducted a comparative experiment of 19 commonly used industrial data augmentation methods and the results are: 

|    Methods   |    Time(s)                   | Accuracy (SVM)              | Accuracy (MLP)              |
|:------------:|:-------------:|:-------------:|:--------------:|:------:|:-----------:|:--------------:|:------:|:-----------:|
|              | Modeling time | Sampling time |     before     |  after | improve (%) |     before     |  after | improve (%) |
|      GNI     |     0.000     |     0.006     |     92.00      | 92.25  |    0.27     |     91.50      | 92.75  |    1.24     |
|      MNI     |     0.000     |     0.155     |     92.00      | 92.50  |    0.54     |     92.50      | 94.00  |    1.62     |
|      SNI     |     0.000     |     0.005     |     94.75      | 95.25  |    0.53     |     91.75      | 94.00  |    2.45     |
|      PNI     |     0.000     |     0.006     |     94.00      | 94.00  |    0.00     |     92.50      | 95.25  |    2.97     |
|     SMOTE    |     0.000     |     0.053     |     92.50      | 93.00  |    0.54     |     93.00      | 93.50  |    0.54     |
| Kmeans_SMOTE |     0.000     |     0.063     |     94.25      | 94.75  |    0.53     |     92.00      | 95.25  |    3.53     |
|     MIXUP    |     0.000     |     0.006     |     92.75      | 93.50  |    0.81     |     90.75      | 96.25  |    6.06     |
|      LLE     |     0.016     |     9.885     |     93.50      | 93.50  |    0.00     |     93.50      | 94.75  |    1.34     |
|      MTD     |     0.000     |     7.653     |     94.25      | 94.25  |    0.00     |     93.25      | 90.75  |    -2.68    |
|    KNNMTD    |     0.000     |    13.739     |     92.00      | 91.75  |    -0.27    |     93.25      | 92.25  |    -1.07    |
|      GMM     |     0.037     |     0.034     |     93.25      | 94.00  |    0.80     |     92.75      | 94.75  |    2.16     |
|      GAN     |    32.247     |     0.121     |     93.25      | 94.00  |    0.80     |     92.00      | 93.50  |    1.63     |
|    WGAN_GP   |    73.206     |     0.109     |     91.00      | 92.50  |    1.65     |     87.75      | 90.75  |    3.42     |
|     LSGAN    |    55.960     |     0.127     |     93.25      | 93.25  |    0.00     |     91.75      | 92.50  |    0.82     |
|      VAE     |    29.695     |     0.072     |     93.00      | 95.50  |    2.69     |     90.75      | 91.50  |    0.83     |
|    VAEGAN    |    955.517    |     0.040     |     94.00      | 94.25  |    0.27     |     92.75      | 94.00  |    1.35     |
|     DDPM     |    57.858     |     3.705     |     93.00      | 93.00  |    0.00     |     93.25      | 89.75  |    -3.75    |
|    REALNVP   |    121.580    |     1.600     |     93.00      | 92.75  |    -0.27    |     92.00      | 92.75  |    0.82     |
|     GLOW     |    153.070    |     1.610     |     93.50      | 93.75  |    0.28     |     92.25      | 93.75  |    1.63     |
 
