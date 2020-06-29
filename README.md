<center>

 

![](https://raw.githubusercontent.com/sergiomora03/MachineLearningSystems/master/notebooks/images/logo-uniandes%20(1).png)

 

</center>

 

# Métodos Avanzados de Análisis de datos

 

Instructor: *Alejandro Correa Bahnsen*

 

* email: [al.bahnsen@gmail.com](mailto:al.bahnsen@gmail.com)
* twitter: [@albahnsen](https://twitter.com/albahnsen)
* github: [albahnsen](https://github.com/albahnsen)

 

The use of statistical models in computer algorithms allows computers to make decisions and predictions, and to perform tasks that traditionally require human cognitive abilities. Machine learning is the interdisciplinary field at the intersection of statistics and computer science which develops such algorithms and interweaves them with computer systems. It underpins many modern technologies, such as speech recognition, internet search, bioinformatics, computer vision, Amazon’s recommender system, Google’s driverless car and the most recent imaging systems for cancer diagnosis are all based on Machine Learning technology.

 

# Integrantes

 

* Nidia Marcela Ortiz Sandoval <nm.ortiz33@uniandes.edu.co>
* Carlos Francisco Silva Ortiz <cf.silvao@uniandes.edu.co>
* German Guillermo Torres Hernandez <gg.torres84@uniandes.edu.co>
* Sergio Alberto Mora Pardo <s.morap@uniandes.edu.co>

 

# Contenido
### Proyecto 2

 

|Description|File|
|----|--------|
|Notebook|[Deploy](https://colab.research.google.com/drive/1egQQUPk7hN66mEf05-9g6jhf-sp9WqSc#scrollTo=Us_FZFQ8OVBq)|
|Dataset|[Kaggle Dataset](https://raw.githubusercontent.com/albahnsen/AdvancedMethodsDataAnalysisClass/master/datasets/dataTrain_carListings.zip)|
|API|[Car Price Predictor](http://ec2-3-14-9-184.us-east-2.compute.amazonaws.com:8888/)|

 

### Ejercicios

 

|Description  | File|
|-----------: |:------------:|
|8 Excercise| deploy |

 

 

# P2 - Machine Learning Systems

 

## Exercise P2.1 (50%)

 

Develop a machine learning model that predicts the price of the of car using as an input ['Year', 'Mileage', 'State', 'Make', 'Model']

 

#### Evaluation:
- 25% - Performance of the models using a manually implemented K-Fold (K=10) cross-validation
- 25% - Notebook explaining the process for selecting the best model. You must specify how the calibration of each of the parameters is done and how these change the performance of the model. It is expected that a clear comparison will be made of all implemented models.. Present the most relevant conslusions about the whole process. 

 

## Exercise P2.2 (50%)

 

Create an API of the model.

 

Example:
![](https://raw.githubusercontent.com/albahnsen/PracticalMachineLearningClass/master/notebooks/images/img015.PNG)

 

#### Evaluation:
- 40% - API hosted on a cloud service
- 10% - Show screenshots of the model doing the predictions on the local machine
