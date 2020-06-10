# Overview dsend_project_4:
- [Project definition](#Definition)
- [Analysis](#Summary)
- [Repository content](#Repository_content)
- [Software requirements](#Software_requirements)
- [How to run locally](#How_to_run)
- [Acknowledgements & licensing](#Acknowledgements)

# Project definition<a name="Definition"></a>

### Overview

Think of trees in urban areas and you might imagine trees in parks, along riverbanks, or in backyards. But have you also thought about street trees? This project is all about them. I stumbled upon this topic searching for a dataset on [Kaggle](https://www.kaggle.com/new-york-city/ny-2015-street-tree-census-tree-data). The City of New York has made there its Tree Census 2015 data on street trees of NYC publicy available. The domain of this topic is thus data science in the context of urban decision-making.

### Problem statement

The idea of this project is to classify trees, given the characteristics of their appearance. In the tree census data, this information is available in the field 'health' which "indicates the user's perception of tree health" (see description in the document in ./data_descriptions/nyctreecensus_2015_description.pdf'. In other words, when this data had been collected every volunteer gave her judgement whether a tree is in a poor, fair, or good condition - or whether a street tree might be even dead / a stump at all.

But classifying trees by their appearance can become tricky. Or how would you judge the health of this street tree (see figure 1)?

**Figure 1: A street tree**

![Example_tree](/images/example_streettree.jpeg)

The tree itself looks healthy. But the wires cutting through the tree crown seem to impede its development. **If you did this short self-experiment with others as well, you will have noticed people can perceive the same tree differently.** Yet city councils could make use of an objective assessment whether a street tree is in a good or bad health condition. For example, a city might decide to plant new trees in areas with many street tree stumps. Such an objective assessment could be offered through a classification system based on machine learning. It would take the information on the characteristics of each tree, as provided by the Tree Census data, learn which health condition has been associated with which of these characterstics, and finally classify each street tree either as a tree in a good, fair, or bad health condition.

### Metrics

From a technical perspective, this problem is a multilabel classification task on a sparse, imbalanced dataset. The dataset becomes sparse as I one-hote encode the categorical characteristics of each tree into dummy variables. Since accuracy scores on imbalanced datasets are not reliable enough, I take both precision and recall via the F-measure into account. In addition, I use the integral under the precision-recall curve (AUC) as a second metric in case the F1 score cannot help in judging the performance of a classification algorithm.

# Analysis<a name="Analysis"></a>

I took the following steps.

### 1. Step: Load datasets

The raw datasets included the New York Tree Census data and shapefiles of both the streets and boroughs of New York City. I pulled the Tree Census data directly via an [API of NYC Open Data](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh). With the description of the dataset (which I mentioned at the problem statement already) at hand, I already knew that missing data would always refer to dead trees or stumps. Hence, I replaced missing values accordingly. Further, I converted the shapefiles of New York City to the World Geodetic System from 1984, i.e., the current standard coordinate reference system for longitudinal and latitudinal geographic data.

### 2. Step: Exploratory data analysis

Since the Tree Census data also offered geographical information on each street tree, I splitted the descriptive part of my analysis into a profile report and a geographical analysis of the data. 

I generated a profile report of the Tree Census data with Pandas Profiling. The report indicated that there was no duplicated data. It also showed that the Tree Census data included several fields which I would not require for my analysis. I noted those fields down and focused on other fields which I wanted to understand better, including the diameter and species of street trees.
 
For the geographical anaylsis I wanted answers to the following questions:

**Which borough of New York has the most street trees?**

![eda_q1](/images/eda_q1.png)

> Queens has the most street trees.

**Which borough has the most diverse street tree flora?**

![eda_q1](/images/eda_q2.png)

> All Queens, Brooklyn, and Bronx have equally diverse street trees. However, the other boroughs have almost the same amount of different tree species.

**In which condition are most of New York's street trees?**

![eda_q1](/images/eda_q3.png)

> The majority of street trees in New York are healthy. The most unhealthy trees can be found in Queens.

I finally dropped those columns which I considered irrelevant for answering the problem statement and saved the dataset to a new .csv file.

### 3. Step: Feature engineering
 
 2. Feature Engineering:
 New feature. One-hot encoding.

### 4. Step: Modelling

After loading the preprocessed data, I checked the distribution of the targets (see figure). Given the class imbalance of the three targets, I decided to split the modelling step into two phases of trying out several classifiers and evaluating those experiments with the F1 score. In case of similar F1 scores, I would further take the AUC in terms of the average precision score into account.

**Figure 0: Distribution of Targets**

![Distribution](/images/sampling_none.png)

In the first phase, I experimented with different data sampling strategies and their effect on test model runs (i.e., using various classifiers with their default parameter settings). I applied oversampling (a bootstrapping approach) of the imblearn package on the training datasets (see figure).

**Figure 0: Distribution of Oversampled Targets**

![Oversampling](/images/sampling_oversampled.png)

As the oversampling approach did not fully balance all target classes, I additionally tested - among other techniques - the use of SMOTE (synthetic minority over sampling) on the already oversampled training datasets. This did not improve the F1 scores of my test models further. In the end, the use of oversampling alone yielded the highest average of all F1 scores.

In the second phase, I chose those classifiers which were among both the three highest F1 scores and average precision scores for an optimization of the respective hyperparameter settings. More specifically, I then applied a sevenfold cross-validated grid-search on the parameters of the LGBMClassifier and the RandomForestClassifier and measured the results with the F1 score. I used small adjustments, to the left and right of each parameter, to arrive at the best performing settings. Once I had tuned these two classifiers, I ran a test run with the validation dataset (i.e, previously untouched data) to decide on a classifier. I finally chose the LGBMClassifier which had both the highest F1 and average precision scores.

With regard to the final hyperparameters, let me elaborate on the robustness of the model. The LGBMClassifier reached a F1 score of 0.76 before the hyperparameter tuning. This metric increased to 0.85 after the grid-search on the training and testing datasets. The optimization thus clearly had an effect. Nonetheless, I eventually did not follow the official recommendations on [parameters tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) for better accuracy of the LightGBM algorithm entirely. On the one hand, I used the `boosting_type` as recommended and changed it to "dart". I also increased `num_leaves`, but did not define it too high to avoid overfitting. On the other hand, I could not find better results in lowering the `learning_rate`. I instead increased this parameter to a higher setting (= 1.0) than the default value (= 0.1). Further, I stayed at almost the default settings for the `n_estimators` and noticed only minor improvements by changing the `reg_lambda` parameter. Overall, if there is a parameter to be set in increase the performance of this model, I would argue that this has been `boosting_type='dart'`. Since I have fixed the `random_state`, feel free to reproduce my results. 

### 4. Step: Voila Web Application

In the last step, I wanted to try the use of [Voilà](https://github.com/voila-dashboards/voila) and interactive widgets in my modelling Jupyter notebook. The Voilà app included two features based on requirements which I had set myself. Note that these features might neither be novelties nor copies, as there are already a [New York City Street Tree Map](https://tree-map.nycgovparks.org/tree-map/) by the New York City Department of Parks & Recreation itself and several other projects on the NYC street tree data following a [hackathon](https://treescountdatajam.devpost.com/) in 2016 sponsored by the same department.

The first feature, "Tree Map", had to map New York City's boroughs, streets, and street trees. The main idea of the map is to show & filter for the (good, fair, or bad) health condition of street trees. This map also refreshes itself if a user wants to clear previous choices by selecting "All street trees". Figure 0 displays the current version's Tree Map.

**Figure 0: Street Tree Map**

![Tree Map](/images/feature_streettreemap.PNG)

The second feature, "Street Tree Questionnaire", had to offer answer options in drop-down menus to questions resembling the original data gathering process of the New York Street Tree Census. Figure 0 shows the questions, drop-down lists, and a reset button on the left. On the right, an output message is generated depending on the answers. For example, the Voilà app will return the statement `This tree is healthy.` if the LGBMClassifier classifies the tree condition based on the characteristics of the tree to be healthy.

**Figure 0: Street Tree Questionnaire**

![Tree Quest](/images/feature_streettreequest.PNG)

**Figure 0: Street Tree Questionnaire in action**

![Tree Quest action](/images/feature_streettreequest_giffed.gif)

### Project summary: Results
 
 
  "n F-measure is the harmonic mean of the precision and recall scores, and provides a more robust overall metric of your results." source 3, p182 (naturallanguageannotationformachinelearning)
 
 -> Robustness of the model? Discuss the parameters.
 "In statistics, robust is a property that is used to describe a resilience to outliers. More generally, the term robust statistics refers to statistics that are not strongly affected by certain degrees of departures from model assumptions." source (2) p. 111 (machinelearningandsecurity)
 
 Example: "The algorithm’s performance is relatively robust to the setting of alpha, meaning that setting alpha is not critical for good performance. "
 o reproducible result given a fixed random_state
 MLPClassifier
 o neural networks are sensitive to the choice of parameters ->source, p.130 (introductiontomachinelearningwithpython)
 o "lbfgs', which is quite robust" ->source 2, p.120 (introductiontomachinelearningwithpython)
 o "SGD is also a popular algorithm for training neural networks due to its robustness in the face of noisy updates. That is, it helps you build models that gen‐ eralize well." --> source 4, p. 98 (deeplearning)
 
 A more appropriate strategy to properly estimate model prediction performance is to use cross-validation (CV), which combines (e.g., averages) multiple prediction errors to measure the expected model performance. CV corrects for the expected stochastic nature of partitioning the training and testing sets and generates a more accurate and robust estimate of the expected model performance. --> source 5, p. 701 (2018_Book_DataScienceAndPredictiveAnalyt)
 

- Summary of results
Justify the results (used various approach to sampling but none of them proved to be more effective than oversampling.

- Conclusion
Reflect on the solution. Difficult or interesting aspects of the project.

 Time consuming experimentation with various sampling techniques, algorithms and hyperparameter tuning.

Improvement: 
 - The dataset has been one-hot encoded without taking multicollinearity into account. If I had set pandas.get_dummies(drop_first=True), I would have one the one hand avoided multicollinearity. On the other hand, the accuracy of my model would have decreases (as test runs have shown) and I would have had far less features for the modelling step (even though these features are being generated through data that is gathered in the tree surveys anyways.) It can be argued though, if the issue of multicollinearity is relevant to a decision tree or a gradient boosting classifier anyways.
 
 - I could have chosen another classifier and with the SMOTE data sampling technique.
 
 - smoother app usage. Other algorithms such as RandomForestClassifier. Make the app less location dependent.


## Repository content:<a name="Repository_content"></a>
TODO: short description of each file.

```bash
.
├── README.md
├── 
└── 
```

## Software requirements:<a name="Software_requirements"></a>
Please use Python version 3.7.1 & the following packages:

```bash
Fiona==1.8.13,
GDAL==3.0.4,
geopandas==0.6.3,
geopy==1.21.0,
imbalanced-learn==0.6.2,
ipympl==0.5.6,
ipywidgets==7.5.1,
joblib==0.14.1,
lightgbm==2.3.1,
matplotlib==3.2.0,
notebook==6.0.2,
numpy==1.18.1,
pandas==1.0.3,
pandas-profiling==2.8.0,
pyproj==2.4.2.post1,
Rtree==0.9.3,
scikit-learn==0.22,
seaborn==0.9.0,
Shapely==1.7.0,
voila==0.1.21,
xgboost==1.1.0
```

I used [pip](https://pip.pypa.io/en/stable/) to install these packages.

## How to run locally:<a name="How_to_run"></a>
After cloning this repository, change to its directory in your terminal. Run the following command:

```bash
voila <name.ipynb> --ExecutePreprocessor.timeout=180
```

Your Internet browser should open now. Otherwise follow the instructions in your terminal.

## Acknowledgements, licensing & sources:<a name="Acknowledgements"></a>

The data of the TreesCount! 2015 Street Tree Census is licensed under `CC0: Public Domain`. See further information on [Kaggle](https://www.kaggle.com/new-york-city/ny-2015-street-tree-census-tree-data).
