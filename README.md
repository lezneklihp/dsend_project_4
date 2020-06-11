# Overview dsend_project_4:
- [Project definition](#Definition)
- [Analysis](#Summary)
- [Repository content](#Repository_content)
- [Software requirements](#Software_requirements)
- [How to run locally](#How_to_run)
- [Acknowledgements & licensing](#Acknowledgements)

# Project definition<a name="Definition"></a>

### Overview

Think of trees in urban areas and you might imagine trees in parks, along riverbanks, or in backyards. But have you also thought about street trees? This project is all about them. I stumbled upon this topic searching for a dataset on [Kaggle](https://www.kaggle.com/new-york-city/ny-2015-street-tree-census-tree-data). The City of New York has made there its Tree Census 2015 data on street trees of NYC publicy available. The topic domain is thus data science in the context of urban decision-making.

### Problem statement

The idea of this project is to classify trees, given the characteristics of their appearance. In the tree census data, this information is available in the field `health` which "indicates the user's perception of tree health" (see [dataset description](/data_descriptions/nyctreecensus_2015_description.pdf)). In other words, this field contains subjective judgements by volunteers on whether a tree is in a poor, fair, or good condition - or whether a street tree might be even dead / a stump at all.

But classifying trees by their appearance can become tricky. Or how would you assess the health of this street tree?

**Figure 1: A street tree**

![Example_tree](/images/example_streettree.jpeg)

The tree itself looks healthy. But the wires cutting through the tree crown seem to impede its development. **If you did this short self-experiment with others as well, you will have noticed people can perceive the same tree differently.** Yet city councils could make use of an objective assessment whether a street tree is in a good or bad health condition. For example, a city might decide to plant new trees in areas with many street tree stumps. Such an assessment could be offered through a classification system based on machine learning. It would take the information on the characteristics of each tree, as provided by the Tree Census data, learn which of these characterstics have been associated with which health condition, and finally classify each street tree either as a tree in a good, fair, or bad health condition.

### Metrics

From a technical perspective, this problem is a multilabel classification task on a sparse, imbalanced dataset. The dataset becomes sparse as I one-hote encode the categorical characteristics of each tree into dummy variables. Since accuracy scores on imbalanced datasets are not reliable enough, I take both precision and recall via the F-measure into account. In addition, I use the integral under the precision-recall curve (AUC) as a second metric in case the F1 score cannot help in judging the performance of an algorithm.

# Analysis<a name="Analysis"></a>

To help you understand why and how I made use of these metrics, let me describe the steps I took.

### 1. Step: Load datasets

The raw datasets included publicly available data from the New York Tree Census and shapefiles of both the boroughs (available via GeoPandas) and the [streets](https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b) of New York City. I pulled the Tree Census data directly via an [API of NYC Open Data](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh). Further, I converted the shapefiles of New York City to the World Geodetic System from 1984, i.e., the current standard coordinate reference system for longitudinal and latitudinal geographic data.

### 2. Step: Exploratory data analysis

Since the Tree Census data also offered geographical information on each street tree, I splitted the descriptive part of my analysis into a profile report and a geographical analysis. 

I generated the [profile report](/eda_trees_report.html) with Pandas Profiling. It indicated that there was no duplicated data. However, the dataset was mostly imbalanced (see figure 2). As this figure also shows, some fields were more strongly imbalanced than others.

**Figure 2: Class imbalance**

![Eda_classimba](/images/eda_classimba.png)

Moreover, the report pointed to missing values. With the [description of the dataset](/data_descriptions/nyctreecensus_2015_description.pdf), I could conclude that missing values always referred to dead trees or stumps. Before replacing these missing values, I investigated specific fields which were not covered by the profile report in depth, including the diameter and species of street trees. Afterwards I noted all fields down which I wanted to process further as features.
 
For the geographical anaylsis I answered the following questions:

**Figure 3: Which borough of New York has the most street trees?**

![Eda_q1](/images/eda_q1.png)

> Queens has the most street trees.

**Figure 4: Which borough has the most diverse street tree flora?**

![Eda_q2](/images/eda_q2.png)

> All Queens, Brooklyn, and Bronx have equally diverse street trees. However, the other boroughs have almost the same amount of different tree species.

**Figure 5: In which condition are most of New York's street trees?**

![Eda_q3](/images/eda_q3.png)

> The majority of street trees in New York are healthy. The most unhealthy trees (i.e., trees with a poor or fair health condition or dead trees or stumps) can be found in Queens.

Finally, I dropped the fields which either provided a constant information or which I considered irrelevant for answering the problem statement. I then saved the cleaned dataset to a new .csv file.

### 3. Step: Feature engineering
 
With the cleaned dataset, I had information on visual characterists of trees (such as wires on trunks or stones at roots). Now I also wanted to make use of the geographic information in the dataset. Therefore, I created a new feature - the number of neighboring trees in a street tree's proximity. I thereby wanted to understand whether the number of neighboring trees had any effect on the health condition of street trees. 

In other words, for this new feature, I needed to count the number of trees within a certain distance for each of the street trees in NYC. I started by searching for official guidelines on the distance between street trees in New York. I found that the maximum distance between two trees should be about 9m (see page 6 of NYC's [tree planting standards](/data_descriptions/tree_planting_standards.pdf)). I then created circles around each street tree with a radius of 4.5m in order to have a maximum distance of 9m between trees which could be considered to be neighbors. Via a spatial join, I next identified which tree circles overlapped. Figure 6 shows an example with several street trees.

**Figure 6: New feature: Count of neighboring trees**

![Feeng_newfeature](/images/feeng_newfeature.png)

In the top left corner of the left graph, there are street trees which have at least two neighboring trees (as their circles intersect). In the bottom right corner of the same graph, there are trees which have no neighbors within 9m (even though they are very close). The right graph shows how many street trees in NYC in total have neighboring trees within 9m. As the figure shows, most street trees do not fulfill the city's requirement for the maximum distance between trees. In addition, I found that if two trees are neighbors, then they are more than mostly 4.5m away each other (the average distance is about 5.3m). I subsequently encoded the new feature `n_neighbors` by replacing values less than 1, equal to 1, or more than 1 with categorical labels.

Finally, I one-hot encoded the entire dataset to create dummy variables. During this conversion, I did not delete any features to deal with multicollinearity, even though some features were strongly correlated with each other (see figure 7).

**Figure 7: Correlation of features (and targets)**

![Feeng_correlation](/images/feeng_corrs.png)

If I had deleted features at this point (e.g., via `pd.get_dummies(df_sel, drop_first=True)`), there would have been - on the one hand - both the need for judging upon less visual characteristics of a tree in the data collection process and a faster, probably more accurate classification. However, - on the other hand - I did not want to change how the data was collected. Therefore, I did not select any features in particular. Again I saved the preprocessed data as a .csv file.

### 4. Step: Modelling

After loading the preprocessed data, I checked the distribution of the targets (see figure 8). Given the class imbalance of the three targets, I decided to split the modelling step into two phases of trying out several classifiers and evaluating those experiments with the F1 score. In case of similar F1 scores, I would further take the AUC in terms of the average precision score into account.

**Figure 8: Distribution of Targets**

![Distribution](/images/sampling_none.png)

In the first phase, I experimented with different data sampling strategies and their effect on test model runs (i.e., using various classifiers with their default parameter settings). I initially applied oversampling (a bootstrapping approach) of the imblearn package on the training datasets (see figure 9).

**Figure 9: Distribution of Oversampled Targets**

![Oversampling](/images/sampling_oversampled.png)

As the oversampling approach did not fully balance all target classes, I additionally tested - among other techniques - the use of SMOTE (synthetic minority over sampling) on the already oversampled training datasets. This did not improve the F1 scores of my test models further. In the end, the use of oversampling alone yielded the highest average of all F1 scores.

In the second phase, I chose those classifiers which were among both the three highest F1 scores and average precision scores for an optimization of the respective hyperparameter settings. More specifically, I then applied a grid-search on the parameters of the LGBMClassifier and the RandomForestClassifier with a sevenfold cross-validation and measured the results with the F1 score. I used small adjustments, to the left and right of each parameter, to arrive at the best performing settings. Once I had tuned these two classifiers, I conducted a test run with the validation dataset (i.e., previously untouched data) to decide on a classifier. I finally chose the LGBMClassifier which had both the highest F1 and average precision scores.

With regard to the final hyperparameters, let me elaborate on the robustness of the model. The LGBMClassifier reached a F1 score of 0.76 before the hyperparameter tuning. This metric increased to 0.85 after the grid-search on the training and testing datasets. The optimization thus clearly had an effect. Nonetheless, I eventually did not follow the official recommendations on [parameters tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) for better accuracy of the LightGBM algorithm entirely. On the one hand, I used the `boosting_type` as recommended and changed it to "dart". I also increased `num_leaves`, but did not define it too high to avoid overfitting. On the other hand, I could not find better results in lowering the `learning_rate`. I instead increased this parameter to a higher setting (= 1.0) than the default value (= 0.1). Further, I stayed at almost the default settings for the `n_estimators`. Overall, if there is a parameter to be set for increasing the performance of this model, I would argue that this has been `boosting_type='dart'`. Since I have fixed the `random_state`, feel free to reproduce my results. 

### 4. Step: Voila Web Application

In the last step, I wanted to try the use of [Voilà](https://github.com/voila-dashboards/voila) and interactive widgets in my modelling Jupyter notebook. The Voilà app included two features based on requirements which I had set myself. Note that these features might neither be novelties nor copies, as there are already a [New York City Street Tree Map](https://tree-map.nycgovparks.org/tree-map/) by the New York City Department of Parks & Recreation itself and several other projects on the NYC street tree data following a [hackathon](https://treescountdatajam.devpost.com/) in 2016 sponsored by the same department.

The first feature, "Tree Map", had to map New York City's boroughs, streets, and street trees. The main idea of the map is to show & filter for the (good, fair, or bad) health condition of street trees. This map also refreshes itself if a user wants to clear previous choices by selecting "All street trees". Figure 10 displays the current version's Tree Map (without the drop-down filter).

**Figure 10: Street Tree Map**

![Tree Map](/images/feature_streettreemap.PNG)

The second feature, "Street Tree Questionnaire", had to offer answer options in drop-down menus to questions resembling the original data gathering process of the New York Street Tree Census. Figure 11 shows the questions, drop-down lists, and a reset button on the left. On the right, an output message is generated depending on the answers. For example, the Voilà app will return the statement `This tree is healthy.` if the LGBMClassifier classifies the tree condition based on the characteristics of the tree to be healthy.

**Figure 11: Street Tree Questionnaire**

![Tree Quest](/images/feature_streettreequest.PNG)

**Figure 12: Street Tree Questionnaire in action**

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

The data of the TreesCount! 2015 Street Tree Census is licensed under `CC0: Public Domain`. See further information on [Kaggle](https://www.kaggle.com/new-york-city/ny-2015-street-tree-census-tree-data). This dataset and the data on NYC's streets are publicy available via the NYC OpenData portal.

The data of the boroughs of New York City
