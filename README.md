# Overview dsend_project_4:
- [Project definition](#Definition)
- [Project analysis](#Analysis)
- [Project conclusion](#Conclusion)
- [Repository content](#Repository_content)
- [Software requirements](#Software_requirements)
- [How to run locally](#How_to_run)
- [Licensing, acknowledgements & references](#Acknowledgements)

# Project definition<a name="Definition"></a>

### Overview & motivation

Think of trees in urban areas and you might imagine trees in parks, along riverbanks, or in backyards. But have you also thought about street trees? This project is all about them. I stumbled upon this topic searching for a dataset on [Kaggle](https://www.kaggle.com/new-york-city/ny-2015-street-tree-census-tree-data). The City of New York has made there its Tree Census 2015 data on street trees of NYC publicy available. The topic domain is thus data science in the context of urban decision-making; a field for generating insights with a potential effect on everyday life.

### Problem statement

The idea of this project is to classify trees, given the characteristics of their appearance. In the tree census data, this information is available in the field `health` which "indicates the user's perception of tree health" (see [dataset description](/data_descriptions/nyctreecensus_2015_description.pdf)). In other words, this field contains subjective judgements by volunteers on whether a tree is in a poor, fair, or good condition - or whether a street tree might be even dead / a stump at all.

But classifying trees by their appearance can become tricky. Or how would you assess the health of this street tree?

**Figure 1: A street tree**

![Example_tree](/images/example_streettree.jpeg)

The tree itself looks healthy. But the wires cutting through the tree crown seem to impede its development. **If you did this short self-experiment with others as well, you will have noticed people can perceive the same tree differently.** Nonetheless, city councils could make use of an objective assessment whether a street tree is in a good or bad health condition.

For example, a city might decide to plant new trees in areas with many street tree stumps. Such an assessment could be offered through a classification system based on machine learning. It would take the information on the characteristics of each tree, as provided by the Tree Census data, learn which of these characterstics have been associated with which health condition, and finally classify each street tree either as a tree in a good, fair, or bad health condition. This classification system should then be easy to interact with, for example in a web application, and it should be able to classify even previously unknown street trees.

The problem statement I thus put forward is:

> **Can a machine learning model accurately assess the health condition of street trees?**<a name="Statement"></a>

### Metrics

From a technical perspective, this problem is a multilabel classification task on a sparse, imbalanced dataset. The dataset becomes sparse as I one-hote encode the categorical characteristics of each tree into dummy variables. Since accuracy scores on imbalanced datasets can be unreliable (Mueller & Guido, 2017), I take both precision and recall via the F-measure into account. The F1 score provides insight into both how precise classifications are and how many of those classifications have been correct. In addition, I use the integral under the precision-recall curve (AUC) as a secondary metric in case the F1 score cannot help in judging the performance of various algorithms in the first place.

# Project analysis<a name="Analysis"></a>

To help you understand why and how I made use of these metrics, let me describe the steps I took.

### 1. Step: Load datasets<a name="Load"></a>

The raw datasets included publicly available data from the New York Tree Census and shapefiles of both the boroughs (available via GeoPandas) and the [streets](https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b) of New York City. I pulled the Tree Census data directly via an [API of NYC Open Data](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh). Further, I converted the shapefiles of New York City to the World Geodetic System from 1984, i.e., the current standard coordinate reference system for longitudinal and latitudinal geographic data.

### 2. Step: Exploratory data analysis<a name="EDA"></a>

Since the Tree Census data offered geographical information on each street tree, I splitted the descriptive part of my analysis into a profile report and a geographical analysis. 

I generated the [profile report](/eda_trees_report.html) with Pandas Profiling. It indicated that there was no duplicated data. However, the dataset was mostly imbalanced (see figure 2). As figure 2 also shows, some fields were more strongly imbalanced than others.

**Figure 2: Class imbalance**

![Eda_classimba](/images/eda_classimba.png)

Moreover, the report pointed to missing values. With the [description of the dataset](/data_descriptions/nyctreecensus_2015_description.pdf), I could conclude that missing values always referred to dead trees or stumps. Before replacing these missing values, I investigated specific fields which had not been covered by the profile report in depth, including the diameter and species of street trees. Afterwards I noted all fields down which I wanted to process as features. These features were all categorical and described visual characteristics of street trees, such as damaged trunks or wires wrapped around branches.
 
For the geographical anaylsis I answered the following questions:

**Figure 3: Which borough of New York has the most street trees?**

![Eda_q1](/images/eda_q1.png)

> Queens has the most street trees.

**Figure 4: Which borough has the most diverse street tree flora?**

![Eda_q2](/images/eda_q2.png)

> All Queens, Brooklyn, and Bronx have equally many species of trees next to streets. However, the other boroughs have almost a similarly diverse street tree flora.

**Figure 5: In which condition are most of New York's street trees?**

![Eda_q3](/images/eda_q3.png)

> The majority of street trees in New York are healthy. The most unhealthy trees (i.e., trees with a poor or fair health condition or dead trees or stumps) can be found in Queens.

Eventually, I dropped the fields which either provided a constant information or which I considered irrelevant for answering [the problem statement](#Statement). In the end, I had a dataset with one attribute of individual street tree IDs, 31 features describing each tree, and 3 targets representing the health condition of a street tree. This dataset encompassed 683788 entries. I thereafter saved the cleaned dataset to a new .csv file.

### 3. Step: Feature engineering<a name="Featureeng"></a>
 
With the cleaned dataset, I had information on visual characterists of trees (such as wires around trunks or stones at roots). Now I also wanted to make use of the geographic information in the dataset. Therefore, I created a new feature - the number of neighboring trees in a street tree's proximity. I thereby wanted to understand whether the number of neighboring trees had any effect on the health condition of street trees. 

In other words, for this new feature I needed to count the number of trees within a certain distance for each of the street trees in NYC. I started by searching for official guidelines on the distance between street trees in New York. I found that the maximum distance between two trees should be about 9m (see page 6 of NYC's [tree planting standards](/data_descriptions/tree_planting_standards.pdf)). I then created circles around each street tree with a radius of 4.5m in order to have a maximum distance of 9m between trees which could be considered to be neighbors. Via a spatial join, I next identified which tree circles overlapped. Figure 6 shows an example with several street trees.

**Figure 6: New feature: Count of neighboring trees**

![Feeng_newfeature](/images/feeng_newfeature.png)

In the top left corner of the left graph, there are street trees which have at least two neighboring trees (as their circles intersect). In the bottom right corner of the same graph, there are trees which have no neighbors within 9m (even though they are very close). The right graph shows how many street trees in NYC in total have neighboring trees within 9m. As the figure highlights, most street trees do not fulfill the city's requirement for the maximum distance between trees. In addition, I found that if two trees are neighbors, then they are mostly more than 4.5m away from each other (the average distance is about 5.3m). I subsequently encoded the new feature `n_neighbors` by replacing values less than 1, equal to 1, or more than 1 with categorical labels.

Finally, I one-hot encoded the entire dataset to create dummy variables. During this conversion, I did not delete any features to deal with multicollinearity, even though some features were strongly correlated with each other (see figure 7).

**Figure 7: Correlation of features (and targets)**

![Feeng_correlation](/images/feeng_corrs.png)

If I had deleted features at this point (e.g., via `pd.get_dummies(df_sel, drop_first=True)`), there would have been - on the one hand - both the need for judging upon less visual characteristics of a tree in the data collection process and a faster, probably more accurate classification. However, - on the other hand - I did not want to change how the data was collected. Therefore, I did not select any features in particular. Again I saved the preprocessed data as a .csv file.

### 4. Step: Modeling<a name="Model"></a>

After loading the preprocessed data, I splitted the data into a training, testing, and validation dataset. With the datasets at hand, I checked the distribution of the targets in the training dataset (see top three plots of figure 8). Given the class imbalance of the three targets, I decided to split the modeling step into two phases.

In the first phase, I tried two classifiers along with various data sampling techniques and evaluated my experiments with the F1 score. In case of similar F1 scores, I took the AUC in terms of the average precision score into account.

**Figure 8: Distribution of Targets**

![Distribution](/images/sampling.png)

The two classifiers were a boosting framework (LGBMClassifier) and a linear model (LogisticRegression). I primarily chose a boosting algorithm because I wanted to emphasize accurate results. Gradient boosting frameworks arguably offer better results than, for example, random forests, whereas linear models are considered to perform well on large datasets (Mueller & Guido, 2017). LightGBM became my preference over other boosting frameworks, such as XGBoost, due to its leaf-wise approach.

I then experimented with different data sampling strategies and measured their effect on test model runs (i.e., fitting the two classifiers with their default parameters). In light of the imbalanced data, I initially applied oversampling (a bootstrapping approach) of the imblearn package on the training datasets (see bottom three plots of figure 8 above). As the oversampling approach did not fully balance all target classes, I additionally tried - among other techniques - the use of SMOTE (synthetic minority over sampling) on the already oversampled training dataset. This procedure did not improve the F1 scores of my test models. In the end, the use of oversampling alone yielded the highest average of all F1 scores. Nonetheless, a comparison of my metrics on the training datasets alone as well as on the training and testing datasets indicated that both models had started to overfit due to the oversampled training data. As a consequence, I decided to not use any of imblearn's sampling techniques in the next phase. Instead, I wanted to try setting the `class_weight` parameters in both classifiers to 'balanced' which is another balancing approach "to assign a larger penalty to wrong predictions on the minority class" (Raschka & Mirjalili, 2017, p. 186).

In the second phase, I then applied a grid-search on the parameters of the two classifiers with a sevenfold cross-validation and measured the results again with the overall, weighted F1 score. I used small adjustments, to the left and right of each parameter, to arrive at the best performing settings.

Once I had tuned the classifiers, I conducted another test run with the validation dataset (i.e., previously untouched data) to decide on a classifier. I finally chose the LGBMClassifier. This classifier had an overall F1 score (0.77) slightly worse than the LogisticRegression's F measure (0.78), but an average precision score (0.6797) which was a bit better than the LogisticRegression's average precision score (0.6796). A comparison of the LGBMClassifier's scores on the training datasets alone and its scores on the training & validation datasets showed that the current model responded well to unseen data (no overfitting).

With regard to the final hyperparameters, let me elaborate on the robustness of the model. The LGBMClassifier reached an overall F1 score of 0.75 before the hyperparameter tuning on an oversampled training dataset. This metric increased to 0.77 after the grid-search on the original training datasets. The optimization thus had an effect. Nonetheless, I eventually did not follow the official recommendations on [parameters tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) for better accuracy of the LightGBM algorithm entirely. On the one hand, I used the `boosting_type` as recommended and changed it to "dart". I also increased `num_leaves` (= 130), but did not define it too high to avoid overfitting. On the other hand, I could not find better results in lowering the `learning_rate`. I instead increased this parameter to a higher setting (= 0.9) than the default value. Further, I stayed at almost the default settings for the `n_estimators` (= 90). Overall, if there were parameters to be set for increasing the performance of this model, I would argue that this has been `boosting_type='dart'` and `class_weight='balanced'`. Especially the latter parameter had increased the F measure of the least balanced class. Since I have fixed the `random_state`, feel free to reproduce my results. 

### 5. Step: Voila Web Application<a name="App"></a>

In the last step, I wanted to try the use of [Voilà](https://github.com/voila-dashboards/voila) and interactive widgets in my modeling Jupyter notebook. The Voilà app included two features based on requirements which I had set myself. Note that these features might neither be novelties nor copies, as there are already a similar [New York City Street Tree Map](https://tree-map.nycgovparks.org/tree-map/) by the New York City Department of Parks & Recreation itself and several other projects on the NYC street tree data following a [hackathon](https://treescountdatajam.devpost.com/) in 2016 sponsored by the same department.

The first feature, "Street Tree Questionnaire", had to offer answer options in drop-down menus to questions resembling the original data gathering process of the New York Street Tree Census. Figure 9 shows the questions, drop-down lists, and a reset button on the left. On the right, an output message is generated depending on the answers. For example, the Voilà app will return the statement `This tree is healthy.` if the LGBMClassifier classifies the tree condition based on the characteristics of the tree to be healthy.

**Figure 9: Street Tree Questionnaire**

![Tree Quest](/images/feature_streettreequest.PNG)

The second feature, "Street Tree Map", had to visualize New York City's boroughs, streets, and street trees. Given a new street tree (based on the answers to the questionnaire), the user can explore similar street trees in New York City. The cosine similarity of the new tree with all the other trees in the dataset determines a degree of similarity at this point. Moreover, the map allows to filter for all street trees in either a good, fair, or bad health condition. The map can also be cleared if a user wants to undo previous choices by selecting "All street trees". Figure 10 displays the current version's Steet Tree Map.

**Figure 10: Street Tree Map**

![Tree Map](/images/feature_streettreemap.PNG)

The following two .gif files present a short demo of the two features.

**Figure 11: Street Tree Questionnaire in action**

![Tree Quest action](/images/feature_streettreequest_giffed.gif)

**Figure 12: Street Tree Map in action**

![Tree Map action](/images/feature_streettreemap_giffed.gif)

In figure 11, the user provides answers to the questionnaire and receives a classification of the new street tree's health condition. In figure 12, the user then asks for similar street trees. They are visualized as black circles on the map.

# Project conclusion<a name="Conclusion"></a>
 
The resulting Voilà app offers users to investigate the health condition of street trees in New York City. It also allows to experiment, i.e., to add new street trees, classify their health condition based on visual characteristics, and to discover similar existing street trees in New York. Under the hood, the app leverages a trained LightGBM classifier (with an overall, weighted F1-score of 0.77 on a validation dataset) and cosine similarity to provide the latter functionalities.

For this project I had to set the requirements for the app myself. I had first plans, but no ideas how to realize them. In addition, I wanted to use packages, such as ipywidgets and Voilà, which I had never applied before. Trying these packages for the first time was definetly of worth, even though I spent some hours understanding how to use them. Seeing what could be done with those new tools then also allowed me to adjust my requirements continously. Moreover, working with the Tree Census data in the course of this project made the domain of urban data science very interesting for me.

It goes without saying that there is still room for improvement. In particular, I refer to quality, access, speed, and relevance. For instance, I have not taken multicollinearity into account. However, a dataset with less, uncorrelated features might produce more accurate classifications. Furthermore, the Voilà app currently runs on localhost. A next step could thus be to host the app in a [cloud environment](https://voila.readthedocs.io/en/stable/deploy.html#cloud-service-providers). This step could make the app easily accessible. Developing the app with other frameworks, such as Flask, could further increase the performance of the app. At the moment, using especially the Street Tree Map requires a bit of patience. Another point for improvement can be found in the current app design. The features of the app could be redesigned by gathering requirements from and running tests with users from instutitions, such as NYC Parks, directly. The latter approach would then increase the relevancy of the app. This includes giving the app a name :-)

# Repository content:<a name="Repository_content"></a>

Please refer to the tree structure below and the subsequent description of this repository.

```bash
.
├── README.md
├── data
│   ├── data_eda
│   │   └── data.csv.gz
│   ├── data_preprocessed
│   │   └── features.csv.gz
│   └── data_raw
│       ├── nyc_boroughs
│       │   ├── nyc_geo.dbf
│       │   ├── nyc_geo.prj
│       │   ├── nyc_geo.shp
│       │   ├── nyc_geo.shp.xml
│       │   ├── nyc_geo.shx
│       │   └── nyc_geo_iso.xml
│       ├── nyc_streets
│       │   ├── geo_export_f311eef5-5969-43eb-a781-39fa54d09b70.prj
│       │   ├── geo_export_f311eef5-5969-43eb-a781-39fa54d09b70.shp
│       │   └── geo_export_f311eef5-5969-43eb-a781-39fa54d09b70.shx
│       └── nyc_trees
│           └── nyc_tree_census_2015.csv.gz
├── data_descriptions
│   ├── nyctreecensus_2015_description.pdf
│   └── tree_planting_standards.pdf
├── eda_trees.ipynb
├── eda_trees_report.html
├── featureeng_trees.ipynb
├── images
│   ├── eda_classimba.png
│   ├── eda_q1.png
│   ├── eda_q2.png
│   ├── eda_q3.png
│   ├── example_streettree.jpeg
│   ├── feature_streettreemap.PNG
│   ├── feature_streettreemap_giffed.gif
│   ├── feature_streettreequest.PNG
│   ├── feature_streettreequest_giffed.gif
│   ├── feeng_corrs.png
│   ├── feeng_newfeature.png
│   └── sampling.png
├── model
│   └── lgbm_clf.pkl
├── modeling_trees.ipynb
└── webapp_trees.ipynb
```

The `data` directory contains three subdirectories. The `data_raw` subdirectory has .dbf, .prj, .shp, .shx, and .xml files for the geographical data of the boroughs and streets of New York City. It also has a .csv file (compressed via gunzip) with the Tree Census data. The `data_eda` subdirectory similarly entails a compressed .csv file, but with a cleaned version of the Tree Census dataset. The `data_preprocessed` subdirectory then has a compressed .csv file of a one-hot encoded version of the cleaned dataset.

The `data_descriptions` directory includes two .pdf files. One file describes the fields of the original Tree Census dataset in detail. The other file holds information which I have referred to while creating a new feature.

The `images` directory encompasses .gif, .jpeg, and .png files. These are the figures shown above.

The `model` directory contains the trained classifier in a .pkl file. This file was generated by the `modeling_trees.ipynb` script.

The .ipynb scripts include the code for this project. I decided to create multiple .ipynb files for the various steps to reduce the load on memory of my machine. The .ipynb files cover the following steps of the project:

- `eda_trees.ipynb`: [1. Step](#Load) and [2. Step](#EDA)
- `featureeng_trees.ipynb`: [3. Step](#Featureeng)
- `modeling_trees.ipynb`: [4. Step](#Model)
- `webapp_trees.ipynb`: [5. Step](#App)

The remaining files are a .html file with the Pandas Profiling report from the [2. Step](#EDA) and this ReadMe itself. Overall this repository had a size of about 310 MB on my machine.

# Software requirements:<a name="Software_requirements"></a>

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
```

I used [pip](https://pip.pypa.io/en/stable/) to install these packages.

# How to run locally:<a name="How_to_run"></a>

After cloning this repository, change to its directory in your terminal. Run there the following command:

```bash
voila webapp_trees.ipynb --ExecutePreprocessor.timeout=180
```

The flag here serves to avoid a too early timeout. Your Internet browser should open now. I used Google Chrome. Otherwise follow the instructions in your terminal. Alternatively you can open the `webapp_trees.ipynb` via Jupyter and use the 'Voilà' button instead.

# Licensing, acknowledgements & references:<a name="Acknowledgements"></a>

**Licensing**
The data of the TreesCount! 2015 Street Tree Census is licensed under `CC0: Public Domain`. See further information on [Kaggle](https://www.kaggle.com/new-york-city/ny-2015-street-tree-census-tree-data). Both this dataset as well as the [dataset on NYC's streets](https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b) are publicy available via the NYC OpenData portal.

**Acknowledgements**
This is the final project of the Udacity Data Scientist for Enterprise Nanodegree. I therefore want to take the opportunity and thank Vodafone for supporting my participation in this course. Please feel free to work with the results of this project!

**References**
- Mueller, A. C., & Guido, S. (2017). *Introduction to Machine Learning with Python*. O'Reilly.
- Raschka, S., & Mirjalili, V. (2017). *Python Machine Learning*. Packt.
