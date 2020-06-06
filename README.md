# Overview dsend_project_4:
- [Project motivation](#Motivation)
- [Project summary](#Summary)
- [Repository content](#Repository_content)
- [Software requirements](#Software_requirements)
- [How to run locally](#How_to_run)
- [Acknowledgements & licensing](#Acknowledgements)

## Project motivation:<a name="Motivation"></a>
background: domain, origin, related data sets

Think of trees in urban areas and you will imagine trees in parks, along riverbanks, or in backyards. Have you also thought about street trees? 

## Project summary:<a name="Summary"></a>
TODO
- Definition: "Problem statement" Problem which needs to be solved. A strategy for solving the problem & discussion of the expected solution. "Metrics" Metrics for model performance; Metrics based on problem characteristics.

Imbalanced classification problem. How can the health condition of trees be objectively categorized. F-1 score for each health condition.

- Analysis

Steps taken:
 1. EDA: Imbalanced classes.
 Data description about NaN values. No duplicate data. Answers to descriptive questions.
 
 2. Feature Engineering:
 New feature. One-hot encoding.

### 3. Step: Modelling

After loading the preprocessed data, I checked the distribution of the targets (see figure). Given the class imbalance of the three targets, I decided to split the modelling step into two phases of trying out several classifiers and evaluating those experiments with the F1 score. In case of similar F1 scores, I would further take the AUC (area under the curve) in terms of the average precision score into account.

**Figure 0: Distribution of Targets**

![Distribution](/images/sampling_none.png)

In the first phase, I experimented with different data sampling strategies and their effect on test model runs (i.e., using various classifiers with their default parameter settings). I applied oversampling (a bootstrapping approach) of the imblearn package on the training datasets (see figure).

**Figure 0: Distribution of Oversampled Targets**

![Oversampling](/images/sampling_oversampled.png)

As the oversampling approach did not fully balance all target classes, I additionally tested - among other techniques - the use of SMOTE (synthetic minority over sampling) on the already oversampled training datasets. This did not improve the F1 scores of my test models further. In the end, the use of oversampling alone yielded the highest average of all F1 scores.

In the second phase, I chose those classifiers which were among both the three highest F1 scores and average precision scores for an optimization of the respective hyperparameter settings. More specifically, I then applied a sevenfold cross-validated grid-search on the parameters of the LGBMClassifier and the RandomForestClassifier and measured the results with the F1 score. I used small adjustments, to the left and right of each parameter, to arrive at the best performing settings. Once I had tuned these two classifiers, I ran a test run with the validation dataset (i.e, previously untouched data) to decide on a classifier. I finally chose the LGBMClassifier which had both the highest F1 and average precision scores.

With regard to the final hyperparameters, let me elaborate on the robustness of the model. The LGBMClassifier reached a F1 score of 0.76 before the hyperparameter tuning. This metric increased to 0.85 after the grid-search on the training and testing datasets. The optimization thus clearly had an effect. Nonetheless, I eventually did not follow the official recommendations on [parameters tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) for better accuracy of the LightGBM algorithm entirely. On the one hand, I used the `boosting_type` as recommended and changed it to "dart". I also increased `num_leaves`, but did not define it too high to avoid overfitting. On the other hand, I could not find better results in lowering the `learning_rate`. I instead increased this parameter to a higher setting (= 1.0) than the default value (= 0.1). Further, I stayed at almost the default settings for the `n_estimators` and noticed only minor improvements by changing the `reg_lambda` parameter. Overall, if there is a parameter to be set in increase the performance of this model, I would argue that this has been `boosting_type='dart'`. Since I have fixed the `random_state`, feel free to reproduce my results. 

### 4. Step: Voila Web Application

In the last step, I wanted to try the use of [Voilà](https://github.com/voila-dashboards/voila) and interactive widgets in my modelling Jupyter notebook. The Voilà app included two features. The first feature, "Tree Map", had to map New York City's boroughs, streets, and street trees. This map should also allow for filtering trees for their health condition and refresh itself if, for example, a user wanted to clear previous choices. Figure 0 shows the current version's Tree Map.

**Figure 0: Tree Map**

![Tree Map](/images/feature_streettreemap.png)

 
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
TODO
