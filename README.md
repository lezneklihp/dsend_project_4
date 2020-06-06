# Overview dsend_project_4:
- [Project motivation](#Motivation)
- [Project summary](#Summary)
- [Repository content](#Repository_content)
- [Software requirements](#Software_requirements)
- [How to run locally](#How_to_run)
- [Acknowledgements & licensing](#Acknowledgements)

## Project motivation:<a name="Motivation"></a>
background: domain, origin, related data sets

Think of trees in urban areas and you will point to trees in parks, along riverbanks, or in backyards. Have you also thought about street trees? 

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

3. Modelling:
 Oversampling. Experimenting with XGBClassifier, LGBMClassifier, & MLPClassifier; Hyperparameter Optimization. Time consuming process. Testing with optimized models.
 
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
 
 4. Voila web application

- Summary of results
Justify the results (used various approach to sampling but none of them proved to be more effective than oversampling.

- Conclusion
Reflect on the solution. Difficult or interesting aspects of the project.

Improvement: 
 - The dataset has been one-hot encoded without taking multicollinearity into account. If I had set pandas.get_dummies(drop_first=True), I would have one the one hand avoided multicollinearity. On the other hand, the accuracy of my model would have decreases (as test runs have shown) and I would have had far less features for the modelling step (even though these features are being generated through data that is gathered in the tree surveys anyways.) It can be argued though, if the issue of multicollinearity is relevant to a decision tree or a gradient boosting classifier anyways.
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
