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
 
 -> Robustness of the model? Discuss the parameters. 
 
 4. Voila web application

- Summary of results
Justify the results (used various approach to sampling but none of them proved to be more effective than oversampling.

- Conclusion
Reflect on the solution. Difficult or interesting aspects of the project.

Improvement: smoother app usage. Other algorithms such as RandomForestClassifier. Make the app less location dependent.


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
ipympl==0.5.6,
ipywidgets==7.5.1,
joblib==0.14.1,
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
voila==0.1.21
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
