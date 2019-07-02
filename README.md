# Electric Vehicle Detection  
Increasing elective vehicle (EV) ownership and thus electricity demand introduces challenges for existing energy infrastructure. Developing predictive models to understand and manage the demand may mitigate risks and enable benefits of increased EV ownership. Using 60 days of labeled smart meter power readings from 1590 houses with and without EVs, binary classifiers were trained to predict (a) which houses have an EV and (b) during which time intervals an EV was charging. The Logistic Regression classifier trained for part A yielded an accuracy between 0.79 and 0.86. The K-Nearest Neighbors classifier for part B yielded an accuracy between 0.75 and 0.96. Predictions were made on a blind test set of 699 houses using the trained models from parts A and B.

![Project Classifiers](https://github.com/AndreaKeane/electric-vehicle-detection/blob/master/figures/classifier_chain.png)

Project Status: Completed

# Project Details  
## Methods  
* Feature Engineering  
* Logistic Regression  
* K-Nearest Neighbors

## Technologies  
* Python  
* JupyterNotebooks 
* Pandas, Numpy, scikit-learn
* MatPlotLib.pyplot, Seaborn  

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

2. Raw data was provided by GridCure and is deliberately excluded from this repository. Interim data structures and final models are available on request. 
    
3. Part A data processing/transformation scripts are kept [here](https://github.com/AndreaKeane/electric-vehicle-detection/tree/master/part_a)  

4. Part B data processing/transformation scripts are kept [here](https://github.com/AndreaKeane/electric-vehicle-detection/tree/master/part_b)

5. Applying the model on a final test set and results are kept [here](https://github.com/AndreaKeane/electric-vehicle-detection/tree/master/final_test)

6. The report with full model development process and analysis is in [here](https://github.com/AndreaKeane/electric-vehicle-detection/blob/master/report.pdf)

# Credits
Raw data and original project requirements provided by GridCure.

