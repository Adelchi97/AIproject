FOLDER CONTAINING USED DATA SETS

Although all data sets are available online I prepared this folder with the csv files for each one.
They all have the same format: class is in the first column and all attributes follow. All data are numbers,
and if the original file contained strings they were encoded as labels.
Each sub-folder contains a dataset, as mentioned, and its description.

Here is where I obtained the data sets:
Sonar:
    https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar,+Mines+vs.+Rocks%29
Glass:
    https://archive.ics.uci.edu/ml/datasets/glass+identification
Flag:
    https://archive.ics.uci.edu/ml/datasets/Flags
Lymphography:
    https://archive.ics.uci.edu/ml/datasets/Lymphography
Ecoli:
    https://archive.ics.uci.edu/ml/datasets/ecoli
Pima:
    https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
Ionosphere:
    https://archive.ics.uci.edu/ml/datasets/ionosphere
Contraceptive:
    https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
Breast Cancer:
    https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)
Adult:
    https://archive.ics.uci.edu/ml/datasets/adult


The python file dataset.py contains a function that I used to create the csv files for every dataset and a function getDataset() that returns a csv. 
