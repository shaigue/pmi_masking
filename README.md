# pmi_masking
This repository contains code that takes a text corpus and creates a PMI masking vocabulary for it.

## Creating a PMI masking vocabulary for a specific dataset

## Creating virtual environment
Make sure you have the latest pip:
```commandline
python3 -m pip install --upgrade pip
```

Create a new environment:
```commandline
python3 -m venv env
```

Activate the virtual environment (UNIX):
```commandline
source env/Scripts/activate
```
(Windows)
```commandline
.\env\Scripts\activate
```

Install dependencies from the `requirements.txt` file:
```commandline
python3 -m pip install -r requirements.txt
```

If you wish to use the environment in a jupyter notebook, 
you should install an IPython kernel:
```commandline
python -m ipykernel install --user --name pmi_masking --display-name "Python (pmi_masking)"
```
