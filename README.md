# nlp-projects
Personal projects using NLP techniques.

## Table of Contents

[Prerequisites](#prerequisites)

[Project: Pitchfork Rating Prediction](#pitchfork-rating-prediction)

---

## Prerequisites

The following project requires `python` 3.10 and `R` 4.3 to be installed.

### Dependencies (`python`)
To run the `python` code in this project, you will first need to install the relevant dependencies. This can be done by executing the following command from the project root:

```
pip install -r requirements.txt
```

Also, since this project contains a custom utilities library `myutilpy`, this must be installed to your environment. To do this, run the following command from the project root:

```
pip install -e myutilpy
```

### Dependencies (`R`)
This project contains a few `R` language `jupyter` notebooks. To execute these, your `R` environment must have the dependencies specified in `requirements_R.txt` installed. This can be done manually for each listed dependency.

## Pitchfork Rating Prediction

Directory: [`notebooks_pitchfork_ratings`](./notebooks_pitchfork_ratings/)

This sequence of notebooks utilizes a [Pitchfork](https://pitchfork.com/) reviews dataset of approximately 20K album reviews ([mattismegevand/pitchfork](https://huggingface.co/datasets/mattismegevand/pitchfork)). The notebooks cover the following steps:

1. Data preprocessing ([`01_initial_data_prep`](./notebooks_pitchfork_ratings/01_initial_data_prep.ipynb)). Loading, cleaning, and preprocessing of data.
2. Exploratory data analysis ([`02_data_explore`](./notebooks_pitchfork_ratings/02_data_explore.ipynb)). Visualization and summary statistics of processed dataset.
3. Model fitting ([`03_rating_pred`](./notebooks_pitchfork_ratings/03_rating_pred.ipynb)). Model fitting and saving of model parameters, along with performance metrics collection.
4. Results analysis ([`04_fit_analysis`](./notebooks_pitchfork_ratings/04_fit_analysis.ipynb)). Post-fit investigation of model performance on test data.
