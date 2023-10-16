# nlp-projects
Personal projects using NLP techniques.

## Table of Contents


[Pitchfork Rating Prediction](#pitchfork-rating-prediction)

---

## Pitchfork Rating Prediction

Directory: [`notebooks_pitchfork_ratings`](./notebooks_pitchfork_ratings/)

This sequence of notebooks utilizes a [Pitchfork](https://pitchfork.com/) reviews dataset of approximately 20K album reviews ([mattismegevand/pitchfork](https://huggingface.co/datasets/mattismegevand/pitchfork)). The notebooks cover the following steps:

1. Data preprocessing ([`01_initial_data_prep`](./notebooks_pitchfork_ratings/01_initial_data_prep.ipynb)). Loading, cleaning, and preprocessing of data.
2. Exploratory data analysis ([`02_data_explore`](./notebooks_pitchfork_ratings/02_data_explore.ipynb)). Visualization and summary statistics of processed dataset.
3. Model fitting ([`03_rating_pred`](./notebooks_pitchfork_ratings/03_rating_pred.ipynb)). Model fitting and saving of model parameters, along with performance metrics collection.
4. Results analysis ([04_fit_analysis](./notebooks_pitchfork_ratings/04_fit_analysis.ipynb)). Post-fit investigation of model performance on test data.
