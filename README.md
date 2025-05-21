**Movie Recommender System**

A Python-based hybrid movie recommendation engine leveraging both content-based and collaborative-filtering techniques on the MovieLens 20M dataset. This repository contains the end-to-end data processing, model building, evaluation, and packaging pipeline for generating top-N movie recommendations.

---

## Features

* **Data Preprocessing**: Cleansed 20M user ratings and enriched with metadata (titles, genres, tags); engineered TF-IDF features for content similarity.
* **Content-Based Filtering**: Utilizes TF-IDF vectorization of metadata reduced via TruncatedSVD to compute cosine-similarity recommendations.
* **Collaborative Filtering**: Builds latent-factor models (TruncatedSVD and Surprise SVD) on the user–item rating matrix to uncover patterns and similarities.
* **Hybrid Recommendations**: Combines content and collaborative similarity scores through weighted averaging to provide diverse suggestions.
* **Evaluation & Tuning**: Evaluates model performance using RMSE on a 75/25 train-test split and fine-tunes hyperparameters.
* **Reusable Pipeline**: Exposes a single Python function that accepts any movie title and returns the top 10 recommendations.

---

## Prerequisites

* Python 3.8 or higher
* pip (Python package installer)

## Installation

1. **Clone the repository**

   ```bash
   git clone <REPOSITORY_LINK>
   cd movie-recommender
   ```
2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # on macOS/Linux
   venv\Scripts\activate     # on Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Download the MovieLens 20M dataset** from [https://grouplens.org/datasets/movielens/20m/](https://grouplens.org/datasets/movielens/20m/) and place the files in a `data/` directory at the project root.
2. **Run the preprocessing and model pipeline**:

   ```bash
   python run_recommender.py --data-dir data/ --output-dir models/
   ```
3. **Generate recommendations**:

   ```python
   from recommender import get_top_recommendations

   recommendations = get_top_recommendations("The Shawshank Redemption (1994)")
   print(recommendations)
   ```

---

## Repository Structure

```
├── data/                # Raw and processed dataset files
├── models/              # Saved model artifacts
├── recommender.py       # Core functions (preprocessing, modeling, inference)
├── run_recommender.py   # Script to execute end-to-end pipeline
├── requirements.txt     # Python dependencies
└── README.md            # Project overview and instructions
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ for data-driven movie discovery.*
