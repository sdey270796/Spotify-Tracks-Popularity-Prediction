# 🎵 Spotify Track Popularity — Clustering, Classification & Deep Learning

> **An end-to-end Data Science capstone project exploring Spotify tracks through Exploratory Data Analysis, Feature Engineering, Unsupervised Learning, Supervised Learning, Deep Learning, Hyperparameter Optimization, and Explainable AI.**

---

## 📌 About the Project

This repository contains my **Master Project completed during the 2nd Semester of my Postgraduate program in Data Science**.

The project began with a simple question:

> **Can we predict the popularity of a Spotify track from its audio characteristics, metadata, and other observable properties?**

However, rather than treating this as a straightforward classification problem, the project was developed as a **complete Data Science pipeline**.

The analysis therefore progresses through several stages:

```text
Raw Spotify Dataset
        ↓
Data Cleaning & Validation
        ↓
Domain-Aware Feature Engineering
        ↓
Extensive Exploratory Data Analysis
        ↓
Statistical Analysis
        ↓
Feature Selection
        ↓
Unsupervised Learning
        ├── K-Means
        └── DBSCAN
        ↓
Cluster-Based Target Construction
        ↓
Supervised Classification
        ├── Decision Tree
        ├── Random Forest
        ├── CatBoost
        ├── XGBoost
        ├── SVM
        └── Logistic Regression
        ↓
Deep Learning
        └── Artificial Neural Network
        ↓
Cross-Validation & Model Evaluation
        ↓
Hyperparameter Optimization
        ↓
Best Model Selection
        ↓
Feature Importance
        ↓
LIME Explainability
        ↓
Saved Models
````

The repository therefore demonstrates not just **how to train a model**, but how to approach a relatively large, messy, real-world-style dataset as a complete Data Science problem.

---

# 👨‍💻 Project Information

**Project Type:** Master / Capstone Project
**Program:** Postgraduate Data Science
**Semester:** 2nd Semester
**Domain:** Music Analytics / Data Science
**Primary Task:** Spotify Track Popularity Analysis & Classification
**Dataset Size:** 114,000 original tracks
**Final Modeling Dataset:** 112,114 observations
**Learning Paradigms:** Unsupervised + Supervised + Deep Learning
**Explainability:** Feature Importance + LIME

---

# ⭐ Repository Highlights

This repository contains the complete ecosystem around the project:

* 📓 **Extensive master notebook**
* 📊 **Master dataset**
* 📈 **EDA outputs**
* 🧹 **Processed data**
* 🤖 **Best classical model**
* 🧠 **Trained deep-learning model**
* 🔍 **Model explainability analysis**
* 📑 **Complete project documentation**

It is therefore intended to be viewed not merely as a notebook, but as a **full Data Science project repository**.

---

# 🎯 Problem Statement

The original objective of the project was to investigate whether the popularity of Spotify tracks could be predicted from their available metadata and audio characteristics.

Spotify tracks contain a wide variety of attributes describing their musical properties, including:

* Danceability
* Energy
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo
* Duration
* Key
* Mode
* Time Signature
* Genre
* Artist
* Album
* Track-related information

The central question was:

> **Can we predict the popularity of a Spotify track using its metadata and audio characteristics?**

The project also investigates a broader question:

> **Can unsupervised learning reveal meaningful groups of tracks, and can those groups subsequently be transformed into a useful prediction target?**

---

# 💼 Business Motivation

Track popularity is an important consideration for several stakeholders in the music ecosystem.

## 🎤 Artists

Popularity prediction could potentially help artists and their teams understand which characteristics of their music are associated with greater audience engagement.

Possible applications include:

* Track development
* Release planning
* Promotional strategy
* Audience targeting
* Content strategy

---

## 💿 Record Labels

Record labels could potentially use popularity estimates to support:

* Talent scouting
* A&R decisions
* Marketing allocation
* Release strategy
* Promotional investment

---

## 🎧 Streaming Platforms

Streaming platforms could potentially use popularity-related predictions for:

* Playlist curation
* Content prioritization
* Recommendation systems
* User engagement
* Retention strategies

---

## 📢 Marketing

Predictive insights could potentially help allocate promotional resources toward tracks with greater predicted potential.

The project therefore frames Machine Learning not merely as an academic exercise, but as a potential **decision-support system for the music industry**.

---

# 📊 Dataset

The project starts with a Spotify dataset containing:

> **114,000 tracks**

and:

> **21 original columns**

The original dataset contains a mixture of:

* Numerical variables
* Categorical variables
* Boolean variables
* Textual metadata

The dataset initially occupies approximately:

> **17.5 MB**

in memory.

---

# 🧾 Original Dataset Features

The original dataset includes variables such as:

| Feature            | Type                  | Description                                  |
| ------------------ | --------------------- | -------------------------------------------- |
| `track_id`         | Categorical           | Unique track identifier                      |
| `artists`          | Categorical           | Artist information                           |
| `album_name`       | Categorical           | Album information                            |
| `track_name`       | Categorical           | Track title                                  |
| `popularity`       | Numerical             | Spotify popularity score                     |
| `duration_ms`      | Numerical             | Track duration                               |
| `explicit`         | Boolean               | Whether the track is explicit                |
| `danceability`     | Numerical             | Danceability score                           |
| `energy`           | Numerical             | Energy score                                 |
| `key`              | Numerical/Categorical | Musical key                                  |
| `loudness`         | Numerical             | Overall loudness                             |
| `mode`             | Numerical/Categorical | Major/minor mode                             |
| `speechiness`      | Numerical             | Presence of spoken words                     |
| `acousticness`     | Numerical             | Acoustic confidence                          |
| `instrumentalness` | Numerical             | Likelihood of instrumental content           |
| `liveness`         | Numerical             | Presence of live-performance characteristics |
| `valence`          | Numerical             | Musical positivity                           |
| `tempo`            | Numerical             | Estimated tempo                              |
| `time_signature`   | Numerical/Categorical | Estimated time signature                     |
| `track_genre`      | Categorical           | Track genre                                  |

---

# 🧹 Data Cleaning

The project does not immediately feed the raw dataset into a Machine Learning model.

Instead, the data goes through a substantial cleaning and validation process.

---

## 🎙️ Removing Non-Music Tracks

Tracks with extremely high speechiness are filtered out.

The project uses:

```python
df_1 = df[df['speechiness'] < 0.8]
```

This reduces the dataset from:

```text
114,000
   ↓
113,198
```

records.

This is an example of **domain-aware preprocessing** rather than blindly applying generic cleaning techniques.

---

# 🔍 Missing-Value Analysis

The original dataset contains a small number of missing values.

The initial inspection identifies missing values in:

* `artists`
* `album_name`
* `track_name`

The project subsequently performs additional cleaning to ensure the modeling dataset is complete.

---

# 🎼 Musical Feature Engineering

One of the distinguishing characteristics of this project is that feature engineering is not limited to generic transformations.

The project incorporates **music-specific knowledge**.

---

## 🎹 `mode`

The numerical representation of mode is converted into:

```text
0 → minor
1 → major
```

This makes the variable more interpretable.

---

## 🎵 `key`

The numerical key representation is transformed into musical key names.

For example:

```text
0  → C
1  → C♯ / D♭
2  → D
3  → D♯ / E♭
...
11 → B
```

This converts an abstract integer representation into a musically meaningful categorical feature.

---

# 🥁 Time Signature Validation

The project checks the validity of the `time_signature` variable.

The expected range is treated as:

```text
3 → 7
```

The analysis finds:

> **0.96% of values below 3**

and:

> **0.00% above 7**

Values below the valid range are removed.

The remaining values are converted into interpretable musical representations:

```text
3 → 3/4
4 → 4/4
5 → 5/4
6 → 6/4
7 → 7/4
```

---

# ⏱️ Duration Validation

Tracks with:

```text
duration_ms = 0
```

are removed.

This prevents clearly invalid duration values from contaminating subsequent analysis.

---

# 📝 Text-Based Feature Engineering

The project does not completely discard textual metadata.

Instead, it extracts information from:

* Track names
* Album names
* Artist names

---

## 🔤 Track Name Features

The project performs text cleaning and frequency analysis on track names.

The most frequent relevant words are identified and converted into binary indicator features.

Examples include features such as:

```text
The_track
You_track
Me_track
I_track
Vivo_track
Others_track
```

These become Boolean features.

---

# 💿 Album Name Features

A similar process is applied to album names.

The most frequent words are identified and transformed into binary features.

Examples include:

```text
The_album
Vol_album
Christmas_album
Others_album
```

This allows textual metadata to contribute to the Machine Learning pipeline without directly feeding raw text into the models.

---

# 🎤 Artist Features

The five most frequent artists are identified.

The project finds:

```text
The Beatles
George Jones
Stevie Wonder
Ella Fitzgerald
Linkin Park
```

These are converted into binary indicator variables.

An additional:

```text
artist_Others
```

feature captures all remaining artists.

This is a form of **domain-specific categorical feature engineering**.

---

# 📉 Outlier Analysis

The project performs an extensive investigation of numerical outliers.

Several approaches are explored, including:

* Z-score analysis
* Boxplots
* IQR-based analysis
* Outlier percentage calculations
* Yeo-Johnson transformation analysis

The project does not simply remove every statistical outlier.

Instead, the distributions are examined to understand whether extreme values represent:

* genuine observations,
* skewed distributions,
* or potentially problematic measurements.

---

# 📐 Statistical Distribution Analysis

The project conducts extensive EDA on the numerical variables.

This includes:

* Distribution plots
* Histograms
* Boxplots
* Statistical summaries
* Normality testing
* Transformation analysis

---

# 🧪 Normality Testing

Several statistical approaches are considered, including:

* Kolmogorov-Smirnov test
* Shapiro-Wilk test
* Jarque-Bera test
* Normality-related diagnostics

The project also explores transformations such as:

> **Yeo-Johnson transformation**

to investigate whether heavily skewed variables can be represented more appropriately.

---

# 📈 Popularity Analysis

A major component of the EDA is the investigation of:

> `popularity`

and its relationship with the available audio and metadata features.

The project examines:

* Popularity distribution
* Feature vs popularity relationships
* Categorical feature distributions
* Correlation structure
* Potential predictors of popularity

---

# 🔗 Feature Selection

After EDA and feature engineering, the project performs feature-selection analysis.

A correlation matrix is constructed to identify relationships between variables.

Highly correlated / auto-correlated features are investigated and removed where appropriate.

This results in a more compact numerical representation before clustering.

---

# 🧩 Unsupervised Learning

One of the most interesting aspects of the project is that **clustering is performed before the final classification target is created**.

Rather than simply choosing arbitrary popularity bins, the project explores whether the tracks can naturally be grouped according to their feature space.

Two clustering approaches are investigated:

```text
K-Means
DBSCAN
```

---

# 🔵 K-Means Clustering

The project first investigates K-Means clustering.

The number of clusters is explored using:

* Within-Cluster Sum of Squares (WCSS)
* Elbow Method
* Silhouette Score
* Silhouette Visualization

The project evaluates multiple candidate cluster counts, including:

```text
K = 5
K = 6
K = 7
K = 8
```

---

# 📊 Silhouette Analysis

The average silhouette scores obtained are:

| Number of Clusters | Silhouette Score |
| -----------------: | ---------------: |
|                  5 |           0.1255 |
|                  6 |       **0.1381** |
|                  7 |           0.1308 |
|                  8 |           0.1277 |

Among the tested configurations:

> **K = 6 provides the highest silhouette score.**

The project therefore selects:

> **6 clusters**

for the K-Means solution.

---

# 🧭 Cluster Analysis

The selected K-Means model is used to assign a:

```text
cluster_id
```

to each track.

Cluster centroids are subsequently examined to understand the characteristics of the resulting groups.

The project also visualizes the distributions associated with the clusters.

---

# 🌐 DBSCAN Clustering

The project then explores a second unsupervised learning technique:

> **DBSCAN — Density-Based Spatial Clustering of Applications with Noise**

The analysis includes:

* Pairwise-distance analysis
* Distance distributions
* k-distance plot
* Nearest-neighbor analysis
* `eps` investigation
* `min_samples` selection

A sample of:

> **5,000 observations**

is used during part of the distance-analysis stage because computing pairwise distances over the complete dataset is computationally expensive.

---

# 🔬 DBSCAN Results

The DBSCAN configuration used in the final clustering stage is:

```python
DBSCAN(
    eps=0.3,
    min_samples=5,
    metric='euclidean'
)
```

The resulting clustering is highly fragmented:

> **1,133 cluster labels**

are produced, with a very large proportion of observations initially identified as noise (`-1`).

The project subsequently treats the noise group as an additional cluster/category for the next stage.

This result itself is informative:

> **The Spotify feature space does not necessarily form a small number of compact, uniformly dense clusters under the chosen DBSCAN configuration.**

---

# 🎯 Creating the Prediction Target

This is one of the most unusual aspects of the project.

Instead of directly using the original `popularity` score as the classification target, the project constructs a new target based on the DBSCAN clustering.

First, popularity is scaled to:

```text
0 → 100
```

Then:

```text
average_popularity
```

is calculated for each DBSCAN cluster.

Every track receives the average popularity of the cluster to which it belongs.

The resulting target contains:

> **65 unique popularity classes**

This transforms the project into a multiclass classification problem.

---

# 🧠 Why This Approach?

The project therefore investigates a two-stage idea:

```text
Track Characteristics
        ↓
Unsupervised Clustering
        ↓
Groups of Similar Tracks
        ↓
Average Popularity of Each Group
        ↓
Popularity-Class Target
        ↓
Supervised Learning
```

This creates a bridge between:

> **Unsupervised Learning → Supervised Learning**

rather than treating clustering and classification as completely independent tasks.

---

# 📦 Final Modeling Dataset

The final modeling dataset contains:

> **112,114 observations**

and:

> **31 columns**

before one-hot encoding.

The modeling features include:

### Numerical

* Duration
* Danceability
* Loudness
* Speechiness
* Acousticness
* Instrumentalness
* Liveness
* Valence
* Tempo

### Categorical

* Key
* Mode
* Time Signature
* Track Genre

### Boolean / engineered

* Explicit
* Track-name indicators
* Album-name indicators
* Artist indicators

### Target

```text
average_popularity
```

---

# 🤖 Supervised Learning

Once the target has been constructed, several classification algorithms are evaluated.

The project does not assume that one algorithm will automatically be superior.

Instead, multiple model families are compared.

---

# 🌳 1. Decision Tree

The first supervised baseline is:

> **Decision Tree Classifier**

The initial model achieves approximately:

| Metric           |      Score |
| ---------------- | ---------: |
| Cross-Validation | **0.9362** |
| Accuracy         | **0.9414** |
| Weighted F1      | **0.9438** |
| Cohen's Kappa    | **0.7119** |

This establishes a strong baseline.

---

# 🧠 2. Artificial Neural Network

A feed-forward neural network is also implemented using TensorFlow/Keras.

The architecture is:

```text
Input
  │
  ▼
Dense(128, ReLU)
  │
Dropout(0.20)
  │
Dense(64, ReLU)
  │
Dropout(0.20)
  │
Dense(65, Softmax)
  │
  ▼
Prediction
```

The model contains approximately:

> **98,117 total parameters**

with:

> **32,705 trainable parameters**

in the reported Keras configuration.

---

# 📈 Neural Network Performance

The neural network achieves:

| Metric                  |     Result |
| ----------------------- | ---------: |
| 5-Fold Cross-Validation | **0.9580** |
| Test Accuracy           | **0.9756** |
| Weighted F1             | **0.9177** |
| Cohen's Kappa           | **0.5120** |

The neural network therefore achieves the highest raw test accuracy among the principal models evaluated in the notebook.

However, the weighted F1 and Cohen's Kappa reveal a more nuanced picture, particularly because the target classes are highly imbalanced.

---

# 🌲 3. Random Forest

A Random Forest classifier is also evaluated.

Its 5-fold cross-validation scores are approximately:

```text
0.9708
0.9679
0.9708
0.9694
0.9695
```

with an average around:

> **0.9697**

The Random Forest performs strongly across the dominant and moderately represented classes.

---

# 🐱 4. CatBoost

The project also evaluates:

> **CatBoost Classifier**

CatBoost performs strongly across many of the less frequent classes compared with several other models.

This makes it a useful comparison against Random Forest and the neural network.

---

# 🚀 5. XGBoost

An XGBoost-based classifier is also implemented.

This provides a comparison with another highly popular gradient-boosting framework.

---

# 📐 6. Support Vector Machine

The project evaluates an:

> **SVM / SVC**

classifier.

The model obtains:

> **90.78% test accuracy**

with an average 5-fold cross-validation performance of approximately:

> **90.66%**

---

# 📊 7. Logistic Regression

As a simpler linear baseline, the project also implements:

> **Logistic Regression**

The model achieves approximately:

> **90.43% test accuracy**

with a 5-fold cross-validation average of approximately:

> **90.39%**

---

# 🏆 Model Comparison

The project therefore compares a broad spectrum of model families:

| Model               | Model Family      |
| ------------------- | ----------------- |
| Decision Tree       | Tree-based        |
| Random Forest       | Ensemble          |
| CatBoost            | Gradient Boosting |
| XGBoost             | Gradient Boosting |
| SVM                 | Kernel-based      |
| Logistic Regression | Linear            |
| Neural Network      | Deep Learning     |

This makes the project substantially more comprehensive than a single-model Machine Learning exercise.

---

# 🔧 Hyperparameter Optimization

After evaluating multiple models, the project performs automated model selection using:

> **Optuna**

The optimization process considers three candidate model families:

```text
Random Forest
CatBoost
Decision Tree
```

and searches over relevant hyperparameters.

---

# 🔍 Hyperparameters Explored

## Random Forest

The search includes:

* Number of estimators
* Maximum depth
* Minimum samples split
* Minimum samples leaf

---

## CatBoost

The search includes:

* Number of iterations
* Tree depth
* Learning rate

---

## Decision Tree

The search includes:

* Maximum depth
* Minimum samples split
* Minimum samples leaf

---

# 🏅 Best Optimized Classical Model

The Optuna search identifies:

> **Decision Tree**

as the best configuration among the models considered in the optimization study.

The selected hyperparameters are:

```text
max_depth = 15
min_samples_split = 8
min_samples_leaf = 6
```

The best Optuna trial obtains:

> **Weighted F1 = 0.89945**

---

# 📊 Final Optimized Model Performance

The selected optimized Decision Tree achieves:

| Metric        |      Score |
| ------------- | ---------: |
| Accuracy      | **0.9175** |
| Weighted F1   | **0.8995** |
| Cohen's Kappa | **0.3873** |

The detailed classification report is also retained in the notebook.

---

# 🧠 Why Accuracy Alone Is Not Enough

One of the important lessons of this project is that:

> **High accuracy does not necessarily imply good multiclass performance.**

The final target is highly imbalanced.

For example, one class contains more than:

> **20,000 observations**

while many other classes contain only a handful.

Consequently, a model can achieve strong overall accuracy while performing poorly on rare classes.

This is why the project evaluates:

* Accuracy
* F1 Score
* Cohen's Kappa
* Cross-Validation
* Class-level precision
* Class-level recall
* Class-level F1

rather than relying exclusively on accuracy.

---

# 🔍 Explainable AI

The project goes beyond prediction.

After selecting the best classical model, it investigates:

> **Why does the model make its predictions?**

Two important explainability approaches are explored.

---

# 🌟 Feature Importance

For the selected Decision Tree, feature importance is extracted using:

```python
feature_importances_
```

The project visualizes:

* Top 20 features
* Features exceeding an importance threshold

This provides an interpretable view of which variables contribute most strongly to the model's decision process.

---

# 🔬 LIME

The project additionally uses:

> **LIME — Local Interpretable Model-Agnostic Explanations**

A `LimeTabularExplainer` is constructed to explain individual predictions.

The explanation identifies the features that most strongly influence a particular prediction.

This transforms the project from:

```text
"Here is the prediction."
```

into:

```text
"Here is the prediction,
and here are the features that influenced it."
```

---

# 💾 Model Artifacts

The repository contains not only the notebook, but also the artifacts required to reproduce and inspect the work.

The repository includes:

```text
Master Dataset
        +
EDA Outputs / Visualizations
        +
Notebook
        +
Best Classical Model
        +
Deep Learning Model
```

The trained neural network is saved as:

```text
spotify_neural_network_model.h5
```

The repository also contains the selected/best classical model in serialized form.

---

# 📁 Repository Structure

A simplified representation of the repository is:

```text
Spotify-Popularity-Classification/
│
├── 📓 spotify_clustering_classification_prediction.ipynb
│
├── 📊 Master Dataset
│   └── Spotify Dataset.csv
│
├── 📈 EDA/
│   ├── Distribution Plots
│   ├── Correlation Analysis
│   ├── Outlier Analysis
│   ├── Normality Analysis
│   ├── Cluster Analysis
│   └── Other Visualizations
│
├── 🤖 Models/
│   ├── Best Classical Model
│   └── spotify_neural_network_model.h5
│
└── 📄 README.md
```

> File names may vary slightly depending on the repository version; the repository itself contains the corresponding model and analysis artifacts.

---

# 🛠️ Technology Stack

The project uses a broad Data Science ecosystem.

## Programming

* Python

## Data Manipulation

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn
* Plotly

## Statistical Analysis

* SciPy
* Statsmodels

## Machine Learning

* Scikit-learn
* XGBoost
* CatBoost
* Yellowbrick

## Clustering

* K-Means
* DBSCAN

## Deep Learning

* TensorFlow
* Keras

## Hyperparameter Optimization

* Optuna / automated hyperparameter search

## Explainable AI

* LIME

---

# 📦 Installation

The major dependencies can be installed using:

```bash
pip install pandas numpy
pip install matplotlib seaborn plotly
pip install scipy statsmodels
pip install scikit-learn
pip install tensorflow
pip install xgboost catboost
pip install yellowbrick
pip install optuna
pip install lime
```

---

# ▶️ Running the Project

## 1. Clone the Repository

```bash
git clone <repository-url>
cd Spotify-Popularity-Classification
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Open the Notebook

Open:

```text
spotify_clustering_classification_prediction.ipynb
```

using:

* Jupyter Notebook
* JupyterLab
* Google Colab

## 4. Dataset

The notebook loads the Spotify dataset from the repository.

The original workflow uses:

```python
pd.read_csv(...)
```

to load:

```text
Spotify Dataset.csv
```

---

# 🧭 Project Workflow in Detail

The entire project can be summarized as:

```text
                         SPOTIFY DATA
                              │
                              ▼
                    Data Acquisition
                              │
                              ▼
                    Data Validation
                              │
                              ▼
                    Data Cleaning
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        Numerical Data                  Text Metadata
              │                               │
              │                    Track / Album / Artist
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    Feature Engineering
                              │
                              ▼
                    Outlier Analysis
                              │
                              ▼
                         EDA
                              │
                              ▼
                    Statistical Testing
                              │
                              ▼
                     Feature Selection
                              │
                              ▼
                      Feature Scaling
                              │
                              ▼
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
                 K-Means             DBSCAN
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     Cluster Analysis
                              │
                              ▼
                  Cluster-Based Target
                              │
                              ▼
                   Multiclass Dataset
                              │
         ┌────────────────────┼────────────────────┐
         │          │         │         │          │
         ▼          ▼         ▼         ▼          ▼
       DT         RF       CatBoost   XGBoost     SVM
         │          │         │         │          │
         └──────────┴─────────┼─────────┴──────────┘
                              │
                              ▼
                    Model Comparison
                              │
                              ▼
                     Deep Learning
                              │
                              ▼
                    Hyperparameter Tuning
                              │
                              ▼
                       Best Model
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
       Feature Importance                    LIME
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                     Explainable Results
```

---

# 🧪 Major Experiments

The project can broadly be divided into six major experimental blocks.

### Phase 1 — Data Understanding

* Dataset inspection
* Missing-value analysis
* Data types
* Feature categorization
* Domain validation

### Phase 2 — Feature Engineering

* Musical key conversion
* Major/minor conversion
* Time-signature transformation
* Text preprocessing
* Track-name features
* Album-name features
* Artist features
* Outlier treatment

### Phase 3 — Exploratory Data Analysis

* Univariate analysis
* Bivariate analysis
* Multivariate analysis
* Correlation analysis
* Normality testing
* Transformation analysis
* Popularity analysis

### Phase 4 — Unsupervised Learning

* K-Means
* Elbow Method
* Silhouette Analysis
* Cluster profiling
* DBSCAN
* k-distance analysis
* Density-based clustering

### Phase 5 — Supervised Learning

* Decision Tree
* Random Forest
* CatBoost
* XGBoost
* SVM
* Logistic Regression
* Neural Network

### Phase 6 — Optimization & Explainability

* Cross-validation
* Hyperparameter optimization
* Best-model selection
* Feature importance
* Decision-tree visualization
* LIME explanations

---

# 📊 Why This Project Is Different

This is not simply:

```text
Dataset → Model → Accuracy
```

Instead, the project attempts to answer several layers of questions.

### Data Question

> What does the dataset actually contain?

### Statistical Question

> What are the distributions and relationships between variables?

### Domain Question

> How can musical knowledge improve the feature representation?

### Unsupervised Learning Question

> Do tracks naturally form meaningful groups?

### Predictive Question

> Can these groups be used to construct a useful prediction target?

### Algorithmic Question

> Which Machine Learning family performs best?

### Deep Learning Question

> Can a neural network outperform conventional models?

### Optimization Question

> Can hyperparameter tuning improve the classical models?

### Explainability Question

> Why does the selected model make a particular prediction?

---

# 🧠 Key Technical Lessons

This project demonstrates several important Data Science concepts.

## 1. Data Cleaning Is Not Just Removing Nulls

Cleaning involved:

* Domain validation
* Invalid musical values
* Non-music filtering
* Zero-duration tracks
* Outlier analysis
* Data-type transformation

---

## 2. Feature Engineering Can Be Domain-Specific

Rather than treating all columns as generic variables, the project uses musical knowledge to transform:

```text
mode
key
time_signature
track_name
album_name
artists
```

into more useful representations.

---

## 3. Unsupervised Learning Can Support Supervised Learning

The project uses:

```text
DBSCAN clusters
      ↓
Average cluster popularity
      ↓
Classification target
```

This creates an interesting bridge between the two major branches of Machine Learning.

---

## 4. Different Models See the Same Data Differently

The project compares:

* Linear models
* Kernel methods
* Trees
* Ensembles
* Gradient boosting
* Neural networks

This demonstrates why model selection should be empirical rather than based solely on algorithm popularity.

---

## 5. Accuracy Can Be Misleading

Because the resulting classes are highly imbalanced, accuracy alone does not adequately describe model quality.

This is why the project also examines:

```text
Precision
Recall
F1
Cohen's Kappa
Cross-Validation
```

---

## 6. Explainability Matters

A model that predicts well is useful.

A model whose predictions can also be interpreted is significantly more useful.

The project therefore includes:

```text
Feature Importance
+
Decision Tree Visualization
+
LIME
```

---

# ⚠️ Important Interpretation

The project should **not** be interpreted as a production-grade Spotify popularity prediction engine.

There are several reasons.

### Popularity Is Dynamic

Popularity changes over time and can depend on factors not contained in the dataset.

### Causality Is Not Established

A feature being important to the model does not mean it **causes** popularity.

### Dataset Bias

The dataset reflects the characteristics and collection methodology of its source.

### Class Construction

The final `average_popularity` target is derived from DBSCAN clustering rather than being a direct original Spotify label.

Therefore, the resulting classification problem should be understood as an experimental modeling framework.

---

# 🚀 Future Improvements

Several directions could take this project further.

## 🎵 1. Regression Instead of Classification

Predict the original continuous popularity score directly.

Possible models:

* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor
* Neural Network Regression

---

## 📅 2. Temporal Modeling

Popularity is inherently time-dependent.

A more advanced project could incorporate:

* Release date
* Historical popularity
* Streaming trajectory
* Time-series features

---

## 🧠 3. Advanced NLP

Instead of extracting a handful of frequent words, track and album names could be processed using:

* TF-IDF
* Word embeddings
* Transformer embeddings
* Sentiment analysis
* Topic modeling

---

## 🎧 4. Audio-Based Deep Learning

Instead of relying only on Spotify's engineered audio features, raw audio could be converted into:

* Spectrograms
* Mel-spectrograms
* MFCCs

and analyzed using CNNs or other deep-learning architectures.

---

## 🤖 5. Ensemble Learning

Combine predictions from:

```text
Random Forest
+
CatBoost
+
XGBoost
+
Neural Network
```

through:

* Voting
* Stacking
* Blending

---

## 🔬 6. Better Clustering Validation

The clustering stage could be extended with:

* Davies-Bouldin Index
* Calinski-Harabasz Index
* Gaussian Mixture Models
* Hierarchical Clustering
* HDBSCAN

---

## 🧠 7. Advanced Explainable AI

Explore:

* SHAP
* Permutation Importance
* Partial Dependence
* Counterfactual explanations

---

# 🎓 Learning Outcomes

Completing this project provides practical exposure to:

* Data acquisition
* Data cleaning
* Data validation
* Exploratory Data Analysis
* Statistical testing
* Outlier analysis
* Feature engineering
* Feature selection
* Scaling
* Dimensionality-related considerations
* K-Means clustering
* DBSCAN
* Cluster validation
* Classification
* Decision Trees
* Random Forest
* CatBoost
* XGBoost
* SVM
* Logistic Regression
* Neural Networks
* Cross-validation
* Hyperparameter optimization
* Model comparison
* Feature importance
* Explainable AI
* LIME

---

# 🏁 Final Perspective

The primary objective of this project was not simply to obtain the highest possible accuracy.

The real objective was to demonstrate the **complete thought process of a Data Scientist**.

Starting with:

> **114,000 raw Spotify tracks**

the project progressively transforms the dataset into a structured analytical problem through:

```text
Question
  ↓
Data
  ↓
Cleaning
  ↓
Domain Understanding
  ↓
Feature Engineering
  ↓
EDA
  ↓
Statistics
  ↓
Clustering
  ↓
Target Construction
  ↓
Classification
  ↓
Deep Learning
  ↓
Optimization
  ↓
Explainability
```

The project ultimately demonstrates an important principle:

> **A strong Data Science project is not defined by the model alone. It is defined by the quality of the reasoning that leads from raw data to a defensible conclusion.**

---

# ⭐ Final Note

This project was developed as my **2nd-semester Master Project in Data Science**, with the intention of bringing together as many aspects of a real Data Science workflow as possible—from understanding and cleaning the data to building, comparing, optimizing, and explaining predictive models.

> **From 114,000 songs to a complete Data Science pipeline — this project explores how data, statistics, unsupervised learning, supervised learning, and deep learning can work together to understand what makes a track popular.**

```
```
