# Customer Personality Analysis: Clustering Project

## Overview
This project implements and compares two clustering techniques (k-Means and Agglomerative Hierarchical Clustering) on customer personality analysis data. The goal is to identify optimal customer segments using clustering algorithms and evaluate their performance.

## Dataset
The analysis uses the [Customer Personality Analysis dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data) from Kaggle, containing 2240 records with 29 attributes including:

- Demographics (Year of Birth, Education, Marital Status, Income)
- Household information (Kids, Teens)
- Customer purchasing behavior
- Campaign responses
- Product preferences

## Methods
The clustering analysis implements two primary methods:

### 1. k-Means Clustering
- Testing multiple cluster counts (k=2 to k=10)
- Determining optimal cluster count through Silhouette Score evaluation
- Preprocessing with standardization of numeric features

### 2. Agglomerative Hierarchical Clustering
Four different linkage methods were evaluated:
- **Single-linkage**: Minimum distance between clusters
- **Complete-linkage**: Maximum distance between clusters
- **Average-linkage**: Average distance between all pairs of objects
- **Ward linkage**: Minimizes variance within clusters

## Results

### k-Means Analysis
| Number of Clusters (k) | Silhouette Score |
|------------------------|------------------|
| 2                      | 0.2532           |
| 3                      | 0.2241           |
| 4                      | 0.1799           |
| 5                      | 0.1048           |
| 6                      | 0.0921           |
| 7                      | 0.0988           |
| 8                      | 0.1050           |
| 9                      | 0.1129           |
| 10                     | 0.1246           |

**Optimal k = 2** with highest Silhouette Score (0.2532)

### Hierarchical Clustering Results
| Linkage Method | Silhouette Score |
|----------------|------------------|
| Single         | 0.7440           |
| Complete       | 0.7440           |
| Average        | 0.7440           |
| Ward           | 0.2112           |

## Key Findings

1. **Optimal Clustering**: The dataset naturally splits into 2 distinct clusters.

2. **Method Comparison**:
   - **Agglomerative clustering** (Single, Complete, and Average linkage) significantly outperformed k-Means with Silhouette Scores of 0.7440.
   - **Ward linkage** performed slightly worse than k-Means (0.2112 vs 0.2532).

3. **Interpretation**:
   - High Silhouette Scores (close to 1) for Single, Complete, and Average linkage indicate very well-defined and separated clusters.
   - The comparable scores across these three linkage methods suggest a robust natural clustering structure.

## Implementation

The project uses Python with the following libraries:
- scikit-learn for clustering algorithms
- pandas for data manipulation
- numpy for numerical operations
- matplotlib for visualization

## Data Preprocessing
- Selection of numeric features
- Handling missing values using median imputation
- Standardization of features using StandardScaler
