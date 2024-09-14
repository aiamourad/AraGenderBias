# Arabic Gender Bias Statistics

This repository contains statistics on gender bias in Arabic word embeddings for various occupations using Direct Bias and WEAT tests across different years and datasets.

## Data Description

### Direct Bias 
The data of the occupation bias is provided in CSV format with the following columns:

- `projection`: The bias projection value
- `year`: The year of the data (for time series datasets)
- `female occupation`: The occupation term for females in Arabic
- `male occupation`: The occupation term for males in Arabic
- `male_bias`: The bias value towards males
- `female_bias`: The bias value towards females
- `occupation_combined`: The combined occupation term (female-male) in Arabic
- `reshaped_labels`: The combined occupation term in reshaped Arabic text
- `color`: An RGBA color value representing the bias visually

#### Bias Projection Calculation

The projection (Direct Bias) is calculated using the following equation:
$\begin{eqnarray} 
DirectBias= |cos(\overrightarrow{w_m}, \vec{d_g})|- |cos(\overrightarrow{w_f}, \vec{d_g})|
\end{eqnarray}$

A positive value indicates bias towards males, while a negative value indicates bias towards females.

### WEAT Tests 
The differential association `s` between the target sets and the attribute sets is calculated as follows:
\begin{eqnarray} 
s(X, Y, A, B)
= \sum\limits_{\vec x \in X}{s(\vec x, A, B)} - \sum\limits_{\vec y \in Y}{s(\vec y, A, B)}
\end{eqnarray}

And the Cohen's effect size `d` as follows: 

\begin{eqnarray} 
\mathnormal{d}= \frac{mean_{\vec x \in X}{s(\vec x, A, B)} - mean_{\vec y \in Y}{s(\vec y, A, B)}}{std-dev_{w \in X \cup Y}s(w, A, B)}
\end{eqnarray}

This statistics file contains results from various Word Embedding Association Tests (WEAT) applied to non-disentangled and disentangled word embeddings. 
The dataset consists of: 
- `s`: A float representing a specific test's association strength.
- `d`: A float representing the effect size statistic.
- `p`: A float representing the p-value, indicating statistical significance.
- `test`: A string describing the word sets used in each WEAT test (e.g., flower-insect, career-family).

## Datasets

The statistics are derived from multiple Arabic corpora:

1. Lebanese news archives (Annahar and Assafir newspapers)
2. Arabic Wikipedia
3. Electronic newspapers (including UAN and MNAD)

## Methodology

The bias values were calculated using adapted versions of the Direct Bias method and Word Embedding Association Test (WEAT), tailored for the Arabic language. The data also includes results after grammatical gender disentanglement.
