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

### WEAT Tests 

This statistics file contains results from various Word Embedding Association Tests (WEAT) applied to non-disentangled and disentangled word embeddings. 
The dataset consists of: 
- `s`: A float representing a specific test's association strength.
- `d`: A float representing the Cohen's effect size statistic.
- `p`: A float representing the p-value, indicating statistical significance.
- `test`: A string describing the word sets used in each WEAT test (e.g., flower-insect, career-family).

## Datasets

The statistics are derived from multiple Arabic corpora:

1. Lebanese news archives (Annahar and Assafir newspapers)
2. Arabic Wikipedia
3. Electronic newspapers (including UAN and MNAD)

## Methodology

The bias values were calculated using adapted versions of the Direct Bias method and Word Embedding Association Test (WEAT), tailored for the Arabic language. The data also includes results after grammatical gender disentanglement.
