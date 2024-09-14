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
$DirectBias = |cos(female_occupation, gender_direction)| - |cos(male_occupation, gender_direction)|$

                                Where:
- `female_occupation` is the vector for the female version of the occupation
- `male_occupation` is the vector for the male version of the occupation
- `gender_direction` is the learned gender direction in the embedding space

A positive value indicates bias towards males, while a negative value indicates bias towards females.

## Datasets

The statistics are derived from multiple Arabic corpora:

1. Lebanese news archives (Annahar and Assafir newspapers)
2. Arabic Wikipedia
3. Electronic newspapers (including UAN and MNAD)

## Methodology

The bias values were calculated using adapted versions of the Direct Bias method and Word Embedding Association Test (WEAT), tailored for the Arabic language. The data also includes results after grammatical gender disentanglement.

## Usage

This data can be used to:

- Analyze trends in gender bias for various occupations over time
- Compare gender bias across different Arabic-language sources
- Visualize gender bias in Arabic word embeddings
- Study the effects of grammatical gender on bias measurements

## Citation

If you use this data in your research, please cite the original paper:

[Insert citation information here]

