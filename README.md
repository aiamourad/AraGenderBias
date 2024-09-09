# Detecting Gender Bias in Arabic Text through Word Embeddings

## Abstract
For generations, women have battled to attain equal rights with those of men. Many historians and social scientists examined this uphill trajectory with a focus on women’s rights and economic status in the West. Other parts of the world, such as the Middle East, remain understudied, with a noticeable shortage in gender-based statistics in the economic arena. In this study, we consider a complementary angle by which to examine gender-based biases in various occupations, as reflected through various textual corpora. Research has shown that word embedding models can learn biases from their training textual data, reflecting societal prejudices. In our study, we adapt WEAT and Direct Bias quantification tests for Arabic, to examine gender bias with respect to a wide set of occupations in various Arabic text datasets. These datasets include the Lebanese news archives, Arabic Wikipedia, and electronic newspapers in UAE, Egypt, and Morocco, thus providing different outlooks into female and male engagements in various professions. Our WEAT tests across all datasets indicate that words related to careers, science, and intellectual pursuits are linked to men. In contrast, words related to family and art are associated with women across all datasets. The Direct Bias analysis shows a consistent female gender bias towards professions such as a nurse, house cleaner, maid, secretary, and dancer. As the Moroccan News Articles Dataset (MNAD) showed, the females were also involved in other occupations such as researcher, doctor, and professor. Considering that the Arab world remains short on census data exploring gender-based disparities across various professions, our work provides evidence that such stereotypes persist till this day.

![Schematic overview of the data and methodology used for quantifying
gender bias. The 300 dimensions of the word embedding are represented in 3D for
diagram simplicity](https://github.com/aiamourad/AraGenderBias/blob/main/figures/Methodology.png?raw=true)

## Project Structure

- `GenderPCA.py`: Contains the `GenderPCA` class for performing Principal Component Analysis.
- `Occupation Bias.ipynb`: Jupyter notebook for computing and visualizing occupation-related gender bias, the statistical output will be saved as the following `output/occupation_notdisentangled.csv` and `output/occupation_disentangled.csv` (occupation bias quantification after removing the grammatical component)  .
- `WEAT Stimuli.ipynb`: Jupyter notebook for running WEAT tests on various word categories, the statistical output will be saved as the following `output/weat_notdisentangled.csv` and `output/weat_disentangled.csv` (WEAT bias quantification after removing the grammatical component) .
- `weat.py`: Python script containing the `weat` class for performing WEAT calculations.
- `nouns/`: Contains files with Arabic masculine and feminine nouns used for grammatical gender identification.
- `models/`: Directory for storing word embedding models (not included in the repository).
- `output/`: Directory for storing output files and results.
- `paper_stats/`: Folder containing statistical results presented in the paper.

## Setup and Dependencies

To run the code in this repository, you'll need:

1. Python 3.9 or higher
2. Required Python packages:
You can install the required packages using pip:

```
pip install numpy pandas matplotlib seaborn scikit-learn gensim mlxtend arabic_reshaper python-bidi
```
or 
```
pip install -r requirements.txt
```

## Usage
To use this code with your own models and data: 
1. Place your word embedding models in the `models/` directory
2. Run the Jupyter notebooks:
   - `Occupation Bias.ipynb` for occupation-related bias analysis
   - `WEAT Stimuli.ipynb` for WEAT tests
   - `plots.ipynb` for generating the plots
3. The statistical results will be saved in the `output/` directory and the images in the `figures/` directory

## Paper Statistics

The `paper_stats/` folder contains the statistical results used in the associated research paper. 
Researchers interested in replicating or extending this work should refer to this folder for detailed information on the study's findings.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this code or the results in your research, please cite our paper:

[Paper citation to be added]

