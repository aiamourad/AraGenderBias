import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import os

class GenderPCAVisualizer:
    def __init__(self, sp1, sp2, m_occ, f_occ):
        self.sp1 = sp1
        self.sp2 = sp2
        self.m_occ = m_occ
        self.f_occ = f_occ

    def calculate_differences(self):
        differences = [np.abs(self.sp1[i][0]) - np.abs(self.sp2[i][0]) for i in range(len(self.m_occ))]
        return differences

    def visualize_histogram(self, title):
        differences = self.calculate_differences()
        fig, ax = plt.subplots(figsize=(10, 6))
        projections_df = pd.DataFrame(differences, columns=['projection'])
        projections_df['year']=title[0:4]
        print(title[0:4])
        projections_df['female occupation'] = self.f_occ
        projections_df['male occupation'] = self.m_occ
        projections_df['male_bias'] = [self.sp1[i][0] for i in range(len(self.m_occ))]
        projections_df['female_bias'] = [self.sp2[i][0] for i in range(len(self.f_occ))]

        projections_df['occupation_combined'] = projections_df['female occupation'] + '-' + projections_df[
            'male occupation']

        projections_df['reshaped_labels'] = projections_df['occupation_combined'].apply(
            lambda x: get_display(reshape(x)))

        projections_df.sort_values(by='projection', ascending=True, inplace=True)
        average_projection = projections_df['projection'].mean()
        cmap = plt.get_cmap('RdBu')
        projections_df['color'] = ((projections_df['projection'] + 0.5)
                                   .apply(cmap))

        sns.barplot(x='projection', y='reshaped_labels', data=projections_df,
            hue='reshaped_labels', palette=dict(zip(projections_df['reshaped_labels'], projections_df['color'])),
            legend=False)

        plt.title('← {} {} {} →'.format("she",
                                        ' ' * 20,
                                        "he"))

        plt.xlabel('Gender Direction Projection')
        plt.ylabel('Occupation Pair')
        # Create a "figures" directory if it doesn't exist
        if not os.path.exists('figures'):
            os.makedirs('figures')

        # Save the plot to the "figures" directory
        plot_file_path = os.path.join('figures', f'{title}.png')
        plt.savefig(plot_file_path,dpi=300)

        if not os.path.exists('results'):
            os.makedirs('results')
        # Save projections_df to the "results" directory
        projections_df.to_csv(os.path.join('results', f'{title}.csv'), index=False, encoding="utf-8")


        # Show the plot
        plt.show()
        return average_projection

    def visualize_single_bias(self,occ,sp):
        differences = self.calculate_differences()

        df = pd.DataFrame()
        df['occupation'] = occ
        df['reshaped_labels'] = df['occupation'].apply(lambda x: get_display(reshape(x)))
        df['bias'] = [sp[i][0] for i in range(len(occ))]
        df.sort_values(by='occupation', ascending=True, inplace=True)

        cmap = plt.get_cmap('RdBu')
        df['color'] = ((df['bias'] + 0.5)
                       .apply(cmap))

        sns.barplot(x='bias', y='reshaped_labels', data=df,
                    palette=df['color'])

        plt.title('← {} {} {} →'.format("she",
                                        ' ' * 20,
                                        "he"))

        plt.xlabel('Gender Direction Projection')
        plt.ylabel('Occupation')
