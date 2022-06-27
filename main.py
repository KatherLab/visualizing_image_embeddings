#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Didem Cifci'
__copyright__ = '2022 Kather Lab at RWTH Aachen'
__license__ = 'MIT'
__version__ = '0.1.0'

import argparse
from cluster_and_plot import DataClustering
from extract_feature_vectors import FeatureExtraction

parser = argparse.ArgumentParser(
    description="Visualize the t-SNE representation of images")
parser.add_argument('-p', '--path_to_spreadsheet', type=str, metavar='', required=True,
                    default='images_path_label_Kather_2016.csv',
                    help='Path to the input spreadsheet file')
parser.add_argument('-o', '--output_directory', type=str, metavar='', required=True,
                    default='/zpool/projects/Visualizing_image_embeddings',
                    help='Path to folder where the results will be saved')
parser.add_argument('-s', '--self_supervised', type=bool, metavar='', required=False,
                    help='Whether to extract features with a self-supervised ResNet18 model from Ciga et al. 2021 trained with histopathology images')

args = parser.parse_args()

if __name__ == "__main__":
    
    fc = FeatureExtraction(args.path_to_spreadsheet, args.output_directory) # args.self_supervised: optional parameter,
                                                                            # set to False by default
    fc.extract_and_save_feature_vectors()

    dc = DataClustering(args.output_directory)

    dc.run_tSNE()
    dc.plot_scatter()
    dc.plot_imgs()
