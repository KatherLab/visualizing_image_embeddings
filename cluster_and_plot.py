from genericpath import exists
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from dataclasses import dataclass
from pathlib import Path
import h5py


@dataclass
class DataClustering:
    """Data clustering class using t-SNE algorithm. 

    Provide plotting functions
    """
    out_dir: str
    is_selfSupervised: bool = False

    def load(self):
        self.df = pd.DataFrame()
        self.out_dir = Path(self.out_dir)
        self.plot_dir = self.out_dir / 'plots'

        path_HDF5 = self.out_dir / 'features_and_labels.hdf5'
        f = h5py.File(path_HDF5, 'r')
        ds_features = f.get('features')
        self.X = np.array(ds_features)
        ds_labels = f.get('labels')
        self.y = np.array(ds_labels)
        ds_im_names = f.get('im_names')
        im_names = np.array(ds_im_names)
        self.df['im_names'] = im_names
        self.df['im_names'] = self.df['im_names'].map(
            lambda x: x.decode('utf-8'))
        f.close()

    def run_tSNE(self):
        self.load()
        print('Clustering with t-SNE is running...')
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(self.X)
        self.df['labels'] = self.y.T.reshape(-1)
        self.df['comp-1'] = X_2d[:, 0]
        self.df['comp-2'] = X_2d[:, 1]
        print('Done.')

    def plot_scatter(self):
        plt.figure(figsize=(16, 10))
        n_classes = len(self.df['labels'].unique())
        sns.scatterplot(x='comp-1', y='comp-2', hue=self.df.labels.tolist(),
                        palette=sns.color_palette('hls', n_classes),
                        data=self.df).set(title='T-SNE projection')
        # plt.show()
        self.plot_dir.mkdir(exist_ok=True)
        path_scatter_dots = self.plot_dir / 'plot_scatter_dots.png'
        plt.savefig(path_scatter_dots)

    def plot_imgs(self):
        fig, ax = plt.subplots()
        plt.figure(figsize=(16, 10))
        x_min = int(self.df['comp-1'].min())
        x_max = int(self.df['comp-1'].max())
        y_min = int(self.df['comp-2'].min())
        y_max = int(self.df['comp-2'].max())

        for im in self.df['im_names']:
            with mpl.cbook.get_sample_data(im) as file:
                arr_image = plt.imread(file)

            ax.set_xlim([x_min-30, x_max+30])
            ax.set_ylim([y_min-30, y_max+30])
            ax.set_title('t-SNE projection with images')
            ax.set_xlabel('comp-1')
            ax.set_ylabel('comp-2')
            #ax.lines = []
            # ax.axis("off")
            # ax.set_visible(False)
            idx = self.df[self.df['im_names'] == im].index.values[0]
            x = self.df['comp-1'][idx]
            y = self.df['comp-2'][idx]
            axin = ax.inset_axes([x, y, 10, 10], transform=ax.transData)
            axin.imshow(arr_image, cmap='gray')
            axin.axis('off')
        # plt.show()

        # fig.set_size_inches(20, 20)
        self.plot_dir.mkdir(exist_ok=True)
        path_out_imgs = self.plot_dir / 'plot_scatter_thumbnails.png'
        fig.savefig(path_out_imgs, dpi=100) # format='tiff'
