# visualizing_image_embeddings

Python implementation for visualizing lower-dimensional representations/embeddings of images that are created by  first feature extraction using pretrained (ImageNet) ResNet-18 (default) and application t-distributed stochastic neighbor embedding (t-SNE) method on those ResNet-18 features

## Installation

```bash
git clone [the url of visualizing_image_embeddings]
cd visualizing_image_embeddings
pip install -r requirements.txt
```

## Usage

Tested on Python 3.8.10 

```bash
python main.py -p [path of the input spreadsheet]
```

The data set, collection of textures in histological images of human colorectal cancer, by Kather et al. 2016 (https://zenodo.org/record/53169#.Y8U85nbMJNY) is used as sample input in this repository. Construction of the sample input spreadsheet is shown in "make_sample_spreadsheet.py" for the required format.

## Sample output plots 

![Thumbnail-plot](https://github.com/KatherLab/visualizing_image_embeddings/tree/main/out/plots/plot_scatter_thumbnails.png)

![Dot-plot](https://github.com/KatherLab/visualizing_image_embeddings/tree/main/out/plots/plot_scatter_dots.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)
