import os
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from fastai.vision.all import *
from PIL import Image
from pathlib import Path
import numpy
import pandas as pd
from dataclasses import dataclass
import h5py

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Function taken from https://github.com/ozanciga/self-supervised-histopathology
def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded...')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model


@dataclass
class FeatureExtraction:
    path_spreadsheet: str
    out_dir: str
    is_selfSupervised: bool = False

    if is_selfSupervised == False:
        model = models.resnet18(pretrained=True).to(device)
        layer = model._modules.get('avgpool')
        model.eval()
    else:
        # Code block taken from https://github.com/ozanciga/self-supervised-histopathology
        model_path = Path(os.getcwd()) / 'models' / 'tenpercent_resnet18.ckpt'
        model = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load(model_path, map_location='cuda:0')
        state_dict = state['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace(
                'resnet.', '')] = state_dict.pop(key)

        model = load_model_weights(model, state_dict)
        model.fc = torch.nn.Sequential()
        model = model.cuda()

    def transform_image(self, image_path: str) -> torch.Tensor:
        scaler = transforms.Scale((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        t_img = Variable(
            normalize(to_tensor(scaler(img)).to(device)).unsqueeze(0))
        return t_img

    def get_feature_vectors(self, trans_img: torch.Tensor) -> torch.Tensor:
        """
        Extract the feature vector of the given image from the avgpool layer of ResNet-18 and
        returns the feature vector as a tensor

        Parameters: 
        trans_img (torch.Tensor): The image transformed into torch tensor

        Returns: 
        embedding (torch.Tensor)
        """

        embedding = torch.zeros(512)

        def copy_data(m, i, o):
            embedding.copy_(o.data.reshape(o.data.size(1)))
        h = self.layer.register_forward_hook(copy_data)
        self.model(trans_img)
        h.remove()

        return embedding

    def get_selfsup_feature_vectors(self, trans_img: torch.Tensor) -> torch.Tensor:
        out = self.model(trans_img)
        embedding = torch.squeeze(out)
        return embedding

    def extract_and_save_feature_vectors(self):
        """
        Extract the feature vectors of all images in the given spreadsheet and
        saves the feature vector and the labels of images in a HDF5 file in the specified output folder
        """
        # if self.deep_med_model != None:
        #    self.model = load_learner(deep_med_model).model.to(device)

        def get_label(x):
            i_image = df[df["file_names"] == x].index.values[0]
            label = df["labels"][i_image]
            if label.isnumeric():
                return torch.tensor(float(df["labels"][i_image]))
            else:
                unique_labels = sorted(df["labels"].unique())
                label_dict = dict()
                for i in range(len(unique_labels)):
                    label_dict[unique_labels[i]] = i
                # TODO: log the label dictionary
                # print(label_dict)
                return torch.tensor(float(label_dict[label]))

        Path(self.out_dir).mkdir(exist_ok=True)

        df = pd.read_csv(self.path_spreadsheet, dtype=str)  # engine='openpyxl'
        path_HDF5 = Path(self.out_dir) / 'features_and_labels.hdf5'
        dt = h5py.special_dtype(vlen=str)

        f = h5py.File(path_HDF5, 'w')

        all_features = []
        labels = []
        image_names = []

        for im_name in df["file_names"]:
            image_path = im_name
            trans_im = self.transform_image(image_path)
            if self.is_selfSupervised == False:
                image_vector = self.get_feature_vectors(trans_im)
            else:
                image_vector = self.get_selfsup_feature_vectors(trans_im)
            label = get_label(im_name)
            all_features.append(image_vector)
            labels.append(label)
            image_names.append(image_path)

        all_features = torch.stack((all_features), dim=0)
        if self.is_selfSupervised == False:
            np_all_features = all_features.numpy()
        else:
            np_all_features = all_features.cpu().detach().numpy()

        labels = torch.stack((labels), dim=0)
        labels = labels.reshape(1, -1).t()
        np_labels = labels.numpy()
        np_image_names = np.array((image_names), dtype=dt)

        f.create_dataset('features', data=np_all_features)
        f.create_dataset('labels', data=np_labels)
        f.create_dataset('im_names', data=np_image_names)

        f.close()
