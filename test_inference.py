import open3d.visualization
from src.pcd_de_noising import WeatherNet, PCDDataset
from pathlib import Path
from typing import Tuple
import pytorch_lightning as pl
import torch
import os
import glob
import h5py
import matplotlib.pyplot as plt
import open3d
from mpl_toolkits.mplot3d import Axes3D
import streamlit
import time

DATA_KEYS = ["distance_m_1", "intensity_1"]
LABEL_KEY = "labels_1"
XYZ_KEYS = ["sensorX_1", "sensorY_1", "sensorZ_1"]

class TestDataLoader:
    def __init__(self, input_sq_dir: str) -> None:
        input_files = sorted(glob.glob(os.path.join(input_sq_dir, "*.hdf5")))
        self.input_sequence = []
        for file in input_files:
            with h5py.File(file, "r") as h5_file:
                data = [h5_file[key][()] for key in DATA_KEYS]
                xyz_data = [h5_file[key][()] for key in XYZ_KEYS]
            data = tuple(torch.from_numpy(data) for data in data)
            xyz_data = tuple(torch.from_numpy(d) for d in xyz_data)
            data = torch.stack(data)
            xyz_data = torch.stack(xyz_data)
            distance = data[0:1, :, :]  # 1 x 32 x 400
            reflectivity = data[1:, :, :]  # 1 x 32 x 400
            self.input_sequence.append([distance, reflectivity, xyz_data])
    
    def init_seq(self) -> None:
        self.seq_idx = 0
    
    def sequence(self) -> Tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.seq_idx >= len(self.input_sequence):
            return (False, torch.Tensor(), torch.Tensor(), torch.Tensor())
        else:
            distance = self.input_sequence[self.seq_idx][0]
            reflectivity = self.input_sequence[self.seq_idx][1]
            xyz_data = self.input_sequence[self.seq_idx][2]
            self.seq_idx += 1
            return (True, distance, reflectivity, xyz_data)

class TestWeatherNetInference:
    def __init__(self, ckpt_file: str) -> None:
        self.model = WeatherNet.load_from_checkpoint(ckpt_file, num_classes=4)
    
    def set_data_loader(self, data_loader: TestDataLoader) -> None:
        self.data_loader = data_loader
    
    def inference(self, distance: torch.Tensor, reflectivity: torch.Tensor) -> torch.Tensor:
        pred = self.model.predict(distance=distance.unsqueeze(1).to("cuda"), reflectivity=reflectivity.unsqueeze(1).to("cuda"))
        return pred

    def play_sequence(self):
        self.data_loader.init_seq()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)
        graph = ax.scatter([], [], [], s = 1, color='orange')
        placeholder = streamlit.empty()
        while True:
            res, distance, reflectivity, xyz_data = self.data_loader.sequence()
            if not res:
                break
            pred = self.inference(distance, reflectivity)
            mask = (pred == 2) | (pred == 3)
            filtered_tensor = xyz_data[:, ~mask.squeeze(0).to("cpu")]

            vis_x = filtered_tensor[0, :]
            vis_y = filtered_tensor[1, :]
            vis_z = filtered_tensor[2, :]
            graph._offsets3d = (vis_x, vis_y, vis_z)
            # # placeholder.pyplot(fig)
            # time.sleep(0.1)
            plt.draw()
            plt.pause(0.1)
        # plt.show()

            

def main()->None:

    data_lodaer = TestDataLoader("./data/test/2018-11-29_111451_Static2-Rain15")
    inferenece_module = TestWeatherNetInference("./tb_logs/WeatherNet/version_11/checkpoints/epoch=49-step=100.ckpt")
    inferenece_module.set_data_loader(data_lodaer)
    inferenece_module.play_sequence()
    # model = WeatherNet.load_from_checkpoint("./tb_logs/WeatherNet/version_11/checkpoints/epoch=49-step=100.ckpt", num_classes=4)
    # # PCDDataset("./data/test", recursive=True)
    # input_files = sorted(Path("./data/test").glob("**/*.hdf5"))
    # with h5py.File(input_files[0], "r") as h5_file:
    #     data = [h5_file[key][()] for key in DATA_KEYS]
    #     xyz_data = [h5_file[key][()] for key in XYZ_KEYS]
    # data = tuple(torch.from_numpy(data) for data in data)
    # xyz_data = tuple(torch.from_numpy(d) for d in xyz_data)
    # data = torch.stack(data)
    # xyz_data = torch.stack(xyz_data)
    # distance = data[0:1, :, :]  # 1 x 32 x 400
    # reflectivity = data[1:, :, :]  # 1 x 32 x 400
    
    # print(xyz_data.shape)
    # # print(distance.shape)
    # # print(reflectivity.shape)
    # # model(distance, reflectivity)
    # # return

    # pred = model.predict(distance=distance.unsqueeze(1).to("cuda"), reflectivity=reflectivity.unsqueeze(1).to("cuda"))
    # mask = (pred == 2) | (pred == 3)
    # filtered_tensor = xyz_data[:, ~mask.squeeze(0).to("cpu")]
    # print(filtered_tensor.shape) 

if __name__=="__main__":
    main()