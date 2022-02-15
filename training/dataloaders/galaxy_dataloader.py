# --- Importations ---------------------------------------------------------
import os

# Python libraries
import h5py
import numpy as np

import numpy.random


# PyTorch libraries
import torch

from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, path, method="train", val_size=0.04, test_size=0.8):
        alpha = 0.55
        shape = "square"

        # Lens model parambounds
        SIE_pb = {"theta_E": (0.5, 2.5),
                  "e1": (-0.5, 0.5),
                  "e2": (-0.5, 0.5),
                  "center_x": (-0.1, 0.1),
                  "center_y": (-0.1, 0.1)}

        gamma_fac = 0.1
        SHEAR_pb = {"gamma1": (-gamma_fac, gamma_fac),
                    "gamma2": (-gamma_fac, gamma_fac),
                    "ra_0": (0, 0),
                    "dec_0": (0, 0)}

        # GalaxyLenser parameters
        GL_params = {"seed": None,
                     "shear": True,
                     "alpha": alpha,
                     "shape": "square",
                     "SIE_pb": SIE_pb,
                     "SHEAR_pb": SHEAR_pb}

        self.h5f = h5py.File(path)
        try:
            import json
            self.method = method
            with open("keys.json", 'r') as infile:
                self.keys = json.load(infile)

            with open("lenses_keys.json", 'r') as infile:
                self.lenses_keys = json.load(infile)
            np.random.shuffle(self.lenses_keys)  # in place operation
            nkeys=len(self.keys)
            arg1 = int((1 - val_size - test_size) * nkeys)
            arg2 = int((1 - test_size) * nkeys)
            self.ids = {

                "train": self.lenses_keys[0:arg1, :],
                "val": self.lenses_keys[arg1:arg2, :],
                "test": self.lenses_keys[arg2::, :]
            }
        except Exception as e:
            print(e)
            self.keys = self.h5f.keys()
            nkeys = len(list(self.keys))

            ids = np.arange(nkeys)
            # np.random.shuffle(ids)
            pair_keys = [f"pairs_s{i:05d}" for i in ids]  # list of all pair keys
            sources_keys = np.asarray(
                [([f"pairs_s{i:05d}"], [f"source_s{i:05d}"]) for i in ids])  # list of all sources keys as tuple
            lenses_keys = []
            for i in range(len(ids)):
                lense_key = []
                for key in self.h5f[pair_keys[i]]:

                    if key[:4] == "sour":
                        source_key = key
                    if key[:4] == "lens":
                        lense_key.append(key)

                for key in lense_key:
                    lenses_keys.append([pair_keys[i], source_key, key])

            self.lenses_keys = np.asarray(lenses_keys)

            np.random.shuffle(lenses_keys)  # in place operation
            arg1 = int((1 - val_size - test_size) * nkeys)
            arg2 = int((1 - test_size) * nkeys)
            self.index = 0
            self.ids = {

                "train": self.lenses_keys[0:arg1, :],
                "val": self.lenses_keys[arg1:arg2, :],
                "test": self.lenses_keys[arg2::, :]
            }

            self.method = method
            with open("keys.json", 'w') as outfile:
                data = json.dumps(list(self.keys))
                outfile.write(data)

            with open("lenses_keys.json", 'w') as outfile:
                data = json.dumps((self.lenses_keys.tolist()))
                outfile.write(data)

    def __len__(self):
        return len(self.ids[self.method])

    def __getitem__(self, idx):

        set = self.ids[self.method][idx]

        pair_keys = set[0]
        lens_keys = set[1]
        source_keys = set[2]

        lenses = torch.tensor(np.array(self.h5f[pair_keys][lens_keys]))

        sources = torch.tensor(np.array(self.h5f[pair_keys][source_keys]))
        # lenses, param = self.GL.forward(sources)
        param = 0  # not none
        npix = sources.shape[1]  # Number of pixels (side)
        nsamp = 1
        nchan = 1

        lenses = lenses.reshape(nsamp, nchan, npix, npix)
        sources = sources.reshape(nsamp, nchan, npix, npix)

        lenses = noise(lenses)

        def minmax_scaler(v):
            new_min, new_max = 0, 1  # TODO repare autoscaler with 0 255
            v_min, v_max = v.min(), v.max()
            v_p = (v - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
            return v_p

        lenses = minmax_scaler(lenses)
        sources = minmax_scaler(sources)
        return lenses.reshape((1,128,128)).float(),sources.reshape((1,128,128)).float()

def noise(x, expo_time=1000, sig_bg=.001):
	"""
	Inputs
		x : (torch tensor)[batch_size x nchan x npix x npix] image
		expo_time : (float) exposure time
		sig_bg : (float) standard deviation of background noise
	Outputs
		noisy_im : (torch tensor)[batch_size x nchan x npix x npix] noisy image
	"""
	poisson = sig_bg*torch.randn(x.size()) # poisson noise
	bckgrd = torch.sqrt(abs(x)/expo_time)*torch.randn(x.size()) # bakcground noise
	noisy_im = bckgrd+poisson+x
	return noisy_im
