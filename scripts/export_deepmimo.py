from pathlib import Path
import json

import deepmimo as dm
from deepmimo.datasets.dataset import MacroDataset
import numpy as np
from scipy.io import savemat

scenario = "i2_28b"
dm.download(scenario)
dataset = dm.load(scenario)
if isinstance(dataset, MacroDataset):
    dataset = dataset.datasets[0]
print("Loaded", scenario, "keys", list(dataset.keys()))
frequency = float(dataset.rt_params.frequency)
assert abs(frequency-28e9)<1e6, frequency
positions = np.asarray(dataset.rx_pos)
power = np.asarray(dataset.power_linear)
active = np.flatnonzero(np.nansum(power, axis=1)>0)
# Spatially separated positions are reserved exclusively for held-out evaluation.
voxels = np.floor(positions[active]/2).astype(int)
_, first = np.unique(voxels, axis=0, return_index=True)
candidates = active[first]
rng = np.random.default_rng(5501)
selected = rng.permutation(candidates)[:96]
assert len(selected)>=32, "Insufficient spatially separated held-out positions"
dataset = dataset.trim(idxs=selected)
paths = min(10, int(dataset.max_paths))
delays = np.asarray(dataset.delay)[:, :paths].copy()
delays -= np.nanmin(delays, axis=1, keepdims=True)
assert np.nanmax(delays)<1e-3
theta = np.deg2rad(np.asarray(dataset.aoa_el)[:, :paths])
phi = np.deg2rad(np.asarray(dataset.aoa_az)[:, :paths])
doppler_factor = -np.sin(theta)*np.cos(phi)
result = dict(scenario=scenario, frequency=frequency, user_indices=selected+1,
              positions=np.asarray(dataset.rx_pos), delays=np.nan_to_num(delays),
              doppler_factor=np.nan_to_num(doppler_factor))
for size in (32, 64):
    params = dm.ChannelParameters(freq_domain=False, num_paths=paths,
                                  bs_antenna={"shape": [8, size//8]},
                                  ue_antenna={"shape": [8, size//8]})
    channel = np.nan_to_num(np.asarray(dataset.compute_channels(params)))
    assert channel.shape==(len(selected), size, size, paths), channel.shape
    # DeepMIMO stores horizontal elements first. MATLAB UPA stores vertical first.
    order = np.array([horizontal+8*vertical for horizontal in range(8)
                      for vertical in range(size//8)])
    channel = channel[:, order, :, :][:, :, order, :]
    matrices = channel.transpose(0, 3, 1, 2)
    u, singular, vh = np.linalg.svd(matrices, full_matrices=False)
    assert np.max(singular[..., 1]) < 1e-4*np.max(singular[..., 0])
    normalization = np.sqrt(np.sum(singular[..., 0]**2, axis=1)/size**2)
    receive = u[..., 0]*singular[..., 0, None]/normalization[:, None, None]
    transmit = np.conj(vh[..., 0, :])
    reconstructed = receive[..., :, None]*np.conj(transmit[..., None, :])
    reference = matrices/normalization[:, None, None, None]
    assert np.allclose(reconstructed, reference, rtol=1e-4, atol=1e-5)
    result[f"receive{size}"] = receive.transpose(0, 2, 1)
    result[f"transmit{size}"] = transmit.transpose(0, 2, 1)
    result[f"normalization{size}"] = normalization
output = Path("external")
output.mkdir(exist_ok=True)
savemat(output/"deepmimo_i2_28b.mat", result, do_compression=True)
metadata = dict(scenario=scenario, frequency=frequency, positions=len(selected), paths=paths,
                package_version="4.0.5", position_voxel_metres=2,
                channel_normalization="unit mean power per antenna pair",
                motion="locally frozen geometry with angle-based Doppler at test speed")
(output/"source.json").write_text(json.dumps(metadata, indent=2))
print(metadata)
