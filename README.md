# IsoPlume-Net

IsoPlume-Net is a multi-modal U-Net for automated, pixel-level delineation of individual fire smoke plumes. It uses Sentinel-5P TROPOMI total column carbon monoxide (TCCO) as the primary input. VIIRS active fire detections, true-colour RGB and false-colour RGB imagery are used as auxiliary inputs.

The model was developed to support the Fire Radiative Energy Method (FREM). In FREM, plume-integrated excess CO mass is regressed against fire radiative energy (FRE) to derive CO emission coefficients. IsoPlume-Net replaces the manual plume digitisation step in this workflow.

This repository contains the code for the model, data preparation, training and evaluation described in the paper below.

## Paper

Li, Z. and Wooster, M. J. (2026). An automatic 'IsoPlume-Net' EO-based atmospheric plume delineation approach to support improved fire emission estimation. *Science of Remote Sensing*. 


## Repository structure

```
IsoPlume-Net/
├── model.py      # IsoPlume-Net architecture and the simple concatenation baseline
├── data.py       # Data loading, augmentation, and train/validation/test splitting
├── train.py      # Training with the CO-aware weighted BCE loss
└── evaluate.py   # Pixel, plume-level and separation metrics on the test set
```

## Requirements

The code is written in Python and uses PyTorch. Install the dependencies with:

```bash
pip install torch numpy opencv-python pillow scipy scikit-image pandas
```

A CUDA-capable GPU is recommended for training. The experiments in the paper were run on NVIDIA A100 GPU.

## Data format

Each sample consists of five aligned PNG images of size 256 × 256 pixels. Place them in five folders with these names:

```
data_dir/
├── co/          # S5P TCCO, single channel, min-max scaled to [0, 1]
├── fire/        # VIIRS active fire mask, binary, single channel
├── rgb1/        # VIIRS true-colour RGB (M5, M4, M3)
├── false_rgb/   # VIIRS false-colour RGB (M7, M2, M1)
└── mask/        # Manually delineated plume mask, binary
```

Each image covers a 4° × 4° window centred on the plume location. The 256 × 256 grid gives a consistent input size for the network only. It is not a physical resampling of the TROPOMI data, so excess CO mass should be computed from the original Level 2 observations within the predicted plume.

## Data and model availability

The trained model weights and the annotated training data are available from the corresponding author on request, for research purposes and collaboration.

The input data are publicly available:

- Sentinel-5P TROPOMI CO: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu)
- VIIRS active fires (VNP14): [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov)
- VIIRS RGB imagery: [NASA Worldview](https://worldview.earthdata.nasa.gov)
- Meteosat SEVIRI FRP-PIXEL (for FRE): [EUMETSAT LSA SAF](https://lsa-saf.eumetsat.int)

## Citation

If you use this code, please cite:

```bibtex
@article{li2026isoplume,
  author  = {Li, Zitong and Wooster, Martin J.},
  title   = {An automatic `IsoPlume-Net' EO-based atmospheric plume delineation approach to support improved fire emission estimation},
  journal = {Science of Remote Sensing},
  year    = {2026},
}
```

## Acknowledgements

This work was funded by NERC National Capability funding to the National Centre for Earth Observation (NCEO; NE/R016518/1) and by EUMETSAT's Satellite Applications Facility Programme through the LSA SAF (CDOP-4).

## Contact

Zitong Li, Department of Geography, King's College London: zitong.li@kcl.ac.uk

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
