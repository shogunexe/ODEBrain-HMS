# Third-party sources and licenses

| Material | Source and attribution | Status in this repository |
|---|---|---|
| HMS Harmful Brain Activity Classification data | [Kaggle competition](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data?select=train_eegs), Jin Jing, Zhen Lin, Chaoqi Yang, Ashley Chow, Sohier Dane, Jimeng Sun, and M. Brandon Westover (2024). Kaggle lists the dataset as Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). | Raw EEG, spectrograms and metadata CSV are excluded from Git and the public resource archive. Six derived case figures in the resource archive must retain this attribution and the dataset terms. |
| ODEBrain reference implementation | [HHJIAnmo/ODEBRAIN](https://github.com/HHJIAnmo/ODEBRAIN), audited at commit `31e251ede69ed82fb47c627d2590286b6376c50a`; MIT, Copyright (c) 2025 Haohui JIA. | Referenced as a method. No official source files are bundled or presented as this project's implementation. See [adaptation audit](docs/odebrain_adaptation.md). |
| Python and system dependencies | See [requirements.txt](requirements.txt) and each dependency's upstream distribution. | Installed separately; their own license notices apply. No dependency source is vendored in this repository. |

The project's original code and weights have the separate notice in [LICENSE.md](LICENSE.md). The static case figures are derived from HMS data and retain its CC BY-NC 4.0 attribution; this file does not grant any rights beyond the upstream terms.
