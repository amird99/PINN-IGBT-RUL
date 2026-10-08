# Data

The raw files are **not** stored in this repository.

**Source:** NASA Prognostics Center of Excellence, *Insulated-Gate Bipolar Transistor (IGBT) Accelerated Aging*
(thermal-overstress aging; devices 2-5 are the square-gate-signal devices used in the paper, "Data Set 8" of the repository).
Landing page: <https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging>
Repository index (as cited by the paper): <https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository>
**Citation:** J. Celaya, Phil Wysocki, and K. Goebel (2009) "IGBT Accelerated Aging Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

This repository does **not** redistribute the data and `data/raw/` is git-ignored: download the files from NASA yourself and cite the dataset
as above. Please read the dataset's terms of use before sharing any of its files.

Optional helper (downloads from NASA to your machine and extracts the four files; UNTESTED, may need adjusting if NASA changes the archive layout):

```bash
python scripts/download_data.py --out data/raw
```

Expected layout (note the two spaces in the file names):

```
data/raw/
├── Device2  1.mat
├── Device3  1.mat
├── Device4  1.mat
└── Device5  1.mat
```

If the download script fails (the archive layout may differ from what it expects), download the zip manually and copy
`.../Thermal Overstress Aging with Square Signal at gate and SMU data/Aging Data/Device N/DeviceN  1.mat` into `data/raw/`.
