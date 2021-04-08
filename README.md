logo
===================================================
LOGO: Learning observation models with graph optimization

<br />

Minimal nav2d example
------

Create a virtual python environment. In the python/ dir execute:
```
pip install -e .
```

Download nav2d examples
```
./download_dataset.sh
```

Alternately, create your own using
```
python src/logopy/dataio/generate_nav2dsim_dataset.py
```

Run
```
python examples/logo_nav2d_gtsam.py
```
