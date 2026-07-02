# HYPERSPECTRAL_FLASK

Quick guide to run and demo this project (final-year project): a Flask web app that performs inference
using a pretrained tflearn/TensorFlow model for hyperspectral image classification.
ef.
Contents
- `app.py` — Flask app (UI + inference flow)
- `training.py` — training script that builds the model and saves a TF1.x checkpoint
- `TRAIN/`, `TEST/` — folders for training/testing images (filenames encode labels: F/W/C/R/D)
- `hyperspectral-0.001-2conv-basic.model.*` — TF checkpoint files (.meta, .index, .data)
- `infer_from_checkpoint.py` — helper script to attempt loading a checkpoint via TF compat.v1

Recommended ways to run (pick one):

## Option A — Quick demo in Google Colab (recommended for presentation)
1. Open a new Colab notebook.
2. Run this cell to enable TF1 and install libs:

```python
%tensorflow_version 1.x
!pip install tflearn opencv-python tqdm
```

3. Upload the three checkpoint files and a test image (use Colab's file upload UI or `files.upload()`).
4. Run the inference cell (same network as `app.py`) to load the model and predict. This avoids local TF1 install issues.

I included a notebook `colab_infer.ipynb` in this repo you can open in Colab.

## Option B — Local (recommended if you can install Conda)
1. Install Miniconda or Anaconda.
2. Create a TF1 environment (most reliable on Windows):

```powershell
conda create -n hyperspec python=3.7 -y
conda activate hyperspec
conda install -c anaconda tensorflow==1.15 -y
pip install tflearn flask numpy opencv-python matplotlib tqdm
python app.py
```

If `tensorflow==1.15` is unavailable for your Python version, use Conda with Python 3.7 or use the Colab option.

## Option C — Docker (isolated)
```powershell
docker run -it --rm -p 5000:5000 -v "%cd%":/app -w /app tensorflow/tensorflow:1.15.5-py3 bash
# inside container:
pip install tflearn flask opencv-python tqdm
python app.py
```

Notes
- The web UI expects images to be present in `TRAIN` or `TEST` (the app will search for the filename there).
- Filenames should start with a letter encoding class: `F` (forest), `W` (water), `C` (crop), `R` (residential), `D` (desert).
- For a guaranteed, trouble-free demo, use the Colab notebook.

If you'd like, I can: add a `requirements.txt`, prepare a one-click Colab notebook link, or produce a small `README_demo.md` with slide notes.
