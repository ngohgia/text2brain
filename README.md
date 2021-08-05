# text2brain
Generating brain activation maps from free-form text query

- Create conda environment from env.yml
- Download checkpoints from [Google Drive](https://drive.google.com/file/d/13Gc0M4i4zj16aVtZzGUQs4oPcoyRTGWw/view?usp=sharing) and do `tar -xzvf checkpoints.tar.gz`
- Download uncased SciBert pretrained model from [ ] and unzip into `scibert_scivocab_uncased` directory
- Run `python predict_cpu_only.py <input_query> <output_file>`, e.g `python predict_cpu_only.py "self-generated thought" prediction.nii.gz`
