# Malafide
Implementation of the attack from *Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems* [[1]](#mipa23).

A technical note: the paper reports results for the following countermeasures (CMs):
- RawNet, described in [[2]](#rawnet);
- AASIST, described in [[3]](#aasist);
- A solution based on a fine-tuned wav2vec 2.0 with a linear classification head, described in [[4]](#tak22).

In this repository, the attack is performed on an even stronger CM which consists of the combination of wav2vec 2.0 with AASIST, again described in [[4]](#tak22). This is due to the fact that we have refactored the code for better usability as we are developing the next version of Malafide. If you want to reproduce the exact results of the paper with the original three CMs, you can do one of the following:
1. Change the CM yourself. This shouldn't be too hard, I suggest you place the model implementation in the `models` folder, then change the configuration file accordingly (see the `model_config` key in `configs/initial.conf`)
2. Contact me and ask me for the original code. My email is in [[1]](#mipa23).

## TL;DR
If you are just here to grab the Malafide implementation, you can find the PyTorch class in `malafide.py`. At its core, it's basically just:
```python
class Malafide(torch.nn.Module):
    def __init__(self, filter_size):
        super().__init__()
        self.naughty_filter = torch.nn.Conv1d(1, 1, filter_size, padding='same', bias=None)
        # dirac delta-like init
        self.naughty_filter.weight.data[0, 0, filter_size//2] = 1

    def forward(self, x):
        return self.naughty_filter(x)
```
Although the full class has some additional boilerplate. But that's the gist. Then you can use it as follows:
```python
malafilter = Malafide(1025)
audio = torch.randn(4,1,16000) # input waveforms of shape (B,1,L)
output_audio = malafilter(audio)

# then forward output_audio into your countermeasure model...
```
Of course, since Malafide is a torch Module, you can optimize it with whatever PyTorch optimizer.

## Installation
As usual, install the required Python packages with
```bash
pip install -r requirements.txt
```
This will install the dependencies to run everything in the codebase **except for the CM system.**

To run the CM used in this codebase, you need to install fairseq.
Specifically, you need the right fairseq version to run the SSL-AASIST model of [[4]](#tak22).  
You can find the instructions in the [original repository](https://github.com/TakHemlata/SSL_Anti-spoofing) but, for your convenience,
I'm reporting here the steps to do it.  According to the repo, you need to install in editable mode [this specific version of fairseq](https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1). You can do it as follows:
```bash
git clone https://github.com/facebookresearch/fairseq
cd fairseq
git reset --hard a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
```
My guess is that you don't *absolutely* need that exact version, it's probably just the version that happened to be out when [[4]](#tak22) was developed. Maybe a pip installation of fairseq will do just fine. You gotta take your chances in life sometimes.

However, you do need the pretrained checkpoints. First, you need to get the pretrained XLS-R 300M backbone from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec/xlsr). Then you have to download the fine-tuned checkpoint from the repository of [[4]](#tak22): the link (taken from the original repository) is [this one](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB), and you have to download the checkpoint named `LA_model.pth`.

Of course, the process will vary if you are using different models.

## Dataset
The dataset is ASVspoof2019, which can be found [here](https://datashare.ed.ac.uk/handle/10283/3336). You should place it into a folder of your liking, then change the configuration file accordingly. In the key `dataset_root`, assign as value the path of the folder containing the flac files.

## Run the Malafide training
To train a Malafide filter, run:
```bash
python Main_script_Malafide_training.py --fine_tuned --config ./configs/initial.conf --adv_filter_size=2049 --attack A10
```
Where you change the configuration file path, the filter size, and the attack type accordingly.  
The attack type can is a string of format `A<n>`, where `<n>` goes from `07` to `19`, which are the attacks contained in the evaluation partition of the ASVspoof2019 dataset. This option will select the appropriate training and validation protocol from the `protocols` folder (FYI, the protocols with prefix `_none` contain both training and validation files. You don't really need them to run the system).

The script performs validation every epoch and saves the checkpoints of the Malafide filters (which are just 1d vectors) and training scores (in tensorboard format) in a directory called `results`. You can change that path using the `--output_folder` script parameter. During validation, EER is computed on the set of Malafide'd spoofed samples (of only the selected attack) and all bonafide samples of ASVspoof2019.  
Malafide checkpoints are saved as torch state dictionaries, so you should be able to load them and use them as with any torch model:
```python
malafilter.load_state_dict('path/to/checkpoint.pth')
```

## Audio samples
We provide samples of audio clips convolved with the Malafide filters **trained to fool RawNet** (the one in [[2]](#rawnet)). They can be found in `rawnet_audio_samples.zip`. Files named `LA_E_<id>.flac` are the original spoofed audio. Files named `LA_E_<id>_conv<atk>_<flt_length>.flac` are the same waveform convolved with a Malafide filter of length `<flt_length>` trained on RawNet. `<atk>` matches the spoofing attack that was used to produce the initial waveform.

## References
<span id="mipa23">[1]</span> Panariello, M., Ge, W., Tak, H., Todisco, M., Evans, N. (2023) Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems. Proc. INTERSPEECH 2023, 2868-2872, doi: 10.21437/Interspeech.2023-703  
<span id="rawnet">[2]</span> H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans and A. Larcher, "End-to-End anti-spoofing with RawNet2," _ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, Toronto, ON, Canada, 2021, pp. 6369-6373, doi: 10.1109/ICASSP39728.2021.9414234  
<span id="aasist">[3]</span> J. -w. Jung _et al_., "AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks," _ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, Singapore, Singapore, 2022, pp. 6367-6371, doi: 10.1109/ICASSP43922.2022.9747766  
<span id="ssl">[4]</span> Tak, H., Todisco, M., Wang, X., Jung, J.-w., Yamagishi, J., Evans, N. (2022) Automatic Speaker Verification Spoofing and Deepfake Detection Using Wav2vec 2.0 and Data Augmentation. Proc. The Speaker and Language Recognition Workshop (Odyssey 2022), 112-119, doi: 10.21437/Odyssey.2022-16
