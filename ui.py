#@title Open the file explorer on the left of your screen and drag-and-drop an audio file anywhere. Then run the below cell.
#@markdown If you get an error relating to numpy please restart the runtime. (Runtime > Restart runtime)
import os
import glob
import json
import copy
import logging
import io
from ipywidgets import widgets
from pathlib import Path
from IPython.display import Audio, display

import torch
from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc
import soundfile
import numpy as np

MODELS_DIR = "./models"

def get_speakers():
  speakers = []
  for _,dirs,_ in os.walk(MODELS_DIR):
    for folder in dirs:
      print("== folder", folder)
      cur_speaker = {}
      # Look for G_****.pth
      g = glob.glob(os.path.join(MODELS_DIR,folder,'G_*.pth'))
      if not len(g):
        print("Skipping "+folder+", no G_*.pth")
        continue
      cur_speaker["model_path"] = g[0]
      cur_speaker["model_folder"] = folder

      # Look for *.pt (clustering model)
      clst = glob.glob(os.path.join(MODELS_DIR,folder,'*.pt'))
      if not len(clst):
        print("Note: No clustering model found for "+folder)
        cur_speaker["cluster_path"] = ""
      else:
        cur_speaker["cluster_path"] = clst[0]

      # Look for config.json
      cfg = glob.glob(os.path.join(MODELS_DIR,folder,'*.json'))
      if not len(cfg):
        print("Skipping "+folder+", no config json")
        continue
      cur_speaker["cfg_path"] = cfg[0]
      with open(cur_speaker["cfg_path"]) as f:
        try:
          cfg_json = json.loads(f.read())
        except Exception as e:
          print("Malformed config json in "+folder)
        for name, i in cfg_json["spk"].items():
          cur_speaker["name"] = name
          cur_speaker["id"] = i
          if not name.startswith('.'):
            speakers.append(copy.copy(cur_speaker))

    return sorted(speakers, key=lambda x:x["name"].lower())

logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")
existing_files = []
slice_db = -40
wav_format = 'wav'

class InferenceGui():
  def __init__(self):
    self.speakers = get_speakers()
    self.speaker_list = [x["name"] for x in self.speakers]
    self.speaker_box = widgets.Dropdown(
        options = self.speaker_list
    )
    display(self.speaker_box)

    def convert_cb(btn):
      self.convert()
    def clean_cb(btn):
      self.clean()

    self.convert_btn = widgets.Button(description="Convert")
    self.convert_btn.on_click(convert_cb)
    self.clean_btn = widgets.Button(description="Delete all audio files")
    self.clean_btn.on_click(clean_cb)

    self.trans_tx = widgets.IntText(value=12, description='Transpose')
    self.cluster_ratio_tx = widgets.FloatText(value=0.0, 
      description='Clustering Ratio')
    self.noise_scale_tx = widgets.FloatText(value=0.4, 
      description='Noise Scale')
    self.auto_pitch_ck = widgets.Checkbox(value=False, description=
      'Auto pitch f0 (do not use for singing)')

    display(self.trans_tx)
    display(self.cluster_ratio_tx)
    display(self.noise_scale_tx)
    display(self.auto_pitch_ck)
    display(self.convert_btn)
    display(self.clean_btn)

  def convert(self):
    trans = int(self.trans_tx.value)
    speaker = next(x for x in self.speakers if x["name"] == 
          self.speaker_box.value)
    spkpth2 = os.path.join(os.getcwd(),speaker["model_path"])
    print(spkpth2)
    print(os.path.exists(spkpth2))

    svc_model = Svc(speaker["model_path"], speaker["cfg_path"], 
      cluster_model_path=speaker["cluster_path"])
    
    input_filepaths = [f for f in glob.glob('./data/**/*.*', recursive=True)
     if f not in existing_files and 
     any(f.endswith(ex) for ex in ['.wav','.flac','.mp3','.ogg','.opus'])]
    for name in input_filepaths:
      print("Converting "+os.path.split(name)[-1])
      infer_tool.format_wav(name)

      wav_path = str(Path(name).with_suffix('.wav'))
      wav_name = Path(name).stem
      chunks = slicer.cut(wav_path, db_thresh=slice_db)
      audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

      audio = []
      for (slice_tag, data) in audio_data:
          print(f'#=====segment start, '
              f'{round(len(data)/audio_sr, 3)}s======')
          
          length = int(np.ceil(len(data) / audio_sr *
              svc_model.target_sample))
          
          if slice_tag:
              print('jump empty segment')
              _audio = np.zeros(length)
          else:
              # Padding "fix" for noise
              pad_len = int(audio_sr * 0.5)
              data = np.concatenate([np.zeros([pad_len]),
                  data, np.zeros([pad_len])])
              raw_path = io.BytesIO()
              soundfile.write(raw_path, data, audio_sr, format="wav")
              raw_path.seek(0)
              _cluster_ratio = 0.0
              if speaker["cluster_path"] != "":
                _cluster_ratio = float(self.cluster_ratio_tx.value)
              out_audio, out_sr = svc_model.infer(
                  speaker["name"], trans, raw_path,
                  cluster_infer_ratio = _cluster_ratio,
                  auto_predict_f0 = bool(self.auto_pitch_ck.value),
                  noice_scale = float(self.noise_scale_tx.value))
              _audio = out_audio.cpu().numpy()
              pad_len = int(svc_model.target_sample * 0.5)
              _audio = _audio[pad_len:-pad_len]
          audio.extend(list(infer_tool.pad_array(_audio, length)))
          
      res_path = os.path.join('./output/',
          f'{wav_name}_{trans}_key_'
          f'{speaker["name"]}.{wav_format}')
      soundfile.write(res_path, audio, svc_model.target_sample,
          format=wav_format)
      display(Audio(res_path, autoplay=True)) # display audio file
    pass

  def clean(self):
     input_filepaths = [f for f in glob.glob('./output/**/*.*', recursive=True)
     if f not in existing_files and 
     any(f.endswith(ex) for ex in ['.wav','.flac','.mp3','.ogg','.opus'])]
     for f in input_filepaths:
       os.remove(f)

inference_gui = InferenceGui()