# @title Open the file explorer on the left of your screen and drag-and-drop an audio file anywhere. Then run the below cell.
# @markdown If you get an error relating to numpy please restart the runtime. (Runtime > Restart runtime)
import os
import glob
import json
import copy
import logging
import io
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import torch
from inference import infer_tool
from inference import slicer
from inference.infer_tool import Svc
import soundfile
import numpy as np

MODELS_DIR = "./models"


def get_speakers():
    speakers = []
    for _, dirs, _ in os.walk(MODELS_DIR):
        for folder in dirs:
            print("== folder", folder)
            cur_speaker = {}
            # Look for G_****.pth
            g = glob.glob(os.path.join(MODELS_DIR, folder, 'G_*.pth'))
            if not len(g):
                print("Skipping "+folder+", no G_*.pth")
                continue
            cur_speaker["model_path"] = g[0]
            cur_speaker["model_folder"] = folder

            # Look for *.pt (clustering model)
            clst = glob.glob(os.path.join(MODELS_DIR, folder, '*.pt'))
            if not len(clst):
                print("Note: No clustering model found for "+folder)
                cur_speaker["cluster_path"] = ""
            else:
                cur_speaker["cluster_path"] = clst[0]

            # Look for config.json
            cfg = glob.glob(os.path.join(MODELS_DIR, folder, '*.json'))
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
                    cur_speaker["name"] = f'{name}/{folder}'
                    cur_speaker["id"] = i
                    if not name.startswith('.'):
                        speakers.append(copy.copy(cur_speaker))
        print(f"== {len(speakers)} SPEAKERS:", speakers)
        return sorted(speakers, key=lambda x: x["name"].lower())


logging.getLogger('numba').setLevel(logging.WARNING)
chunks_dict = infer_tool.read_temp("inference/chunks_temp.json")
existing_files = []
slice_db = -40
wav_format = 'wav'


class InferenceGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inference GUI")
        self.geometry("600x300")

        self.speakers = get_speakers()
        self.speaker_list = [x["name"] for x in self.speakers]
        self.create_widgets()
        self.update_file_list()

    def create_widgets(self):
        ttk.Label(self, text="Speaker:").grid(column=0, row=0)
        self.speaker_var = tk.StringVar()
        self.speaker_box = ttk.Combobox(
            self, textvariable=self.speaker_var, values=self.speaker_list)
        self.speaker_box.grid(column=1, row=0)
        if self.speaker_list:
          self.speaker_box.insert(0, self.speaker_list[0])

        ttk.Label(self, text="Transpose (int):").grid(column=0, row=1)
        self.trans_tx = ttk.Entry(self)
        self.trans_tx.grid(column=1, row=1)
        self.trans_tx.insert(0, '0')

        ttk.Label(self, text="Clustering Ratio (float):").grid(column=0, row=2)
        self.cluster_ratio_tx = ttk.Entry(self)
        self.cluster_ratio_tx.grid(column=1, row=2)
        self.cluster_ratio_tx.insert(0, '0')

        ttk.Label(self, text="Noise Scale (float):").grid(column=0, row=3)
        self.noise_scale_tx = ttk.Entry(self)
        self.noise_scale_tx.grid(column=1, row=3)
        self.noise_scale_tx.insert(0, '0.4')
        

        self.auto_pitch_var = tk.BooleanVar()
        self.auto_pitch_ck = ttk.Checkbutton(
            self, text="Auto pitch f0 (do not use for singing)", variable=self.auto_pitch_var)
        self.auto_pitch_ck.grid(column=0, row=4, columnspan=2)

        self.convert_btn = ttk.Button(
            self, text="Convert", command=self.convert)
        self.convert_btn.grid(column=0, row=5)

        self.clean_btn = ttk.Button(
            self, text="Delete all audio files", command=self.clean)
        self.clean_btn.grid(column=1, row=5)

        ttk.Label(self, text="Input Files:").grid(column=0, row=6, sticky="W")
        self.file_listbox = tk.Listbox(self, width=50, height=10)
        self.file_listbox.grid(column=0, row=7, columnspan=2)

    def _get_input_filepaths(self):
        return [f for f in glob.glob('./_svc_in/**/*.*', recursive=True)
                if f not in existing_files and any(f.endswith(ex) for ex in ['.wav', '.flac', '.mp3', '.ogg', '.opus'])]

    def update_file_list(self):
        self.file_listbox.delete(0, tk.END)
        input_files = self._get_input_filepaths()
        for filepath in input_files:
            filename = os.path.basename(filepath)
            self.file_listbox.insert(tk.END, filename)

    def convert(self):
        trans = int(self.trans_tx.get() or '0')
        print("CURRENT SPEAKER:", self.speaker_box.get())
        speaker = next(x for x in self.speakers if x["name"] == self.speaker_box.get())
        spkpth2 = os.path.join(os.getcwd(), speaker["model_path"])
        print(spkpth2)
        print(os.path.exists(spkpth2))

        svc_model = Svc(speaker["model_path"], speaker["cfg_path"],
                        cluster_model_path=speaker["cluster_path"])

        input_filepaths = self._get_input_filepaths()
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
                        _cluster_ratio = float(self.cluster_ratio_tx.get() or '0')
                    out_audio, out_sr = svc_model.infer(
                        speaker["name"].split('/')[0], trans, raw_path,
                        cluster_infer_ratio=_cluster_ratio,
                        auto_predict_f0=bool(self.auto_pitch_var.get()),
                        noice_scale=float(self.noise_scale_tx.get() or '0.4'))
                    _audio = out_audio.cpu().numpy()
                    pad_len = int(svc_model.target_sample * 0.5)
                    _audio = _audio[pad_len:-pad_len]
                audio.extend(list(infer_tool.pad_array(_audio, length)))
            model_output_name = speaker["name"].split('/')[-1]
            res_path = os.path.join('./_svc_out/',
                                    f'{wav_name}_{trans}_key_'
                                    f'{model_output_name}.{wav_format}')
            soundfile.write(res_path, audio, svc_model.target_sample,
                            format=wav_format)

    def clean(self):
        input_filepaths = [f for f in glob.glob('./_svc_out/**/*.*', recursive=True)
                           if f not in existing_files and
                           any(f.endswith(ex) for ex in ['.wav', '.flac', '.mp3', '.ogg', '.opus'])]
        for f in input_filepaths:
            os.remove(f)


if __name__ == "__main__":
    app = InferenceGui()
    app.mainloop()
