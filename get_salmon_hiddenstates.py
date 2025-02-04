# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample
import json
from tqdm import tqdm

LibriSQA_path = "/home/mila/a/ali.parviz/pooneh_prj/LibriSQA-PartI_LibriSQA-PartI-test.json"
IEMOCAP_path= "/home/mila/a/ali.parviz/pooneh_prj/test.json"
data_root = "/network/scratch/a/ali.parviz/salmon_ckpt/users/rwhetten/IEMOCAP/IEMOCAP_full_release/"


import h5py
import torch

import h5py
import torch

def generate_outputs(UID, wav_path, transcript, prompt, h5file):
    """Generates model outputs and saves them efficiently in the open HDF5 file."""
    try:
        print("=====================================")

        # Prepare the input sample
        samples = prepare_one_sample(wav_path, wav_processor)

        # Generate prompts for both models
        prompt_audio = [
            cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
        ]
        prompt_text = [
            cfg.config.model.prompt_template.format("<Text>" + transcript + "</Text>" + prompt.strip())
        ]

        print("Processing UID:", UID)

        # Ensure we don't overwrite existing data for the UID
        if UID in h5file:
            print(f"Skipping {UID}, already exists in the HDF5 file.")
            return  # Skip processing if already saved

        # Create a group for this UID
        group = h5file.create_group(UID)

        # Generate outputs for the first model (audio-based prompt)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            response_text_audio, hidden_states_audio = model.generate(
                samples, cfg.config.generate, prompts=prompt_audio
            )
            # print(response_text_audio)
            # print(len(hidden_states_audio))

            # Save audio model's hidden states and response text
            audio_group = group.create_group("audio_model")
            audio_group.attrs["response_text"] = response_text_audio
            for idx, state in enumerate(hidden_states_audio):
                state_group = audio_group.create_group(f"tuple_{idx}")
                for t_idx, tensor in enumerate(state):
                    state_group.create_dataset(
                        f"tensor_{t_idx}", data=tensor.detach().cpu().numpy(), compression="gzip"
                    )

        # Generate outputs for the second model (text-based prompt)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            response_text_text, hidden_states_text = model.generate(
                None, cfg.config.generate, prompts=prompt_text
            )
            # print(response_text_text)
            # print(len(hidden_states_text))

            # Save text model's hidden states and response text
            text_group = group.create_group("text_model")
            text_group.attrs["response_text"] = response_text_text
            for idx, state in enumerate(hidden_states_text):
                state_group = text_group.create_group(f"tuple_{idx}")
                for t_idx, tensor in enumerate(state):
                    state_group.create_dataset(
                        f"tensor_{t_idx}", data=tensor.detach().cpu().numpy(), compression="gzip"
                    )

    except Exception as e:
        print(f"Error processing UID {UID}: {e}")
        import pdb
        pdb.set_trace()


def extract_LibriSQA(data_path,h5_filename):

    with h5py.File(h5_filename, "a") as h5file:  # Open file in append mode
        with open(data_path, "r") as file:
            data = json.load(file)
            for i in tqdm(data):
                wav_path = "/home/mila/a/ali.parviz/pooneh_prj/" + i["speech_path"].replace(".wav", ".flac")
                transcript = i["text"]
                prompt = i["question"]+". Answer the  question as short as possible (in 10-15 words)"
                uid= i["speech_path"].split('/')[-1].split('.')[0]
                generate_outputs(uid,wav_path,transcript,prompt,h5file)

def extract_IEMOCAP(data_path,h5_filename):
    prompt = "Describe the emotion of the speaker in one word. The meotion could be only from these four category: anger, happiness, sadness, neutrality."
    # Step 1: Load JSON data from the file
    with h5py.File(h5_filename, "a") as h5file:  # Open file in append mode
        with open(data_path, "r") as file:
            json_data = json.load(file)

        # Step 2: Update the `wav` paths
        for key, value in tqdm(json_data.items()):
            wav_path = value["wav"].replace("{data_root}", data_root)
            transcript =value["text"]
            generate_outputs(key,wav_path,transcript,prompt,h5file)


import argparse

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run tasks based on the given input.")
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )

    # Define arguments
    parser.add_argument(
        "--task",
        type=str,
        default="SQA",
        # required=True,
        choices=["SQA", "ER"],
        help="Task to perform. Options: 'SQA' for LibriSQA extraction, 'EER' for IEMOCAP extraction."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=LibriSQA_path,
        # required=True,
        help="Path to the dataset for the selected task."
    )
    # Parse arguments
    args = parser.parse_args()

    cfg = Config(args)

    model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()

    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)



    # Run task based on the input
    if args.task == "SQA":
        extract_LibriSQA(args.path,"/network/scratch/a/ali.parviz/salmon_ckpt/LibriSQA-hidden.h5")
    elif args.task == "ER":
        extract_IEMOCAP(args.path,"/network/scratch/a/ali.parviz/salmon_ckpt/IEMOCAP-hidden.h5")