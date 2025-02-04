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
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import json



LibriSQA_path = "path/LibriSQA-PartI_LibriSQA-PartI-test.json"
IEMOCAP_path= "path/test.json"
data_root = "path/IEMOCAP/IEMOCAP_full_release/"


import h5py
import torch

import h5py
import torch
def move_to_cuda(input_dict):
    return {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in input_dict.items()}

def generate_outputs(UID, wav_path, transcript, prompt, h5file):
    """Generates model outputs and saves them efficiently in the open HDF5 file."""
    try:
        print("=====================================")
        print("Processing UID:", UID)

        # Ensure we don't overwrite existing data for the UID
        if UID in h5file:
            print(f"Skipping {UID}, already exists in the HDF5 file.")
            return  # Skip processing if already saved
        
        # Create a group for this UID
        group = h5file.create_group(UID)
        

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": wav_path},
                {"type": "text", "text":  prompt.strip()},
            ]},
        ]
        prompt_audio = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
       
        if wav_path is None:
            audios = [[]]
        else:
            audios = []
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    wav_path, 
                                    sr=processor.feature_extractor.sampling_rate)[0]
                    )

        inputs = processor(text=prompt_audio, audios=audios, return_tensors="pt", padding=True)
        inputs =move_to_cuda(inputs)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        generate_ids = outputs.sequences[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response_text_audio, hidden_states_audio = response, outputs.hidden_states
        audio_group = group.create_group("audio_model")
        audio_group.attrs["response_text"] = response_text_audio
        for idx, state in enumerate(hidden_states_audio):
            state_group = audio_group.create_group(f"tuple_{idx}")
            for t_idx, tensor in enumerate(state):
                state_group.create_dataset(
                    f"tensor_{t_idx}", data=tensor.detach().cpu().numpy(), compression="gzip"
                )


        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": None},
                {"type": "text", "text":  "Audio Transcription: {text}\n\n{question}".format(text=transcript, question=prompt.strip())},
            ]},
        ]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs = processor(text=prompt_text, audios= [[]], return_tensors="pt", padding=True)
        inputs =move_to_cuda(inputs)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        print(outputs.sequences.shape)
        print(len(outputs.hidden_states))
        generate_ids = outputs.sequences[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response_text_text, hidden_states_text = response, outputs.hidden_states
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
            for i in data:
                wav_path = "/home/mila/a/ali.parviz/pooneh_prj/" + i["speech_path"].replace(".wav", ".flac")
                transcript = i["text"]
                prompt = i["question"]+". Answer the  question as short as possible (in 10-15 words)"
                uid= i["speech_path"].split('/')[-1].split('.')[0]
                generate_outputs(uid,wav_path,transcript,prompt,h5file)

def extract_IEMOCAP(data_path,h5_filename):
    prompt = "classify the emotion of the speaker in only one word including anger, happiness, sadness, neutrality."
    # Step 1: Load JSON data from the file
    with h5py.File(h5_filename, "a") as h5file:  # Open file in append mode
        with open(data_path, "r") as file:
            json_data = json.load(file)

        # Step 2: Update the `wav` paths
        for key, value in json_data.items():
            wav_path = value["wav"].replace("{data_root}", data_root)
            transcript =value["text"]
            generate_outputs(key,wav_path,transcript,prompt,h5file)


import argparse

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run tasks based on the given input.")
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


    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")


    # Run task based on the input
    if args.task == "SQA":
        extract_LibriSQA(args.path,"path/QWEN-LibriSQA-hidden.h5")
    elif args.task == "ER":
        extract_IEMOCAP(args.path,"path/QWEN-IEMOCAP-hidden2.h5")