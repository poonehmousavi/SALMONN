import h5py
import torch
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import argparse
import math
import matplotlib.pyplot as plt
import speechbrain
import speechbrain.lobes.models.FastSpeech2 as fs2
from dtw import *
import tqdm


def norm_and_convolve(seq1, seq2):
    seq1, seq2 = seq1.squeeze(), seq2.squeeze()

    len1 = seq1.shape[0]
    len2 = seq2.shape[0]
    
    if len2 > len1: 
        longerseq = seq2
        shorterseq = seq1
    else:
        longerseq = seq1
        shorterseq = seq2

    norm = (longerseq ** 2).sum(-1, keepdim=True).sqrt()
    longerseq = longerseq / norm
    
    norm = (shorterseq ** 2).sum(-1, keepdim=True).sqrt()
    shorterseq = shorterseq / norm

    innerprods = (shorterseq.unsqueeze(0) * longerseq.unsqueeze(1)).sum(-1).cpu()
    return innerprods

def find_path(mat):
    for t in range(mat.shape[1]):
        pass

def convert_whisper(whisper_data, simmat):

    whisper_mat = torch.zeros(simmat.shape)
    chunks = whisper_data['whisper_results']['chunks']

    eps = 1e-20
    max_time = chunks[-1]['timestamp'][0]
    ntime_steps = simmat.shape[0]
    for t, chunk in enumerate(chunks):
        begin = round((chunk['timestamp'][0] / (max_time + eps)) * ntime_steps)

        if chunk['timestamp'][1] is not None:
            end = round((chunk['timestamp'][1] / (max_time + eps)) * ntime_steps)
        else:
            end = simmat.shape[0]
        whisper_mat[begin:end, t] = 1
    return whisper_mat

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run tasks based on the given input.")
    parser.add_argument("--input_path", 
                        default="/path/QWEN-LibriSQA-hidden.h5",
                        type=str,  help='path to configuration file')

    parser.add_argument(
        "--output_path",
        type=str,
        default="/path/librisqa/qwen/librisqa_sumarized_data.pkl",
        # required=True,
        help="Path to the dataset for the selected task."
    )

    parser.add_argument(
        "--whisper_path",
        type=str,
        default=".",
        # required=True,
        help="Path to the dataset for the selected task."
    )

    parser.add_argument("--hidden_dim", default=4096, type=int, help='path to configuration file. 4096 for qwen and 5120 for salmon')
    # Parse arguments
    args = parser.parse_args()
# Load pre-trained sentence embedding model
    hidden_dim= args.hidden_dim
    text_embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Similarity Threshold (adjustable)
    SIMILARITY_THRESHOLD = 0.7  # Only keep pairs with >70% similarity

    # Load the HDF5 file
    h5_filename = args.input_path

    # Store filtered pairs
    filtered_data = {}
    skip_count = 0
    count = 0

    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    processor_qwen = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model_qwen = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
    
    whisper_data = pickle.load(open(args.whisper_path, 'rb'))
    #with h5py.File(h5_filename, "r") as h5file:
    h5file = h5py.File(h5_filename, "r")
    for UID in tqdm.tqdm(h5file.keys()):
        count +=1
        print(f"Processing UID: {UID}")

        # Load response texts
        #response_text_audio = h5file[UID]["audio_model"].attrs["response_text"][0]  #change to input_audio_model if you want to get input hiddensates and audio_model if you want to get output hiddenstates
        #response_text_text = h5file[UID]["text_model"].attrs["response_text"][0] #change to input_text_model if you want to get input hiddensates and text_model if you want to get output hiddenstate

        response_text_audio = h5file[UID]["audio_model"].attrs["response_text"]  #change to input_audio_model if you want to get input hiddensates and audio_model if you want to get output hiddenstates
        response_text_text = h5file[UID]["text_model"].attrs["response_text"] #change to input_text_model if you want to get input hiddensates and text_model if you want to get output hiddenstate


        # Convert response texts into sentence embeddings
        audio_text_embedding = text_embedder.encode(response_text_audio)
        text_text_embedding = text_embedder.encode(response_text_text)

        # Compute Cosine Similarity of response texts
        text_similarity = cosine_similarity([audio_text_embedding], [text_text_embedding])[0, 0]
        print(f"Text Similarity for {UID}: {text_similarity:.2f}")

        # **Filter by similarity threshold**
        if text_similarity >= SIMILARITY_THRESHOLD:
            print(f"Keeping {UID} (Similarity: {text_similarity:.2f})")

            # Load embeddings
            audio_group = h5file[UID]["audio_model"]
            text_group = h5file[UID]["text_model"]

            # **Extract Correct Number of Layers**
            num_layers_audio = len(audio_group["tuple_0"])  # Number of layers
            num_layers_text = len(text_group["tuple_0"])

            assert num_layers_audio == num_layers_text, "Mismatch in number of layers!"

            # **Extract Sequence Lengths**
            seq_len_audio = len(audio_group)  # Variable sequence length
            seq_len_text = len(text_group)

            # Initialize storage for layer-wise embeddings (Excluding layer 0)
            audio_embeddings = np.zeros((seq_len_audio, num_layers_audio, hidden_dim))
            text_embeddings = np.zeros((seq_len_text, num_layers_text, hidden_dim))

            # **Extract All Layers but **
            # for seq_idx in range(0,seq_len_audio):
            #     for layer_idx in range(0, num_layers_audio):

            #         layer_group = audio_group[f"tuple_{seq_idx}"]
            #         
            #         # Extract and fix sequence dimension
            #         tensor = torch.tensor(layer_group[f"tensor_{layer_idx}"][:])  # (1, 1, 5120) or (1, input_seq. 5120) if it is seq-0 since it contains input hidden states.

            #         audio_embeddings[seq_idx, layer_idx, :] = tensor.numpy()  
            # only taking the input 
            
            all_innerprods = []
            for nlayer in range(num_layers_audio):
                audio_embeddings = audio_group["tuple_0"][f'tensor_{nlayer}'][:]
                text_embeddings = text_group["tuple_0"][f'tensor_{nlayer}'][:]

                audio_embeddings = torch.from_numpy(audio_embeddings).cuda()
                text_embeddings = torch.from_numpy(text_embeddings).cuda()

                all_innerprods.append(norm_and_convolve(audio_embeddings, text_embeddings))

            # eliminate the prompt part
            # for qwen these are fixed
            promptpre_audio = 20
            promptpre_text = 27

            whisper_uid = whisper_data[UID]
            text_whisper = whisper_uid['transcript']
            qwen_prompt = whisper_uid['qwen_prompt']

            qwen_tokens = processor_qwen.tokenizer(qwen_prompt)['input_ids']
            decoded = processor_qwen.tokenizer.decode(qwen_tokens[promptpre_text:])

            for t in range(1, len(qwen_tokens[promptpre_text:])):
                candidate = processor_qwen.tokenizer.decode(qwen_tokens[promptpre_text:-t])
                if candidate.strip() == whisper_uid['transcript']:
                    break
            promptpost = -t

            # just doing this in case strings do not match due to special characters 
            if t > (len(qwen_tokens[promptpre_text:]) - 1):
                t = len(qwen_tokens[promptpre_text:]) // 2

            # qwen_tokens = processor_qwen.tokenizer(whisper_uid['qwen_prompt'])
            l2_errors = []
            l1_errors = []
            dtw_errors = []
            l1_errors_v2 = []
            l1_errors_v3 = []
            l1_path_errors = []

            for nlayer in range(num_layers_audio):
                layer = all_innerprods[nlayer][promptpre_audio:promptpost, promptpre_text:promptpost]
                whisper_mat = convert_whisper(whisper_uid, layer)

                # path_whisher = fs2.maximum_path_numpy(whisper_mat.unsqueeze(0), torch.ones(whisper_mat.unsqueeze(0).shape))

                # get the paths

                # if we use this, than the dtw seems to turn into a trivial distance. path_whisper = fs2.maximum_path_numpy(whisper_mat.transpose(1,0).unsqueeze(0), torch.ones(whisper_mat.transpose(1,0).unsqueeze(0).shape))
                path_layer = fs2.maximum_path_numpy(layer.transpose(1,0).unsqueeze(0), torch.ones(layer.transpose(1,0).unsqueeze(0).shape)).squeeze().transpose(1, 0)

                # get the errors
                l1_errors.append((whisper_mat - path_layer).abs().mean().item())
                l2_errors.append(((whisper_mat - path_layer)**2).sqrt().mean().item())
                l1_errors_v2.append((whisper_mat - layer).abs().mean().item())
                l1_errors_v3.append((path_layer - layer).abs().mean().item())

                path_whisper_ind = whisper_mat.argmax(1).cpu().numpy()
                path_layer_ind = path_layer.argmax(1).cpu().numpy()

                l1_path_errors.append(np.abs(path_whisper_ind - path_layer_ind).mean().item())

                dtw_alignment = dtw(path_whisper_ind, path_layer_ind)
                dtw_errors.append(dtw_alignment.distance.item())


            # if we want to see what is going on
            if 0:
                plt.imshow(whisper_mat.transpose(1, 0).cpu())
                plt.savefig('whisper_mat.png')

                plt.imshow(layer.transpose(1, 0).cpu())
                plt.savefig('last_layer.png')

                plt.imshow(path_layer.transpose(1, 0).cpu())
                plt.savefig('path_last_layer.png')

            filtered_data[UID] = {#'innerprods': all_innerprods,
                                  'l1_errors': l1_errors,
                                  'l1_errors_v2': l1_errors_v2,
                                  'l1_errors_v3': l1_errors_v3,
                                  'l1_path_errors': l1_path_errors,
                                  'l2_errors': l2_errors,
                                  'dtw_errors': dtw_errors}
            with open(args.output_path, "wb") as f:
                pickle.dump(filtered_data, f)

        else:
            print(f"Skipping {UID} (Similarity: {text_similarity:.2f})")
            skip_count +=1

    # Save Filtered Data
    print(f"total number of skipped is {skip_count} from total {count}")

    #with open(args.output_path, "wb") as f:
    #    pickle.dump(filtered_data, f)

    print("Filtering complete. Saved filtered dataset.")
