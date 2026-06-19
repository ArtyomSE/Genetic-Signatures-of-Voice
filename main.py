import numpy as np
import pandas as pd
import torch
import torchaudio
import whisper

import re
import os
import sys
import string
from tqdm import tqdm

model = whisper.load_model("large").to('cuda')


def listdir(dir_path):
    fnames = os.listdir(dir_path)
    return [
        fname for fname in fnames if fname not in ('.DS_Store', '.ipynb_checkpoints')
    ]


def diarize_audio(wav_path, out_path, padding=0.15):
    punct_table = str.maketrans('', '', string.punctuation)

    for wav_hash in tqdm(listdir(wav_path)):
        if wav_hash + '_user.wav' not in os.listdir(wav_path + wav_hash): continue
        if wav_hash + '.csv' in os.listdir(out_path): continue

        diarization = model.transcribe(
            f'{wav_path}{wav_hash}/{wav_hash}_user.wav',
            language='russian', word_timestamps=True
        )

        data = []
        for segment_num, segment in enumerate(diarization['segments']):
            for word in segment['words']:
                data.append([
                    wav_hash, segment_num,
                    word['word'].lower().translate(punct_table),
                    word['start'], word['end'], word['probability']
                ])

        data = pd.DataFrame(
            data, columns=('audio', 'segment_num', 'word', 'start', 'end', 'prob')
        )

        data['end'] += padding

        data.to_csv(out_path + wav_hash + '.csv')


def find_most_common_words(drz_path, out_path, min_prob, threshold):
    word_data = pd.concat([
        pd.read_csv(
            drz_path + fname, header=0, index_col=0
        ) for fname in listdir(drz_path)
    ])

    word_data = word_data[word_data.prob > min_prob].drop(columns='prob')

    freq_table = np.vstack(np.unique(word_data['word'].values, return_counts=True)).T
    
    n_wavs = [round(
        100 * word_data[word_data['word'] == word]['audio'].nunique() / word_data['audio'].nunique(), 1
    ) for word, _ in freq_table]
    
    freq_table = np.hstack((freq_table, np.array(n_wavs).reshape(-1, 1)))
    
    freq_table = freq_table[freq_table[:, 2].argsort()[::-1]]

    pd.DataFrame(
        freq_table, columns=('word', 'freq', 'n_wavs (%)')
    ).to_csv(
        f'{out_path}word_freqs_minp={min_prob}.csv', index=False
    )

    most_common_words = freq_table[freq_table[:, 2] >= threshold]

    word_data = word_data[
        word_data['word'].apply(lambda x: x in most_common_words)
    ].reset_index().drop(columns='index')
    
    pd.DataFrame(word_data).to_csv(
        f'{out_path}chosen_words_minp={min_prob}_thrs={threshold}.csv',
        index=False
    )

    return word_data


def slice_waveform(waveform, sample_rate, timecodes):
    hzcodes = np.array([[
        timecode[0] * sample_rate, timecode[1] * sample_rate
    ] for timecode in timecodes], dtype=int)

    sliced_waveform = []
    for hzcode in hzcodes:
        sliced_waveform += waveform[:, hzcode[0]:hzcode[1]]

    return torch.hstack(sliced_waveform)


def extract_most_common_words(wav_path, out_path, word_data):
    for wav_hash in word_data.audio.unique():
        waveform, sample_rate = torchaudio.load(
            f'{wav_path}{wav_hash}/{wav_hash}_user.wav'
        )

        timecodes = word_data[word_data.audio == wav_hash][['start', 'end']].values
        
        sliced_waveform = slice_waveform(waveform, sample_rate, timecodes)
        sliced_waveform = sliced_waveform.resize_(1, sliced_waveform.shape[0])

        torchaudio.save(out_path + wav_hash + '.wav', sliced_waveform, sample_rate)


def make_word_experiment(wav_path, res_path, drz_path, sls_path, min_prob, threshold):
    if res_path[:-1] not in os.listdir(): os.makedirs(res_path)

    if drz_path[:-1] not in os.listdir(res_path): os.makedirs(res_path + drz_path)

    diarize_audio(wav_path, res_path + drz_path)

    word_data = find_most_common_words(res_path + drz_path, res_path, min_prob, threshold)

    if sls_path[:-1] not in os.listdir(res_path): os.makedirs(res_path + sls_path)

    extract_most_common_words(wav_path, res_path + sls_path, word_data)


if __name__ == '__main__':
    wav_path, res_path, drz_path, min_prob, threshold = sys.argv[1:]
    sls_path = f'wav_sliced_minp={min_prob}_thrs={threshold}/'
    
    make_word_experiment(wav_path, res_path, drz_path, sls_path, min_prob, threshold)
