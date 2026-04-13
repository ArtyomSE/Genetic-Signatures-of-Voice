import numpy as np
import pandas as pd
import torch
import torchaudio
import whisper

import re
import string
import os


def diarize_audio():
    model = whisper.load_model("large")

    punct_table = str.maketrans('', '', string.punctuation)

    for audio in os.listdir('wav-parser/audio/result'):
        if audio in ('.DS_Store', '.ipynb_checkpoints'): continue
        if f'{audio}_words.csv' in os.listdir(f'word_diarization'): continue

        diarization = model.transcribe(
            f'wav-parser/audio/result/{audio}/{audio}_user.wav',
            language='russian', word_timestamps=True
        )

        data = []
        for segment_num, segment in enumerate(diarization['segments']):
            for word in segment['words']:
                data.append([
                    audio, segment_num,
                    word['word'].lower().translate(punct_table),
                    word['start'], word['end'], word['probability']
                ])

        data = pd.DataFrame(
            data, columns=('audio', 'segment_num', 'word', 'start', 'end', 'prob')
        )

        data.to_csv('word_diarization/' + audio + '_words.csv')


def find_most_common_words(n_top, min_prob=0.75):
    word_data = pd.concat([
        pd.read_csv(
            'word_diarization/' + fname, header=0, index_col=0
        ) for fname in os.listdir('word_diarization')
    ])

    word_data = word_data[word_data.prob > min_prob].drop(columns='prob')

    freq_table = np.array(sorted(
        np.vstack(np.unique(word_data.word.values, return_counts=True)).T,
        key=lambda x: x[1], reverse=True
    ))

    pd.DataFrame(freq_table, columns=('word', 'freq')).to_csv('word_frequencies.csv')

    most_common_words = freq_table[:n_top, 0]

    word_data = word_data[
        word_data.word.apply(lambda x: x in most_common_words)
    ].reset_index().drop(columns='index')

    return freq_table[:n_top, 0], word_data


def slice_waveform(waveform, sample_rate, timecodes):
    hzcodes = np.array([[
        timecode[0] * sample_rate, timecode[1] * sample_rate
    ] for timecode in timecodes], dtype=int)

    sliced_waveform = []
    for hzcode in hzcodes:
        sliced_waveform += waveform[:, hzcode[0]:hzcode[1]]

    return torch.hstack(sliced_waveform)


def main():
    if 'word_diarization' not in os.listdir():
        os.makedirs('word_diarization')

    diarize_audio()

    most_common_words, word_data = find_most_common_words(10)

    if 'word_audio' not in os.listdir():
        os.makedirs('word_audio')

    for audio in word_data.audio.unique():
        waveform, sample_rate = torchaudio.load(
            f'wav-parser/audio/result/{audio}/{audio}_user.wav'
        )
        timecodes = word_data[word_data.audio == audio][['start', 'end']].values

        torchaudio.save(
            f'word_audio/{audio}.wav',
            slice_waveform(waveform, sample_rate, timecodes),
            sample_rate
        )


if __name__ == "__main__":
    main()