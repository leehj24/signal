import copy
import os
import shutil
import wave

import librosa
import numpy as np
import scipy
import scipy.fftpack
import soundfile
from scipy.io import wavfile

SCALE = 5000
SAMPLE_RATE = 48000
NUM_HPS = 4  # max number of harmonic product spectrum
CONCERT_PITCH = 440  # defining a1
ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
ALL_NOTES_KOR = ["라", "라#", "시", "도", "도#", "레", "레#", "미", "파", "파#", "솔", "솔#"]


def initFile():
    shutil.rmtree("./mod_wav")
    os.mkdir("./mod_wav")


def find_pitch(in_data):
    hann_samples = in_data * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples) // 2])

    mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1 / NUM_HPS), np.arange(0, len(magnitude_spec)),
                              magnitude_spec)
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # normalize it

    hps_spec = copy.deepcopy(mag_spec_ipol)

    for i in range(NUM_HPS):
        tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))], mag_spec_ipol[::(i + 1)])
        if not any(tmp_hps_spec):
            break
        hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind / NUM_HPS

    return max_freq


def pitch_modulate(note, target, exist):
    dataMod = []
    if f"{note}{target}" in exist:
        return
    else:
        infile = f"{note}4.wav"
        w2 = wave.open(infile, 'rb')
        dataMod = [w2.getparams(), w2.readframes(w2.getnframes())]
        w2.close()

        output2 = wave.open(f"./mod_wav/{note}{target}.wav", 'wb')
        target = target - 4
        output2.setparams(dataMod[0])
        output2.setframerate(48000 * (2.0 ** target))
        output2.writeframes(dataMod[1])
        output2.close()
        exist.append(f"{note}{target}")


filename = "plane.wav"
initFile()

data, fs = soundfile.read(filename)
data = data[:, 0]  # convert to mono
length = len(data)
countForNote = (length / SAMPLE_RATE) * SCALE
WINDOW = length / countForNote
length_iter = length / WINDOW
countForNote = int(countForNote)
WINDOW = int(WINDOW)
length_iter = int(length_iter)

ArrayNote = []

tmp_data = np.empty(shape=SAMPLE_RATE)
HANN_WINDOW = np.hanning(48000)
for j in range(length_iter):
    if ((WINDOW * j + SAMPLE_RATE) >= length):
        for k in range(SAMPLE_RATE):
            if ((WINDOW * j + k) >= length):
                break
            tmp_data[k] = data[WINDOW * j + k]
    else:
        tmp_data = data[WINDOW * j : WINDOW * j + SAMPLE_RATE]
    pitch = find_pitch(tmp_data)
    if pitch == 0.0:
        continue
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    ArrayNote.append(closest_note)
    if j % 100 == 0:
        print(f'{j}/{countForNote}')

countedNote = []
countArray = []

count = 1
prev = ArrayNote[0]
for i, v in enumerate(ArrayNote):
    if i == 0:
        continue
    else:
        if prev == v:
            count = count + 1
        else:
            countedNote.append(prev)
            countArray.append(count)
            count = 1
            prev = v

countedNote.append(prev)
countArray.append(count)
print(countedNote)
print(countArray)

exist = []
for i, j in enumerate(countedNote):
    if len(j) == 4:
        targetNote = j[:-2]
        targetOctave = int(j[-2:])
    else:
        targetNote = j[:-1]
        targetOctave = int(j[-1:])
    pitch_modulate(targetNote, targetOctave, exist)

data = []
for i in ArrayNote:
    infile = f"./mod_wav/{i}.wav"
    w = wave.open(infile, 'rb')
    data.append([w.getparams(), w.readframes(w.getnframes())])
    w.close()

# make smooth

final = wave.open("./mod_wav/result.wav", 'wb')
final.setparams(data[0][0])
final.setframerate(48000)
for i in range(len(data)):
    final.writeframes(data[i][1])
final.close()

song, fs = librosa.load("./mod_wav/result.wav")
times_faster = librosa.effects.time_stretch(song, SCALE // 10)
scipy.io.wavfile.write("./mod_wav/result.wav", fs, times_faster)

os.system("start ./mod_wav/result.wav")
