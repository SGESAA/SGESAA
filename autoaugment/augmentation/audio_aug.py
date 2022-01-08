from torch_audiomentations import Gain as G, PolarityInversion as PI, AddBackgroundNoise as ABN, \
    ApplyImpulseResponse as AIR, PeakNormalization as PN, Shift as S, ShuffleChannels as SC

BACKGROUND_PATH = [
    '/home/wizcheu/Workspace/AutoAugment/autoaugment/domain/audio/classification/res/bg.wav'
]
IR_PATH = [
    '/home/wizcheu/Workspace/AutoAugment/autoaugment/domain/audio/classification/res/impulse_response.wav'
]


def AddBackgroundNoise(audio, v, sr):
    return ABN(background_paths=BACKGROUND_PATH, p=v, sample_rate=sr)(audio)


def Gain(audio, v, sr):
    return G(p=v, sample_rate=sr)(audio)


def ImpulseResponse(audio, v, sr):
    return AIR(ir_paths=IR_PATH, p=v, sample_rate=sr)(audio)


def PeakNormalization(audio, v, sr):
    return PN(p=v, sample_rate=sr)(audio)


def PolarityInversion(audio, v, sr):
    return PI(p=v, sample_rate=sr)(audio)


def Shift(audio, v, sr):
    return S(p=v, sample_rate=sr)(audio)


def ShuffleChannels(audio, v, sr):
    return SC(p=v, sample_rate=sr)(audio)


audio_transformations = [
    # (AddBackgroundNoise, 0, 1),   # 有bug
    (Gain, 0, 1),
    (ImpulseResponse, 0, 1),
    (PeakNormalization, 0, 1),
    (PolarityInversion, 0, 1),
    (Shift, 0, 1),
    (ShuffleChannels, 0, 1),
]
