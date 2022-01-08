# 预处理，将音频路径和标签存为csv
import os
import pandas as pd

from sklearn.model_selection import train_test_split


def process_ESC():
    num_folds = 5
    csv_file = '/data/torch/datasets/audio/ESC-50/meta/esc50.csv'
    data_dir = '/data/torch/datasets/audio/ESC-50/audio'
    save_dir = '/data/torch/datasets/audio/ESC-50/preprocessed'
    audio_df = pd.read_csv(csv_file, skipinitialspace=True)
    training_audio_paths, training_labels = [], []
    training_audios = audio_df.loc[audio_df["fold"] != num_folds]
    training_audio_names = list(training_audios.filename.unique())
    for audio in training_audio_names:
        training_audio_paths.append(os.path.join(data_dir, audio))
        training_labels.append(training_audios.loc[
            training_audios["filename"] == audio]['target'].values[0])
    test_audio_paths, test_labels = [], []
    test_audios = audio_df.loc[audio_df["fold"] == num_folds]
    test_audio_names = list(test_audios.filename.unique())
    for audio in test_audio_names:
        test_audio_paths.append(os.path.join(data_dir, audio))
        test_labels.append(test_audios.loc[test_audios["filename"] == audio]
                           ['target'].values[0])
    print(
        f'training size: {len(training_audio_paths)}\ntest size: {len(test_audio_paths)}'
    )
    train_df = pd.DataFrame({
        'audio_path': training_audio_paths,
        'label': training_labels
    })
    test_df = pd.DataFrame({
        'audio_path': test_audio_paths,
        'label': test_labels
    })
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, 'train.csv'))
    test_df.to_csv(os.path.join(save_dir, 'test.csv'))
    print('save finished.')


def process_GTZAN():
    data_dir = '/data/torch/datasets/audio/GTZAN/genres_original'
    save_dir = '/data/torch/datasets/audio/GTZAN/preprocessed'
    audio_paths, labels = [], []
    for root, dirs, files in os.walk(data_dir):
        class_names = dirs
        break
    for _class in class_names:
        class_dir = os.path.join(data_dir, _class)
        for root, dirs, files in os.walk(class_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_paths.append(os.path.join(root, file))
                    labels.append(class_names.index(_class))
    training_audio_paths, test_audio_paths, training_labels, test_labels = train_test_split(
        audio_paths, labels, train_size=0.75, stratify=labels, random_state=42)
    os.makedirs(save_dir, exist_ok=True)
    train_df = pd.DataFrame({
        'audio_path': training_audio_paths,
        'label': training_labels
    })
    test_df = pd.DataFrame({
        'audio_path': test_audio_paths,
        'label': test_labels
    })
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, 'train.csv'))
    test_df.to_csv(os.path.join(save_dir, 'test.csv'))
    print(f'training size: {len(train_df)}\ntest size: {len(test_df)}')
    print('save finished.')


def process_USC():
    csv_file = '/data/torch/datasets/audio/UrbanSound8K/metadata/UrbanSound8K.csv'
    data_dir = '/data/torch/datasets/audio/UrbanSound8K/audio'
    save_dir = '/data/torch/datasets/audio/UrbanSound8K/preprocessed'
    audios = pd.read_csv(csv_file, skipinitialspace=True)
    training_audio_paths, training_labels = [], []
    test_audio_paths, test_labels = [], []
    for i in range(len(audios)):
        if audios.loc[i, 'fold'] == 1:
            test_audio_paths.append(
                os.path.join(data_dir, 'fold1', audios.loc[i,
                                                           'slice_file_name']))
            test_labels.append(audios.loc[i, 'classID'])
        else:
            training_audio_paths.append(
                os.path.join(data_dir,
                             f'fold{audios.loc[i, "fold"]}',
                             audios.loc[i, 'slice_file_name']))
            training_labels.append(audios.loc[i, 'classID'])
    # training_audios = audios.loc[audios["fold"] != 1]
    # training_audio_names = list(training_audios.slice_file_name.unique())
    # print(training_audio_names)
    # for audio in training_audio_names:
    #     training_audio_paths.append(os.path.join(data_dir, audio))
    #     training_labels.append(training_audios.loc[
    #         training_audios["slice_file_name"] == audio]['classID'].values[0])
    # test_audios = audios.loc[audios["fold"] == 1]
    # test_audio_paths, test_labels = [], []
    # test_audio_names = list(test_audios.slice_file_name.unique())
    # for audio in test_audio_names:
    #     test_audio_paths.append(os.path.join(data_dir, audio))
    #     test_labels.append(test_audios.loc[test_audios["slice_file_name"] == audio]
    #                        ['classID'].values[0])
    train_df = pd.DataFrame({
        'audio_path': training_audio_paths,
        'label': training_labels
    })
    test_df = pd.DataFrame({
        'audio_path': test_audio_paths,
        'label': test_labels
    })
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, 'train.csv'))
    test_df.to_csv(os.path.join(save_dir, 'test.csv'))
    print(f'training size: {len(train_df)}\ntest size: {len(test_df)}')
    print('save finished.')


if __name__ == '__main__':
    process_USC()
