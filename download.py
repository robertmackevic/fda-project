import shutil
from pathlib import Path

import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.datasets.speechcommands import FOLDER_IN_ARCHIVE, URL
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent
DATASET_DIR = ROOT_DIR / FOLDER_IN_ARCHIVE

CLASSES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
SAMPLE_RATE = 16000

if __name__ == "__main__":
    print("Downloading dataset...")
    dataset = SPEECHCOMMANDS(root=ROOT_DIR, download=True)
    original_dataset_dir = DATASET_DIR / URL

    with (original_dataset_dir / "validation_list.txt").open("r") as file:
        original_validation_list = file.readlines()

    with (original_dataset_dir / "testing_list.txt").open("r") as file:
        original_testing_list = file.readlines()

    for class_name in CLASSES:
        (DATASET_DIR / class_name).mkdir(parents=True, exist_ok=True)

    train_filenames, val_filenames, test_filenames = [], [], []
    train_speakers, val_speakers, test_speakers = [], [], []

    print("Processing dataset...")
    for i in tqdm(range(len(dataset))):
        src_filename, _, class_name, speaker, _ = dataset.get_metadata(n=i)

        if class_name not in CLASSES:
            continue

        src_filepath = DATASET_DIR / src_filename
        waveform, sample_rate = torchaudio.load(src_filepath, normalize=False)

        # Filter sample if it's not 16kHz or not 1 second in duration
        if sample_rate != SAMPLE_RATE or waveform.shape[1] != SAMPLE_RATE:
            continue

        dest_filename = f"{class_name}/{src_filepath.name}"
        shutil.move(src_filepath, DATASET_DIR / dest_filename)
        dest_filename += "\n"

        if dest_filename in original_validation_list:
            val_filenames.append(dest_filename)
            val_speakers.append(speaker)

        elif dest_filename in original_testing_list:
            test_filenames.append(dest_filename)
            test_speakers.append(speaker)

        else:
            train_filenames.append(dest_filename)
            train_speakers.append(speaker)

    with (DATASET_DIR / "train.txt").open("w") as file:
        file.writelines(train_filenames)

    with (DATASET_DIR / "val.txt").open("w") as file:
        file.writelines(val_filenames)

    with (DATASET_DIR / "test.txt").open("w") as file:
        file.writelines(test_filenames)

    print(
        f"Training samples: {len(train_filenames)}\n"
        f"Training speakers: {len(set(train_speakers))}\n"
        f"Validation samples: {len(val_filenames)}\n"
        f"Validation speakers: {len(set(val_speakers))}\n"
        f"Testing samples: {len(test_filenames)}\n"
        f"Testing speakers: {len(set(test_speakers))}\n"
        f"Total samples: {len(train_filenames) + len(val_filenames) + len(test_filenames)}\n"
        f"Total speakers: {len(set(train_speakers + val_speakers + test_speakers))}"
    )

    print("Removing original dataset...")
    shutil.rmtree(original_dataset_dir, ignore_errors=True)
