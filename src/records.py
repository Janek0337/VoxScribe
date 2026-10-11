import os
from datacollective import download_dataset
from src.config import DATA_DIR, RAW_DATA_DIR
import tarfile
import ffmpeg
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def normalize_sentence(sentence: str):
    small = sentence.lower()
    good = "".join(symbol for symbol in small if symbol.isalpha() or symbol.isspace())
    return good


def convert_audio(path_from, path_to):
    path_from_str = str(path_from)
    path_to_str = str(path_to)
    
    if os.path.exists(path_to_str) and os.path.getsize(path_to_str) > 0:
        return (True, path_from_str, "Pominięto (już istnieje)")
        
    try:
        (
            ffmpeg
            .input(path_from_str)
            .output(
                path_to_str,
                format='wav',
                acodec='pcm_s16le',
                ar=16000,
                ac=1
            )
            .run(quiet=True, overwrite_output=True)
        )
        return (True, path_from_str, "Sukces")
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode('utf8') if e.stderr else str(e)
        return (False, path_from_str, error_msg)

def download_all_records():
    dataset_id = "cmu5wsxwp00e0nq07m8v0274j"
    tar_name = 'common-voice-scripted-speech-27-0-polish-83b3f213.tar.gz'

    # downloading
    print(f"Downloading records...")    
    if not tar_name in os.listdir(RAW_DATA_DIR):
        records_dir = download_dataset(dataset_id)
        print(f"Finished! Downloaded to: {records_dir}")
    else:
        print('Recordings already downloaded')

    # extracting
    tar_path = RAW_DATA_DIR / tar_name
    unpacked_root = RAW_DATA_DIR / 'cv-corpus-27.0-2026-09-11' / 'pl'

    if not unpacked_root.exists():
        print('Extracting...')
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=RAW_DATA_DIR)
        print('Extracted data')
    else:
        print('Data already extracted')

    sentences = pd.read_csv(unpacked_root / 'validated.tsv', sep='\t', usecols=['path', 'sentence'])

    clip_times = pd.read_csv(unpacked_root / 'clip_durations.tsv', sep='\t')
    clip_times = clip_times.rename(columns={'clip': 'path'})

    records = pd.merge(sentences, clip_times, on='path')
    records['path'] = records['path'].apply(lambda x: x.replace('.mp3', '.wav'))
    records['sentence'] = records['sentence'].apply(normalize_sentence)

    records.to_csv(DATA_DIR / 'audio_data.csv', index=False)

    # conversion to standard properties
    clips_dir = unpacked_root / 'clips'

    tasks = []
    for filename in records['path']:
        input_path = clips_dir / filename
        output_path = DATA_DIR / filename.replace('.mp3', '.wav')
        tasks.append((input_path, output_path))
        
    print(f"Starting file conversion: {len(tasks)} files...")
    errors = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(convert_audio, task[0], task[1]): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Conversion"):
            success, path_from, message = future.result()
            if not success:
                errors.append((path_from, message))

    if errors:
        print(f"\nFinished converting, problem with {len(errors)} files.")
        for path, msg in errors:
            print(f"File: {path}\nError: {msg}\n{'-'*40}\n")
    else:
        print("\nAll files converted successfully")

if __name__ == '__main__':
    download_all_records()