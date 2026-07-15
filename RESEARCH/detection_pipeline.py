import os
import subprocess
from datetime import datetime 
import pandas as pd
from remotezip import RemoteZip 
from feat import Detectorv2
import warnings

REMOTE, SERVE = 'gdrive-candor:', 'http://localhost:8080'
OUT_DIR, TEMP_DIR, METADATA = 'au_activity', 'data/temp', 'data/asym_data.csv'
keep_cols = ['frame', 'approx_time', 'FaceScore', 'input',
             'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
             'AU10', 'AU11', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20',
             'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43']

detector = Detectorv2(device='mps')

# builds zipfile path for easy access 
def zip_url(d): return f'{SERVE}/candor_raw/raw_media_part_{int(d):03d}.zip'

def find_video(zf, convo, uid): 
    hits = [n for n in zf.namelist() if convo in n and 
            '/processed/' in n and n.endswith(f'{uid}.mp4')] 
    
    # there should only be one match; if not, make sure to flag it 
    if len(hits) != 1: 
        # raise error so it pops out in terminal 
        raise FileNotFoundError(f'{convo}/{uid}: expected 1 video, found {len(hits)}')
    return hits[0] 

def alr_done():
    # skips videos where the respective csv's havae already been uploaded

    res = subprocess.run(['rclone', 'lsf', '-R', '--files-only', f'{REMOTE}{OUT_DIR}/'],
                         capture_output=True, text=True)
    
    # if nothings been uploaded, return empty set 
    if res.returncode != 0: return set() 
    return set(res.stdout.splitlines()) # all csvs uploaded so far 

def extract_aus(zf, convo, side, uid, done): 
    # skip if csv is already in gdrive 
    if f'{convo}/{side}.csv' in done: return 'skip'

    # partially unzip + extract video from dir 
    filepath = find_video(zf, convo, uid)
    video = zf.extract(filepath, TEMP_DIR)
    csv = f'{TEMP_DIR}/{convo}_{side}.csv' 

    # try to extract au activity from video
    # if it crashes, make sure old files are deleted 
    try: 
        data = detector.detect(video, data_type='video', skip_frames=5,
                               batch_size=32, num_workers=0)
        data[keep_cols].to_csv(csv, index=False)
        subprocess.run(['rclone', 'copyto', csv, f'{REMOTE}{OUT_DIR}/{convo}/{side}.csv'], check=True)
    finally: 
        if os.path.exists(video): os.remove(video)
        if os.path.exists(csv): os.remove(csv) 

    return 'done'
    
def main():
    # ignore benign warning about division w/ zero
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*encountered in matmul")

    # if temp dir already exists, skip 
    os.makedirs(TEMP_DIR, exist_ok=True)
    metadata = pd.read_csv(METADATA)
    done = alr_done() 
    total = len(metadata)

    # count how many convos are fully finished 
    count = 0 
    for r in metadata.itertuples():
        if f'{r.convo_id}/left.csv' in done and f'{r.convo_id}/right.csv' in done: 
            count += 1 
        
    print(f'{count}/{total} convos already complete, resuming extraction\n', flush=True)

    r_start = datetime.now() # for logging time 

    
    for i, r in enumerate(metadata.itertuples(), start=1):
        # skips if au activity is already extracted 
        if f'{r.convo_id}/left.csv' in done and f'{r.convo_id}/right.csv' in done: 
            continue 

        c_start = datetime.now() # for logging time 

        # extract aus for both speakers of a convo 
        with RemoteZip(zip_url(r.dir_num)) as zf: 
            for side, uid in [('left', r.left_id), ('right', r.right_id)]: 
                try:
                    status = extract_aus(zf, r.convo_id, side, uid, done)
                except Exception as e: 
                    # log the error and move on 
                    status = f'ERROR: {e}'
                print(f'[{i}/{total}] part {int(r.dir_num):03d} | {r.convo_id[:8]} | {side:5} | {status}', flush=True)

        end = datetime.now() 
        print(f'   convo time: {end - c_start} || total time: {end - r_start}\n', flush=True)
        
    print(f'all done!\ntotal run time: {datetime.now() - r_start}', flush=True)

if __name__ == '__main__': main() 

