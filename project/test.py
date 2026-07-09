from feat import Detectorv2
import os 
import pandas as pd 
import json 

"""
DIRECTORY STRUCTURE

current path: /Users/mandiwu/pioneer_sp26/RESEARCH 
all data in: ~/data 
then: ~/data/raw_media_part_[XXX]/[code]/processed/[code].mp4 
OR use the key-values from processed/channel_map.json (L or R)
then: ~/data/raw_media_part_[XXX]/[code]/processed/[L_code].mp4 then swap out for [R_code]

to-do: 
-> determine a threshold for whether or not theres a valid face 
-> only start analyzing synchrony at a timeframe where both speakers' facescores >= threshold 
-> add a failsafe for in case theres >1 face in a video 

"""

# use a for loop to iterate in final pipeline 

os.chdir('data')
os.chdir('raw_media_part_001')
convo_codes = sorted(c for c in os.listdir() if os.path.isdir(c)) # make sure to sort so the list is the same everytime
curr = convo_codes[0]
os.chdir(f'{curr}/processed')

with open('channel_map.json', 'r') as file:
    channel_map = json.load(file)
print(channel_map)

# for professor fox: everything below this line will be the only code i will 
# have to eventually integrate into the colab pipeline. 

detector = Detectorv2(device='mps')
keep_cols = ['frame', 'approx_time', 'FaceScore', 'input', 'AU01', 'AU02',
             'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 'AU12',
             'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 
             'AU28', 'AU43']

# this line takes about 5-10 minutes to run, depending on how long the video 
# is, so it's my primary concern especially w/ colab potentially crashing mid-run
left = detector.detect(f'{os.getcwd()}/{channel_map.get('L')}.mp4', 
                        data_type='video',
                        skip_frames=5,
                        batch_size=32,
                        num_workers=0)

left[keep_cols].to_csv('left_aus.csv', index=False)
