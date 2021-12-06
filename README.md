# ReSORT

Repo for the dataset, and code for the associated paper "ReSORT: ReSORT: an ID-recovery multi-face tracking method for surveillance cameras". This reporsitory contains the annotations for dataset and the code from the associated paper, for the task of multi-face tracking.

## Dataset: 
We will release three face-tracking annotations files for three respective public dataset, include MSU-AVIS, LAB-dataset and Chokepoint.
## Code:
The code to produce multi-face tracking results:
```
bash install.sh
python3 test.py --network <DETECTOR> --tracker <DEFAULT_RESORT>
```

## Demo:
Our proposed tracker - ReSORT, compared to SORT and DeepSORT, decreases the number of switching IDs effectively.

[![Watch the video](https://github.com/tantm97/ReSORT/blob/main/demo_clip/Chokepoint_Demo.gif)](https://youtu.be/ijmlr71cksg)
