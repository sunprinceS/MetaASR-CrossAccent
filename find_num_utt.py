#!/usr/bin/env python3  

from pathlib import Path
import numpy as np
# import src.monitor.logger as logger


ACCENT_LIST=['african','australia','bermuda','canada','england','hongkong','indian','ireland','malaysia','newzealand', 'scotland', 'philippines', 'singapore', 'southatlandtic', 'us', 'wales', 'all']

if __name__ == "__main__":
    cur_path = Path.cwd()

    for accent in ACCENT_LIST:
        print(accent)
        for mode in ['train', 'dev', 'test']:
            data_path = cur_path.joinpath('data',accent, mode)
            print(f"{mode}: {len(np.load(data_path.joinpath('ilens.npy')))}")

        print('\n')
