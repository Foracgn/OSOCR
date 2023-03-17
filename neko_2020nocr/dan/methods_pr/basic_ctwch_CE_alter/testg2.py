# coding:utf-8
from __future__ import print_function

import sys

from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C
from cfgs_scene5 import scene_cfg_te as scene_cfg_te_5
from cfgs_scene10 import scene_cfg_te as scene_cfg_te_10
from cfgs_scene15 import scene_cfg_te as scene_cfg_te_15
from cfgs_scene20 import scene_cfg_te as scene_cfg_te_20

DICT = {
    "500": scene_cfg_te_5,
    "1000": scene_cfg_te_10,
    "1500": scene_cfg_te_15,
    "2000": scene_cfg_te_20,
}

# ---------------------------------------------------------
# --------------------------Begin--------------------------
# ---------------------------------------------------------
if __name__ == '__main__':
    for k in DICT:
        cfgs = DICT[k]("/home/sunhongyi/桌面/sunhongyi/OSOCR/prfinal/ctw_extra/")
        cfgs.dataset_cfgs['dataloader_test']['num_workers'] = 0
        print(cfgs.dataset_cfgs)
        input()
        runner = HDOS2C(cfgs)
        runner.run()
        print(k, "Done")
