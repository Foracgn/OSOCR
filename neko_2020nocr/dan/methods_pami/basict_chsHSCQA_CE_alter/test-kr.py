from __future__ import print_function

from cfgs_scene import scene_cfg_tekr;
from neko_2020nocr.dan.danframework.HEXOScvpr21 import HDOS2C;


#---------------------------------------------------------
#--------------------------Begin--------------------------
#---------------------------------------------------------
if __name__ == '__main__':
    cfgs=scene_cfg_tekr(root_override="/run/media/lasercat/ssddata/pamidump/ablchs_md_scene/")
    runner=HDOS2C(cfgs);
    runner.run(dbgpath="/run/media/lasercat/ssddata/pamidump/kr/");

