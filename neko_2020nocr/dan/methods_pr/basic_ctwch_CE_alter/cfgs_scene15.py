# coding:utf-8
from neko_2020nocr.dan.dataloaders.dataset_scene import *
from neko_2020nocr.dan.configs.global_cfg import get_test_cfg, get_save_cfgs, get_train_cfg
from neko_2020nocr.dan.configs.optimizers import get_dos_optim

from neko_sdk.root import find_data_root
from neko_2020nocr.dan.configs.nets.pami import get_dos_basic_cco as get_net
from neko_2020nocr.dan.methods_pr.CTWS_cfg import *
from neko_2020nocr.dan.methods_pr.loss_cfg import cls_emb as loss

cnt = 1500
prefix = "basic" + "_" + DSPRFIX + "_" + loss[0] + "_" + "alter_" + str(cnt)
print(prefix)
DSROOT = find_data_root()
dataset = DSCFG(cnt, T, "dict.pt", DSROOT)
PDICT = dataset["dict_dir"]


class scene_cfg:
    def __init__(self, root_override=None):
        self.global_cfgs = get_train_cfg()
        self.net_cfgs = get_net(PDICT, prefix, None, maxT=T)
        self.optimizer_cfgs = get_dos_optim()
        self.saving_cfgs = get_save_cfgs(prefix, root_override=root_override)
        self.dataset_cfgs = dataset
        self.loss_weight = loss[1]

    def mkdir(self, path_):
        paths = path_.split('/')
        command_str = 'mkdir '
        for i in range(0, len(paths) - 1):
            command_str = command_str + paths[i] + '/'
        command_str = command_str[0:-1]
        os.system(command_str)

    def showcfgs(self, s):
        for key in s.keys():
            print(key, s[key])
        print('')


class scene_cfg_te(scene_cfg):
    def __init__(self, root_override=None):
        self.global_cfgs = get_test_cfg()
        self.net_cfgs = get_net(PDICT, prefix, "E9", maxT=T, root_override=root_override)
        self.optimizer_cfgs = get_dos_optim()
        self.saving_cfgs = get_save_cfgs(prefix, root_override=root_override)
        self.dataset_cfgs = dataset
        self.loss_weight = loss[1]
