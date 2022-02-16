# MODEL_ROOT='../../trained_models/';
MODEL_ROOT='/run/media/lasercat/ssddata/pamidump/trained_models/models/scene/ablchs_md_scene/';

def get_train_cfg(epochs=10):
    return {
        'state': 'Train',
        'epoch': epochs,
        'show_interval': 50,
        'test_interval': 1000,
        'test_miter':100,
    };

def get_test_cfg():
    return {
        'state': 'Test',
        'epoch': 10,
        'show_interval': 50,
        'test_interval': 1000,
        'test_miter': 100000,
    };
def get_save_cfgs(prefix,root_override=None):
    if(root_override is None):
        models=MODEL_ROOT+prefix+"_";
    else:
        models=root_override+prefix+"_";
    return {
        'saving_iter_interval': 20000,
        'saving_epoch_interval': 1,
        'saving_path': models,
    }