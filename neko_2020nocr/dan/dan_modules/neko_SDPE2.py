import torch;
from torch.nn import functional as trnf
from neko_sdk.ocr_modules.prototypers.neko_nonsemantical_prototyper_core import neko_nonsematical_prototype_core_basic
from neko_sdk.ocr_modules.prototypers.neko_nonsemantical_prototyper_core import \
    neko_nonsematical_prototype_core_basic_g2rand
import regex


class neko_SDPE2(torch.nn.Module):
    def setupcore(self, backbone=None, val_frac=0.8):
        try:
            meta = torch.load(self.meta_path)
        except:
            meta = None
            print("meta loading failed")
        self.dwcore = neko_nonsematical_prototype_core_basic(self.nchannel, meta, backbone, None,
                                                             {"master_share": not self.case_sensitive,
                                                              "max_batch_size": 512,
                                                              "val_frac": val_frac,
                                                              "neg_servant": True
                                                              }, dropout=0.3);

    def __init__(self, meta_path, nchannel, case_sensitive, backbone=None, val_frac=0.8):
        super(neko_SDPE2, self).__init__()
        self.EOS = 0
        self.nchannel = nchannel
        self.case_sensitive = case_sensitive
        self.meta_path = meta_path
        self.setupcore(backbone, val_frac)

    def dump_all(self):
        return self.dwcore.dump_all();

    def sample_tr(self, text_batch):
        # if(not this.case_sensitive):
        #     label_batch=[l.lower() for l in text_batch]
        return self.dwcore.sample_charset_by_text(text_batch)

    def encode_fn_naive(self, tdict, label_batch):
        max_len = max([len(regex.findall(r'\X', s, regex.U)) for s in label_batch])
        out = torch.zeros(len(label_batch), max_len + 1).long() + self.EOS
        for i in range(0, len(label_batch)):
            cur_encoded = torch.tensor([tdict[char] if char in tdict else tdict["[UNK]"]
                                        for char in regex.findall(r'\X', label_batch[i], regex.U)])
            out[i][0:len(cur_encoded)] = cur_encoded
        return out

    def encode(self, proto, plabel, tdict, label_batch):
        if not self.case_sensitive:
            label_batch = [l.lower() for l in label_batch]
        return self.encode_fn_naive(tdict, label_batch)

    def decode(self, net_out, length, protos, labels, tdict, thresh=None):
        # decoding prediction into text with geometric-mean probability
        # the probability is used to select the more realiable prediction when using bi-directional decoders
        out = []
        out_prob = []
        net_out = trnf.softmax(net_out, dim=1)
        for i in range(0, length.shape[0]):
            current_idx_list = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[1][:,
                               0].tolist()
            current_text = ''.join([tdict[_] if 0 < _ <= len(tdict) else '' for _ in current_idx_list])
            current_probability = net_out[int(length[:i].sum()): int(length[:i].sum() + length[i])].topk(1)[0][:, 0]
            current_probability = torch.exp(torch.log(current_probability).sum() / current_probability.size()[0])
            if thresh is not None:
                filteredtext = []
                for i in range(len(current_text)):
                    if current_probability[i] > thresh:
                        filteredtext.append(current_text[i])
                    else:
                        filteredtext.append('⑨')
                current_text = ''.join(filteredtext)
            out.append(current_text)
            out_prob.append(current_probability)
        return out, out_prob


class neko_SDPE3_rand(neko_SDPE2):
    def setupcore(self, backbone=None, val_frac=0.8):
        try:
            meta = torch.load(self.meta_path)
        except:
            meta = None
            print("meta loading failed")
        self.dwcore = neko_nonsematical_prototype_core_basic_g2rand(self.nchannel, meta, backbone, None,
                                                                    {"master_share": not self.case_sensitive,
                                                                     "max_batch_size": 512,
                                                                     "val_frac": val_frac,
                                                                     "neg_servant": True
                                                                     }, dropout=0.3)

    def dump_all(self, force_idx=0):
        return self.dwcore.dump_all(idx=force_idx)
