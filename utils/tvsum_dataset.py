import h5py
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
class TVSumLLaMADataset(Dataset):

    def __init__(self, mode, split_idx, llama_embedding='llama_emb/tvsum_sum/',
                 audio_path='audio_features/tvsum_whisper.h5', visual_path='clip_features/tvsum_clip.h5',
                 tr_val_ts=False):
        self.mode = mode
        self.dataset = 'TVSum/eccv16_dataset_tvsum_google_pool5.h5'
        self.userprompt = '{}user_prompt/user_prompt_pool.h5'.format(llama_embedding)
        self.generation = '{}gen/gen_pool.h5'.format(llama_embedding)
        self.split_file = 'dataset/tvsum_splits.json'
        self.video_data = h5py.File(self.dataset, 'r')
        self.llama_emb_userprompt = h5py.File(self.userprompt, 'r')
        self.llama_emb_generation = h5py.File(self.generation, 'r')
        self.audio_data = h5py.File(audio_path, 'r')
        self.visual_data = h5py.File(visual_path, 'r')

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())
            self.data = dict(self.data[split_idx])

        # deterministic train/val carve-out: first 80% of train_keys stay train,
        # last 20% become val. test_keys are left untouched.
        if tr_val_ts:
            train_keys = self.data['train_keys']
            n_train = int(round(0.8 * len(train_keys)))
            self.data['train_keys'] = train_keys[:n_train]
            self.data['val_keys'] = train_keys[n_train:]
        elif mode == 'val':
            raise ValueError("mode='val' requires tr_val_ts=True")


    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name

        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.as_tensor(np.array(self.video_data[video_name + '/gtscore']))
        d['llama_embedding_userprompt'] = torch.as_tensor(np.array(self.llama_emb_userprompt[str(video_name)]))
        d['llama_embedding_generation'] = torch.as_tensor(np.array(self.llama_emb_generation[str(video_name)]))
        d['audio_features'] = torch.as_tensor(np.array(self.audio_data[video_name + '/features']))
        d['visual_features'] = torch.as_tensor(np.array(self.visual_data[video_name + '/features']))

        if self.mode != 'train':
            d['n_frames'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frames']))
            d['picks'] = torch.as_tensor(np.array(self.video_data[video_name + '/picks']))
            d['change_points'] = torch.as_tensor(np.array(self.video_data[video_name + '/change_points']))
            d['n_frame_per_seg'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frame_per_seg']))
            d['gt_summary'] = torch.as_tensor(np.array(self.video_data[video_name + '/user_summary']))
        
        return d
    

class TrainBatchCollator(object):
    def __call__(self, batch):
        video_name, video_filename, features, gtscore, llama_embedding_userprompt, llama_embedding_generation = [],[],[],[],[],[]
        audio_features, visual_features = [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                llama_embedding_userprompt.append(data['llama_embedding_userprompt'])
                llama_embedding_generation.append(data['llama_embedding_generation'])
                audio_features.append(data['audio_features'])
                visual_features.append(data['visual_features'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], llama_embedding_userprompt)))
        max_len = max(list(map(lambda x: x.shape[0], llama_embedding_userprompt)))
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)
        llama_embedding_userprompt = pad_sequence(llama_embedding_userprompt, batch_first=True)
        llama_embedding_generation = pad_sequence(llama_embedding_generation, batch_first=True)
        audio_features = pad_sequence(audio_features, batch_first=True)
        visual_features = pad_sequence(visual_features, batch_first=True)

        batch_data = {'video_name' : video_name,  'features' : frame_feat, 'gtscore':gtscore, 'mask':mask,
                      'llama_embedding_userprompt':llama_embedding_userprompt, 'llama_embedding_generation':llama_embedding_generation,
                      'audio_features': audio_features, 'visual_features': visual_features}
        return batch_data

class ValBatchCollator(object):
    def __call__(self, batch):
        video_name, video_filename, features, gtscore, llama_embedding_userprompt, llama_embedding_generation = [],[],[],[],[],[]
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []
        audio_features, visual_features = [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                llama_embedding_userprompt.append(data['llama_embedding_userprompt'])
                llama_embedding_generation.append(data['llama_embedding_generation'])
                audio_features.append(data['audio_features'])
                visual_features.append(data['visual_features'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], llama_embedding_userprompt)))
        max_len = max(list(map(lambda x: x.shape[0], llama_embedding_userprompt)))
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)
        llama_embedding_userprompt = pad_sequence(llama_embedding_userprompt, batch_first=True)
        llama_embedding_generation = pad_sequence(llama_embedding_generation, batch_first=True)
        audio_features = pad_sequence(audio_features, batch_first=True)
        visual_features = pad_sequence(visual_features, batch_first=True)


        batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask,
                      'llama_embedding_userprompt':llama_embedding_userprompt, 'llama_embedding_generation':llama_embedding_generation,
                      'audio_features': audio_features, 'visual_features': visual_features,
                      'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps,
                      'gt_summary': gt_summary}
        return batch_data