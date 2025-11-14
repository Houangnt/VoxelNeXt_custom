import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--output_dir', type=str, default='output_results', help='directory to save results')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Results will be saved to: {output_dir}')
    
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    all_results = []
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Processing sample index: \t{idx + 1}/{len(demo_dataset)}')
            
            sample_file = demo_dataset.sample_file_list[idx]
            sample_name = Path(sample_file).stem
            
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()

            score_thresh = 0.4
            mask = pred_scores > score_thresh

            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            result = {
                'sample_name': sample_name,
                'frame_id': idx,
                'pred_boxes': pred_boxes,
                'pred_scores': pred_scores,
                'pred_labels': pred_labels,
            }
            all_results.append(result)
            
            output_txt = output_dir / f'{sample_name}.txt'
            with open(output_txt, 'w') as f:
                for i in range(len(result['pred_boxes'])):
                    bbox = result['pred_boxes'][i]
                    class_label = result['pred_labels'][i]
                    
                    f.write(f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} {bbox[4]:.4f} {bbox[5]:.4f} {bbox[6]:.4f} {class_label}\n")
            
    
    logger.info(f'\n{"="*80}')
    logger.info(f'All results saved to: {output_dir}')
    logger.info(f'Total detections processed: {len(all_results)} frames')
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
#python tools/demo3.py --cfg_file tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext_doubleflip.yaml --data_path Test/ --ckpt voxelnext_nuscenes_kernel1.pth --ext .bin --output_dir output_results
#python tools/demo3.py --data_path Test/ --ckpt voxelnext_nuscenes_kernel1.pth --cfg_file tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml --ext .bin --output_dir output_results
    
