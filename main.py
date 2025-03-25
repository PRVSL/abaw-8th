"""
Author: PRVSL
https://pr.ai.vn
"""
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping, BasePredictionWriter

from abaw_dataset import ABAWDataModule
from models import ABAWModel
from pathlib import Path
import numpy as np
import pandas as pd
import os


class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval='epoch', task='AU'):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.task = task

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Make prediction folder
        predictions_postfix = 0
        while os.path.isdir(os.path.join(self.output_dir, "predictions_{}".format(predictions_postfix))):
            predictions_postfix += 1

        prediction_folder = os.path.join(self.output_dir, "predictions_{}".format(predictions_postfix))

        os.makedirs(prediction_folder, exist_ok=True)

        torch.save(predictions, os.path.join(self.output_dir, "predictions.pt"))
        print('Saved file: ', os.path.join(self.output_dir, "predictions.pt"))
        print(f'\nCreating txt file to {prediction_folder}...\n')
        preds = []
        # ytruths = []
        # findexes = []
        video_ids = []
        for k in predictions:
            preds.append(k[0])
            image_names = k[1]
            video_ids += [x.split('/')[0] for x in image_names]
            # findexes.append([int(x.split('/')[1].split('.')[0]) for x in image_names])

        if self.task == 'AU':
            preds = np.squeeze(1 * (torch.concat(preds).float().numpy() >= 0.5))
            header_name = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']
        elif self.task == 'VA':
            preds = np.squeeze(torch.concat(preds).float().numpy().astype(float))
            header_name = ['valence', 'arousal']
        elif self.task == 'EXPR':
            preds = np.squeeze(torch.concat(preds).float().numpy())
            header_name = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
        else:
            raise ValueError('Do not support write prediction for {} task'.format(self.task))

        # ytruths = np.squeeze(torch.concat(ytruths).numpy())
        # findexes = np.array(findexes)
        video_ids = np.array(video_ids)

        video_id_uq = list(pd.unique(video_ids))
        num_classes = preds.shape[-1]

        sample_prediction = pd.read_csv(f'dataset/test_set/CVPR_8th_ABAW_{self.task}_test_set_example.txt')
        sample_prediction_indexes = np.array([x.split('/')[0] for x in sample_prediction['image_location'].values])

        all_prediction = []

        for vd in video_id_uq:
            num_sample_pred = np.sum(sample_prediction_indexes == vd)
            list_row = video_ids == vd
            list_preds = preds[list_row, :, :].reshape(-1, preds.shape[-1])
            # list_indexes = findexes[list_row, :].reshape(-1)

            # if np.sum(np.diff(list_indexes) < 0):
            #     print('Please check: {}. Indexes are not consistent'.format(vd))
            # Remove duplicate rows. Because we split sequentially => only padding at the end

            num_frames = num_sample_pred  # len(np.unique(list_indexes))

            write_prediction = list_preds[:num_frames, :]
            write_prediction_index = np.array(
                ['{}/{:05d}.jpg'.format(vd, idx + 1) for idx in range(num_frames)]).reshape(-1, 1)

            cur_prediction = np.concatenate((write_prediction_index, write_prediction), axis=1)

            all_prediction.append(cur_prediction)

        all_prediction = np.concatenate(all_prediction)

        if self.task in ['AU', 'VA']:
            pd.DataFrame(data=all_prediction, columns=['image_location'] + header_name).to_csv(
                '{}/predictions.txt'.format(prediction_folder), index=False)
        elif self.task == 'EXPR':
            with open('{}/predictions.txt'.format(prediction_folder), 'w') as fd:
                fd.write(','.join(['image_location'] + header_name) + '\n')
                fd.write('\n'.join(all_prediction))
        else:
            raise ValueError('Do not support write prediction for {} task'.format(self.task))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ABAW - AFFWILD2 training')
    parser.add_argument('--data_dir', type=str, default='/media/vthuynh/XProject/emotion/abaw8/affwild2/')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--task', type=str, default='AU', choices=['AU', 'VA', 'EXPR'])
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--img_size', type=int, default=112, help='Image size')
    parser.add_argument('--seq_len', type=int, default=32, help='Sequence length')
    parser.add_argument('--wd', type=float, default=2e-5, help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--fc_alpha', type=float, default=0.75, help='Focal alpha')
    parser.add_argument('--fc_gamma', type=float, default=2., help='Focal gamma')
    parser.add_argument('--mixed_precision', type=str, default='bf16-mixed', help='Mixed precision',
                        choices=['bf16-mixed', '16-mixed'])
    parser.add_argument('--train_ratio', type=float, default=0.5, help='Limit train batches')
    parser.add_argument('--n_mixer', type=int, default=3, help='Number of mixer blocks')
    parser.add_argument('--testing', type=str, default='', help='Generate predictions, path to checkpoint')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(args.seed)

    dm = ABAWDataModule(args.data_dir, args.task, seq_len=args.seq_len, img_size=args.img_size, num_folds=0,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size)

    mixed_precision = args.mixed_precision

    if os.path.exists(args.testing):
        print('Loading checkpoint from {}'.format(args.testing))
        model = ABAWModel.load_from_checkpoint(args.testing)
        cbacks = [RichProgressBar(leave=True),
                  PredictionWriter('/'.join(x for x in args.testing.split('/')[:-1]), 'epoch', args.task)]
        is_training = False
    else:
        model = ABAWModel(task=args.task, seq_len=args.seq_len, learning_rate=args.lr, focal_alpha=args.fc_alpha,
                          focal_gamma=args.fc_gamma,
                          label_smoothing=args.label_smoothing, wd=args.wd, batch_size=args.batch_size,
                          n_mixer=args.n_mixer)

        cbacks = [RichProgressBar(leave=False),
                  ModelCheckpoint(monitor='val_metric', mode='max', save_weights_only=True),
                  EarlyStopping(monitor='val_metric', patience=3, mode='max')]
        is_training = True

    trainer = pl.Trainer(max_epochs=args.epochs, precision=mixed_precision, num_sanity_val_steps=0,
                         limit_train_batches=args.train_ratio, callbacks=cbacks,
                         default_root_dir=args.data_dir)
    if is_training:
        trainer.fit(model, dm)
        print('Evaluating on val with checkpoint ', trainer.checkpoint_callback.best_model_path)
        trainer.validate(model, dm, ckpt_path='best')
    else:
        dm.setup()
        trainer.validate(model, dm)
        trainer.predict(model, dm.test_dataloader())
        print('Finished')
