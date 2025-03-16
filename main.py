"""
Author: Huynh Van Thong
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
        image_location = []
        for k in predictions:
            preds.append(k[0])
            image_location.append(k[1])

        image_location_arr = [np.array(x).T.flatten() for x in image_location]
        image_location_arr = np.concatenate(image_location_arr).reshape(-1, 1)

        if self.task == 'AU':
            preds_arr = np.concatenate([x.float().numpy().reshape(-1, 12) for x in preds])
            header_name = ['image_location', 'AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24',
                           'AU25', 'AU26']
            header_dtype = {'AU1': float, 'AU2': float, 'AU4': float, 'AU6': float, 'AU7': float, 'AU10': float,
                            'AU12': float, 'AU15': float, 'AU23': float, 'AU24': float, 'AU25': float, 'AU26': float}
            header_dtype_out = {'AU1': int, 'AU2': int, 'AU4': int, 'AU6': int, 'AU7': int, 'AU10': int,
                                'AU12': int, 'AU15': int, 'AU23': int, 'AU24': int, 'AU25': int, 'AU26': int}
        elif self.task == 'VA':
            preds_arr = np.concatenate([x.float().numpy().reshape(-1, 2) for x in preds])
            header_name = ['image_location', 'valence', 'arousal']
            header_dtype = {'valence': float, 'arousal': float}
        elif self.task == 'EXPR':
            # preds = np.squeeze(torch.concat(preds).float().numpy())
            # header_name = ['Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Other']
            raise NotImplementedError
        else:
            raise ValueError('Do not support write prediction for {} task'.format(self.task))

        all_prediction = np.hstack([image_location_arr, preds_arr])
        all_prediction_df = pd.DataFrame(all_prediction, columns=header_name)
        write_df = all_prediction_df.astype(header_dtype)
        write_df = write_df.groupby('image_location', sort=False).mean()
        if self.task == 'AU':
            write_df[write_df >= 0.5] = 1
            write_df[write_df < 0.5] = 0
            write_df = write_df.astype(header_dtype_out)

        if self.task in ['AU', 'VA']:
            write_df.to_csv(
                '{}/predictions.txt'.format(prediction_folder))
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
                  EarlyStopping(monitor='val_metric', patience=5, mode='max')]
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
        # trainer.validate(model, dm)
        trainer.predict(model, dm.test_dataloader())
        print('Finished')
