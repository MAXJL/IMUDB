import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from networks.limu_bert import LIMUBertModel4Pretrain
from box import Box


class Model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        # Below shall be moved to configs
        self.save_hyperparameters("config")
        self.starting_learning_rate = float(config['model']['hyper_params']['starting_learning_rate'])
        self.hyper_params = Box(config['model']['hyper_params'])
        self.limu_bert_mlm = LIMUBertModel4Pretrain(self.hyper_params)
        self.limu_bert_nsp = LIMUBertModel4Pretrain(self.hyper_params)

        self.mse_loss = F.mse_loss
        torch.cuda.empty_cache()


    # #要引入动态权重调整机制，我们可以在模型中实现一个自适应的权重更新策略，根据各个任务在验证集上的表现来调整它们的损失函数权重。
    # def adjust_task_weights(self, validation_performance):
    #     # 根据验证集表现动态调整任务权重
    #     base_weights = {
    #         'MLM': self.hyper_params.mlm_loss_weights,
    #         'denoise': self.hyper_params.denoise_loss_weights,
    #         'NSP': self.hyper_params.nsp_loss_weights,
    #         'continuity': self.hyper_params.continuity_loss_weight
    #         }
    #     performance_factor = 2.0
    #     new_weights = {}
    #     for task, performance in validation_performance.items():
    #         base_weight = base_weights.get(task, 1)  # 默认为1，如果没有找到配置的权重
    #         if performance is None :
    #             print(f"Invalid performance value for {task}: {performance}")
    #             adjustment = base_weight  # 使用基础权重
    #         else:
    #             performance = performance.cpu().numpy() if isinstance(performance, torch.Tensor) else performance
    #             if np.isnan(performance) or np.isinf(performance):
    #                 print(f"Invalid performance value for {task}: {performance}")
    #                 adjustment = base_weight  # 使用基础权重
    #             else:
    #                 # adjustment = base_weight * (1 - (performance * performance_factor))
    #                 # adjustment = base_weight * np.exp(-performance * performance_factor)
    #                 # adjustment = base_weight * (1 - np.tanh(performance * performance_factor))
    #                 adjustment = base_weight * (1 + np.tanh(performance * performance_factor) - 0.5)
    #                 print(f"Task: {task}, Base Weight: {base_weight}, Performance: {performance}, Adjustment: {adjustment}")
    
    #         if task == 'denoise':# 特定任务权重调整不应用最大最小限制
    #             new_weights[task + '_loss_weights'] = max(min(adjustment,20), 10)
    #         else:
    #             new_weights[task + '_loss_weights'] = max(min(adjustment, 3), 0.5)
    #     print("New weights: ", new_weights)
    #     self.hyper_params.update(new_weights)


    def training_step(self, batch, batch_idx):
        """
        mask_seqs.size(): torch.Size([B, seq, 6])
        masked_pos.size(): torch.Size([B, seq * mask_ratio])
        gt_imu_seq.size(): torch.Size([B, seq * mask_ratio, 6])
        normed_imu_seq.size(): torch.Size([B, seq, 6])

        """
        mask_seqs = batch['inputs']['mask_seqs'] # (B, seq, 6)

        masked_pos = batch['inputs']['masked_pos'] # (B, seq * mask_ratio)

        gt_masked_seq = batch['outputs']['gt_masked_seq']  # (B, seq * mask_ratio, 6)
        normed_input_imu = batch['outputs']['normed_input_imu']  # (B, Seq, 6)
        normed_future_imu = batch['outputs']['normed_future_imu']  # (B, Seq-future, 6)

        # MLM task
        # hat_imu_MLM = self.limu_bert_mlm.forward(mask_seqs, masked_pos)
        # print("mask_seqs.size(): ", mask_seqs.size())
        hat_imu_MLM = self.limu_bert_mlm.forward(mask_seqs)
        # 使用 torch.gather 从 hat_imu_MLM 中选取掩码位置的预测值
        gather_indices = masked_pos.unsqueeze(2).expand(-1, -1, hat_imu_MLM.size(2))
        selected_hat_imu_MLM = torch.gather(hat_imu_MLM, 1, gather_indices)  # 输出形状应为 [1024, 4, 6]

      
        MLM_loss = self.mse_loss(gt_masked_seq, selected_hat_imu_MLM) * float(
            self.hyper_params.mlm_loss_weights)

        # Denoise task
        hat_imu_denoise = self.limu_bert_mlm.forward(normed_input_imu)
        denoise_loss = self.mse_loss(normed_input_imu[:, -1, :], hat_imu_denoise[:, -1, :]) * float(
            self.hyper_params.denoise_loss_weights)

        # NSP task
        hat_imu_future = self.limu_bert_nsp.forward(normed_input_imu)

        hat_imu_future_denoised = self.limu_bert_nsp.forward(hat_imu_denoise)
        NSP_loss = (self.mse_loss(normed_future_imu, hat_imu_future)
                    + self.mse_loss(hat_imu_future_denoised, hat_imu_future)
                    ) * float(
            self.hyper_params.nsp_loss_weights)


        continuity_loss_future = self.mse_loss(hat_imu_future[:, :-1, :], hat_imu_future[:, 1:, :]) * float(
            self.hyper_params.continuity_loss_weight) 
        continuity_loss_future_denoised = self.mse_loss(hat_imu_future_denoised[:, :-1, :], hat_imu_future_denoised[:, 1:, :]) * float(
            self.hyper_params.continuity_loss_weight)
        continuity_loss = continuity_loss_future + continuity_loss_future_denoised

        # loss = MLM_loss + denoise_loss + NSP_loss
        loss = MLM_loss + denoise_loss + NSP_loss+continuity_loss

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_MLM_loss", MLM_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_denoise_loss", denoise_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_NSP_loss", NSP_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_continuity_loss", continuity_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        mask_seqs.size(): torch.Size([B, seq, 6])
        masked_pos.size(): torch.Size([B, seq * mask_ratio])
        gt_imu_seq.size(): torch.Size([B, seq * mask_ratio, 6])
        normed_imu_seq.size(): torch.Size([B, seq, 6])

        """
        mask_seqs = batch['inputs']['mask_seqs']  # (B, seq, 6)
        masked_pos = batch['inputs']['masked_pos']  # (B, seq * mask_ratio)
        gt_masked_seq = batch['outputs']['gt_masked_seq']  # (B, seq * mask_ratio, 6)
        normed_input_imu = batch['outputs']['normed_input_imu']  # (B, Seq, 6)
        normed_future_imu = batch['outputs']['normed_future_imu']  # (B, Seq-future, 6)



        ##########修改#######
        #打印mask_seqs的shape
        # print("mask_seqs.size(): ", mask_seqs.size())
        # #打印masked_pos的shape
        # print("masked_pos.size(): ", masked_pos.size())
        # #打印gt_masked_seq的shape
        # print("gt_masked_seq.size(): ", gt_masked_seq.size())
        # #打印normed_input_imu的shape
        # print("normed_input_imu.size(): ", normed_input_imu.size())
        # #打印normed_future_imu的shape
        # print("normed_future_imu.size(): ", normed_future_imu.size())


        # new_seq_length = 15
        # masked_pos = torch.clamp(masked_pos, max=new_seq_length - 1)
        # gt_masked_seq = torch.clamp(gt_masked_seq, max=new_seq_length - 1)
        #################################

        # MLM task
        hat_imu_MLM = self.limu_bert_mlm.forward(mask_seqs)
        # print("hat_imu_MLM.size(): ", hat_imu_MLM.size())
            # 使用 torch.gather 从 hat_imu_MLM 中选取掩码位置的预测值
        gather_indices = masked_pos.unsqueeze(2).expand(-1, -1, hat_imu_MLM.size(2))
        selected_hat_imu_MLM = torch.gather(hat_imu_MLM, 1, gather_indices)  # 输出形状应为 [1024, 4, 6]
        # print("After masked hat_imu_MLM.size(): ", selected_hat_imu_MLM.size())
        # hat_imu_MLM = self.limu_bert_mlm.forward(mask_seqs, masked_pos)
        # print("hat_imu_MLM.size(): ", hat_imu_MLM.size())
        # print("gt_masked_seq.size(): ", gt_masked_seq.size())
        MLM_loss = self.mse_loss(gt_masked_seq, selected_hat_imu_MLM) * float(
            self.hyper_params.mlm_loss_weights)

        # Denoise task
        hat_imu_denoise = self.limu_bert_mlm.forward(normed_input_imu)
        denoise_loss = self.mse_loss(normed_input_imu[:, -1, :], hat_imu_denoise[:, -1, :]) * float(
            self.hyper_params.denoise_loss_weights)

        # NSP task
        hat_imu_future = self.limu_bert_nsp.forward(normed_input_imu)
        hat_imu_future_denoised = self.limu_bert_nsp.forward(hat_imu_denoise)
        NSP_loss = (self.mse_loss(normed_future_imu, hat_imu_future)
                    + self.mse_loss(hat_imu_future_denoised, hat_imu_future)
                    ) * float(
            self.hyper_params.nsp_loss_weights)
        

        continuity_loss_future = self.mse_loss(hat_imu_future[:, :-1, :], hat_imu_future[:, 1:, :]) * float(
            self.hyper_params.continuity_loss_weight) 
        continuity_loss_future_denoised = self.mse_loss(hat_imu_future_denoised[:, :-1, :], hat_imu_future_denoised[:, 1:, :]) * float(
            self.hyper_params.continuity_loss_weight)
        continuity_loss = continuity_loss_future + continuity_loss_future_denoised
        # 时间连续性的约束。加入对时间序列数据平滑性的要求，使得模型在去噪过程中不仅关注单个数据点的准确性
        # ，而且保持相邻数据点之间的连续性
        loss = MLM_loss + denoise_loss + NSP_loss + continuity_loss
        # loss = MLM_loss + denoise_loss + NSP_loss 

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_MLM_loss", MLM_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_denoise_loss", denoise_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_NSP_loss", NSP_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_continuity_loss", continuity_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}

   

    # def validation_epoch_end(self, validation_step_outputs):

    #     # print("validation_step_outputs: ", validation_step_outputs)

    #     avg_MLM_loss = self.trainer.callback_metrics['val_MLM_loss']
    #     avg_denoise_loss = self.trainer.callback_metrics['val_denoise_loss']
    #     avg_NSP_loss = self.trainer.callback_metrics['val_NSP_loss']
    #     avg_continuity_loss = self.trainer.callback_metrics['val_continuity_loss']


    #     validation_performance = {
    #         'MLM': avg_MLM_loss,
    #         'denoise': avg_denoise_loss,
    #         'NSP': avg_NSP_loss,
    #         'continuity': avg_continuity_loss
    #     }
    #     print("Validation performance: ", validation_performance)
    #     self.adjust_task_weights(validation_performance)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.starting_learning_rate)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                              T_0=int(self.hyper_params.T_0),
                                                                              T_mult=int(self.hyper_params.T_mult),
                                                                              eta_min=float(self.hyper_params.eta_min)),
            "interval": "epoch",
            "frequency": 1,
            'name': 'learning_rate'
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}



  