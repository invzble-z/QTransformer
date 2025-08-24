from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')


class WeightedMSELoss(nn.Module):
    """Weighted MSE Loss cho multi-market forecasting theo roadmap Phase 1"""
    def __init__(self, market_weights=None):
        super(WeightedMSELoss, self).__init__()
        # Trọng số theo roadmap: [Europe=0.35, LATAM=0.30, USCA=0.35]
        if market_weights is None:
            self.market_weights = torch.tensor([0.35, 0.30, 0.35])
        else:
            self.market_weights = torch.tensor(market_weights)
    
    def forward(self, predictions, targets):
        """
        predictions: [batch_size, pred_len, c_out] = [batch, 7, 3]
        targets: [batch_size, pred_len, c_out] = [batch, 7, 3]
        """
        # Ensure weights are on same device
        device = predictions.device
        weights = self.market_weights.to(device)
        
        # Tính MSE cho từng market
        mse_per_market = torch.mean((predictions - targets) ** 2, dim=(0, 1))  # [3]
        
        # Apply market weights
        weighted_loss = torch.sum(weights * mse_per_market)
        
        return weighted_loss


class Exp_Long_Term_Forecast_Embedding(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Embedding, self).__init__(args)
        
        # Load target scaler for denormalization if available
        self.target_scaler = None
        scaler_path = './scalers/target_scaler.pkl'
        if Path(scaler_path).exists():
            try:
                self.target_scaler = joblib.load(scaler_path)
                print(f"🔧 Loaded target scaler from {scaler_path}")
            except Exception as e:
                print(f"⚠️ Could not load target scaler: {e}")
                self.target_scaler = None

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # Sử dụng WeightedMSELoss cho multi-market theo roadmap Phase 1
        if hasattr(self.args, 'c_out') and self.args.c_out == 3:
            # Multi-market với trọng số: Europe=0.35, LATAM=0.30, USCA=0.35
            criterion = WeightedMSELoss(market_weights=[0.35, 0.30, 0.35])
            print("🎯 Sử dụng WeightedMSELoss cho multi-market: [Europe=0.35, LATAM=0.30, USCA=0.35]")
        else:
            # Fallback to standard MSE
            criterion = nn.MSELoss()
            print("📊 Sử dụng standard MSELoss")
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                # Handle both embedding and non-embedding data
                if len(batch_data) == 5:  # Embedding data
                    batch_x, batch_y, batch_x_mark, batch_y_mark, categorical_features = batch_data
                    categorical_features = {k: v.to(self.device) for k, v in categorical_features.items()}
                else:  # Standard data
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    categorical_features = None
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Model forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if categorical_features is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, categorical_features)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if categorical_features is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, categorical_features)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_data in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # Handle both embedding and non-embedding data
                if len(batch_data) == 5:  # Embedding data
                    batch_x, batch_y, batch_x_mark, batch_y_mark, categorical_features = batch_data
                    categorical_features = {k: v.to(self.device) for k, v in categorical_features.items()}
                else:  # Standard data
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    categorical_features = None
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Model forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if categorical_features is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, categorical_features)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if categorical_features is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, categorical_features)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                # Handle both embedding and non-embedding data
                if len(batch_data) == 5:  # Embedding data
                    batch_x, batch_y, batch_x_mark, batch_y_mark, categorical_features = batch_data
                    categorical_features = {k: v.to(self.device) for k, v in categorical_features.items()}
                else:  # Standard data
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    categorical_features = None
                    
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Model forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if categorical_features is not None:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, categorical_features)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if categorical_features is not None:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, categorical_features)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        
        # Ensure correct dimensions
        if len(preds.shape) == 2:
            preds = preds[:, :, np.newaxis]
        if len(trues.shape) == 2:
            trues = trues[:, :, np.newaxis]
            
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('final test shape:', preds.shape, trues.shape)

        # Result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Calculate both original and corrected metrics
        mae_orig, mse_orig, rmse_orig, mape_orig, mspe_orig = metric(preds, trues)
        mae_corr, mse_corr, rmse_corr, mape_corr, mspe_corr = self._calculate_corrected_metrics(preds, trues)
        
        print('Original (normalized) - mse:{}, mae:{}'.format(mse_orig, mae_orig))
        print('Corrected (denormalized) - mse:{}, mae:{}, mape:{:.4f}%, mspe:{:.4f}%'.format(
            mse_corr, mae_corr, mape_corr, mspe_corr))
        
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        # Save corrected metrics to file
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(
            mse_corr, mae_corr, rmse_corr, mape_corr, mspe_corr))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae_corr, mse_corr, rmse_corr, mape_corr, mspe_corr]))
        np.save(folder_path + 'metrics_original.npy', np.array([mae_orig, mse_orig, rmse_orig, mape_orig, mspe_orig]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def _calculate_corrected_metrics(self, preds, trues):
        """
        Calculate metrics with proper denormalization if target_scaler is available
        """
        if self.target_scaler is None:
            # Use original metrics if no scaler
            return metric(preds, trues)
        
        try:
            # Reshape for denormalization 
            original_pred_shape = preds.shape
            original_true_shape = trues.shape
            
            # Flatten to (samples, 1) for scaler
            preds_flat = preds.reshape(-1, 1)
            trues_flat = trues.reshape(-1, 1)
            
            # Denormalize
            preds_denorm = self.target_scaler.inverse_transform(preds_flat)
            trues_denorm = self.target_scaler.inverse_transform(trues_flat)
            
            # Reshape back to original
            preds_denorm = preds_denorm.reshape(original_pred_shape)
            trues_denorm = trues_denorm.reshape(original_true_shape)
            
            # Calculate corrected metrics
            mae_corrected = np.mean(np.abs(preds_denorm - trues_denorm))
            mse_corrected = np.mean((preds_denorm - trues_denorm) ** 2)
            rmse_corrected = np.sqrt(mse_corrected)
            
            # Avoid division by zero for MAPE
            mask = np.abs(trues_denorm) > 1e-8
            if np.sum(mask) > 0:
                mape_corrected = np.mean(np.abs((trues_denorm[mask] - preds_denorm[mask]) / trues_denorm[mask])) * 100
                mspe_corrected = np.mean(((trues_denorm[mask] - preds_denorm[mask]) / trues_denorm[mask]) ** 2) * 100
            else:
                mape_corrected = float('inf')
                mspe_corrected = float('inf')
            
            print(f"📊 Denormalized metrics - MSE: {mse_corrected:.4f}, MAE: {mae_corrected:.4f}, MAPE: {mape_corrected:.4f}%")
            
            return mae_corrected, mse_corrected, rmse_corrected, mape_corrected, mspe_corrected
            
        except Exception as e:
            print(f"⚠️ Error in denormalization, using original metrics: {e}")
            return metric(preds, trues)
