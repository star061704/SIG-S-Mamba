import random
import torch.nn.functional as F
from keras_sig import signature as ks_signature
import os
os.environ['KERAS_BACKEND'] = 'torch'
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
class SigLossModule(nn.Module):
    def __init__(self, depth=2, use_leadlag=False, max_nodes=64):
        super().__init__()
        self.depth = depth
        self.use_leadlag = use_leadlag
        self.max_nodes = max_nodes

    def build_path(self, x_1d: torch.Tensor) -> torch.Tensor:
        # x_1d: [L] 或 [L,1]
        x_1d = x_1d.view(-1, 1)  # [L,1]
        L = x_1d.size(0)
        t = torch.linspace(0.0, 1.0, L, device=x_1d.device, dtype=x_1d.dtype).unsqueeze(-1)
        path = torch.cat([t, x_1d], dim=-1)  # [L,2]

        if self.use_leadlag:
            # 简单 lead–lag 展开
            lead = torch.repeat_interleave(path, 2, dim=0)[1:]
            lag  = torch.repeat_interleave(path, 2, dim=0)[:-1]
            path = torch.cat([lead, lag], dim=-1)  # [2L-1,4]

        return path

    def sig_1series(self, x_1d: torch.Tensor) -> torch.Tensor:
        # x_1d: [L]
        path = self.build_path(x_1d)             # [L,2] 或 [2L-1,4]
        s = ks_signature(path, depth=self.depth) # [D]
        return s

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, L, N]
        返回: [B, N_sel, D]（只对一部分节点算）
        """
        B, L, N = seq.shape

        # 随机挑一些节点来算 signature（比如 8 或 16 个）
        if N > self.max_nodes:
            idx = torch.randperm(N, device=seq.device)[: self.max_nodes]
        else:
            idx = torch.arange(N, device=seq.device)

        seq = seq[:, :, idx]   # [B,L,N_sel]
        N_sel = seq.size(-1)

        sigs = []
        for b in range(B):
            row = []
            for j in range(N_sel):
                s = self.sig_1series(seq[b, :, j])  # [D]
                row.append(s)
            sigs.append(torch.stack(row, dim=0))     # [N_sel,D]
        sigs = torch.stack(sigs, dim=0)              # [B,N_sel,D]
        return sigs, idx


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        self.use_sig_loss = getattr(args, 'use_sig_loss', False)
        if self.use_sig_loss:
            self.sig_module = SigLossModule(
                depth=getattr(args, 'sig_loss_depth', 2),
                use_leadlag=getattr(args, 'sig_loss_leadlag', False),
            ).to(self.device)
            self.lambda_sig = getattr(args, 'lambda_sig', 1e-4)

        # Two-stage training state for dual_axis_sig:
        #   None  -> stage 1 (gate is free, both branches train)
        #   0./1. -> stage 2 (gate frozen at this value, single branch only)
        self.forced_alpha = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

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
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # use forced_alpha during stage-2 of dual_axis_sig two-stage training
                _use_force = (self.args.model == 'dual_axis_sig'
                              and self.forced_alpha is not None)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)[0]
                                       if _use_force else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])
                        else:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)
                                       if _use_force else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                else:
                    if self.args.output_attention:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)[0]
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])
                    else:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
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
        # min_epochs=1: the first epoch is warm-up -- it does not set the best
        # checkpoint and does not advance the early-stopping counter. Prevents
        # an over-fit first epoch from being locked in as the saved model.
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       min_epochs=1)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ---- two-stage training: stage 1 alternates single-branch training
        # (first half temporal, second half feature) so each branch sees a
        # clean full-gradient probe; at the boundary we measure each branch's
        # validation MSE and pick the lower one; stage 2 re-inits the chosen
        # branch and trains fresh.
        _two_stage_epochs = int(getattr(self.args, 'two_stage_epochs', 0) or 0)
        _two_stage_active = (self.args.model == 'dual_axis_sig'
                             and _two_stage_epochs >= 2)
        if (self.args.model == 'dual_axis_sig'
                and 0 < _two_stage_epochs < 2):
            print(f"[warning] --two_stage_epochs={_two_stage_epochs} < 2; "
                  f"alternate-probe needs >=2 (one epoch per branch). Disabling two-stage.")
        _stage1_temp_epochs = _two_stage_epochs // 2 if _two_stage_active else 0
        _stage2_start_epoch = None
        _two_stage_committed = False

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            epoch_alphas = []

            # ---- decide self.forced_alpha for this epoch ----
            if _two_stage_active and not _two_stage_committed:
                if epoch < _stage1_temp_epochs:
                    # stage 1, first half: train temporal branch only
                    self.forced_alpha = 1.0
                elif epoch < _two_stage_epochs:
                    # stage 1, second half: train feature branch only
                    self.forced_alpha = 0.0
                else:
                    # ---- transition: pick winner by val MSE, re-init, restart ----
                    self.forced_alpha = 1.0
                    _val_mse_temp = self.vali(vali_data, vali_loader, criterion)
                    self.forced_alpha = 0.0
                    _val_mse_feat = self.vali(vali_data, vali_loader, criterion)
                    print(f"\n--- Stage 1 complete; comparing branches on validation set ---")
                    print(f"    branch_temp val MSE = {_val_mse_temp:.6f}")
                    print(f"    branch_feat val MSE = {_val_mse_feat:.6f}")

                    self.forced_alpha = (1.0 if _val_mse_temp < _val_mse_feat
                                         else 0.0)
                    _picked = ('temporal' if self.forced_alpha == 1.0
                               else 'feature')

                    # ---- reset RNG so stage 2 is byte-equivalent to a fresh
                    # single-axis training (same random init weights, same
                    # dataloader shuffle order, same dropout sequence).
                    import random as _random
                    _seed = getattr(self.args, 'seed', None) or 2023
                    _random.seed(_seed)
                    np.random.seed(_seed)
                    torch.manual_seed(_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(_seed)

                    # Re-init chosen branch from scratch -- stage 1 was probe.
                    # Built right after the seed reset, so the branch's initial
                    # weights match what a standalone single-axis Exp would
                    # produce on the same seed.
                    from model import time_sig_fast, feature_sig_fast
                    _model_inner = (self.model.module
                                    if hasattr(self.model, 'module') else self.model)
                    if self.forced_alpha == 1.0:
                        _model_inner.branch_temp = (
                            time_sig_fast.Model(self.args).float().to(self.device))
                    else:
                        _model_inner.branch_feat = (
                            feature_sig_fast.Model(self.args).float().to(self.device))

                    # Re-create dataloaders so their samplers consume the post-
                    # reset RNG (matching a standalone single-axis run).
                    train_data, train_loader = self._get_data(flag='train')
                    vali_data, vali_loader = self._get_data(flag='val')
                    test_data, test_loader = self._get_data(flag='test')
                    train_steps = len(train_loader)

                    # Rebuild optimizer & reset early stopping for fresh stage 2.
                    model_optim = self._select_optimizer()
                    early_stopping = EarlyStopping(patience=self.args.patience,
                                                   verbose=True, min_epochs=1)
                    _stage2_start_epoch = epoch
                    _two_stage_committed = True

                    print(f"=== STAGE 2 (epoch {epoch + 1}): pick={_picked} "
                          f"(force_alpha={self.forced_alpha:.1f}); RNG reset, "
                          f"branch re-initialised, dataloaders/optimizer/early-"
                          f"stopping rebuilt ===\n")

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # use forced_alpha if set (dual_axis_sig two-stage)
                _use_force_train = (self.args.model == 'dual_axis_sig'
                                    and self.forced_alpha is not None)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)[0]
                                       if _use_force_train else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])
                        else:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)
                                       if _use_force_train else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        if self.use_sig_loss:
                            print("before sig loss")
                            with torch.no_grad():
                                sig_true, idx = self.sig_module(batch_y)   # [B,N,D]
                            sig_pred, _ = self.sig_module(outputs[:, :, idx])       # [B,N,D]
                            loss_sig = F.mse_loss(sig_pred, sig_true)
                            loss = loss + self.lambda_sig * loss_sig

                            print(f"loss_mse={loss.item():.4f}, "
                                f"loss_sig={loss_sig.item():.4f}, "
                                f"lambda*loss_sig={self.lambda_sig * loss_sig.item():.6f}")
                        train_loss.append(loss.item())
                else:
                    # use forced_alpha during stage-2 of dual_axis_sig two-stage training
                    _use_force = (self.args.model == 'dual_axis_sig'
                                  and self.forced_alpha is not None)
                    if self.args.output_attention:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)[0]
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])
                    else:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)


                    if self.use_sig_loss:
                        # print("before sig loss")
                        with torch.no_grad():
                            sig_true, idx = self.sig_module(batch_y)
                        sig_pred, _ = self.sig_module(outputs[:, :, idx])
                        loss_sig = F.mse_loss(sig_pred, sig_true)
                        loss = loss + self.lambda_sig * loss_sig
                        # print(f"loss_mse={loss.item():.4f}, "
                        #         f"loss_sig={loss_sig.item():.4f}, "
                        #         f"lambda*loss_sig={self.lambda_sig * loss_sig.item():.6f}")

                    # dual_axis_sig auxiliary endpoint loss: force alpha=1 and alpha=0
                    # forwards so each pathway is trained to be standalone-useful.
                    # Skipped during two-stage stage 2 -- alpha is already pinned.
                    _aux_lambda = getattr(self.args, 'dual_aux_lambda', 0.0)
                    if (self.args.model == 'dual_axis_sig' and _aux_lambda > 0
                            and self.forced_alpha is None):
                        _model_inner = self.model.module if hasattr(self.model, 'module') else self.model
                        out_t1 = _model_inner(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=1.0)
                        out_t1 = out_t1[:, -self.args.pred_len:, f_dim:]
                        out_t0 = _model_inner(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=0.0)
                        out_t0 = out_t0[:, -self.args.pred_len:, f_dim:]
                        loss_t1 = criterion(out_t1, batch_y)
                        loss_t0 = criterion(out_t0, batch_y)
                        loss = loss + _aux_lambda * (loss_t1 + loss_t0)

                    train_loss.append(loss.item())

                # collect dual_axis_sig gate alpha (no-op for other models /
                # for forced-alpha epochs; method-3 two-stage forces alpha
                # during stage 1, so this only fires in pure free-routing runs)
                _m = self.model.module if hasattr(self.model, 'module') else self.model
                _a = getattr(_m, 'last_alpha', None)
                if _a is not None and self.forced_alpha is None:
                    _a_val = _a.item() if torch.is_tensor(_a) else float(_a)
                    epoch_alphas.append(_a_val)

                if (i + 1) % 100 == 0:
                    _alpha_str = (f' | alpha={epoch_alphas[-1]:.3f}'
                                  if epoch_alphas else '')
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}{3}".format(
                        i + 1, epoch + 1, loss.item(), _alpha_str))
                    speed = (time.time() - time_now) / iter_count
                    # print(speed)
                    # allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
                    # cached_memory = torch.cuda.memory_cached() / (1024 * 1024 * 1024)
                    # total = allocated_memory + cached_memory
                    # print('allocated_memory:', allocated_memory)
                    # print('cached_memory:', cached_memory)
                    # print('total:', total)
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
            if epoch_alphas:
                _ea = np.array(epoch_alphas)
                print("\tgate alpha: mean={:.4f}  std={:.4f}  min={:.4f}  max={:.4f}".format(
                    _ea.mean(), _ea.std(), _ea.min(), _ea.max()))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # In two-stage stage 2, restart the lr schedule from epoch 1.
            if _stage2_start_epoch is not None:
                adjust_learning_rate(model_optim,
                                     (epoch - _stage2_start_epoch) + 1,
                                     self.args)
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # time_points = random.sample(range(batch_x.size()[1]), 5)
                # 假设您有一个包含每个变量标准差的张量stds，形状为(321,)
                # 定义扰动强度
                # epsilon = 1
                # # 创建一个与原始张量形状相同的张量来存储扰动
                # perturbed_tensor = torch.zeros_like(batch_x)
                # # 对每个选定的时间点添加扰动
                # for time_point in time_points:
                #     # 生成与tensor在该时间点形状相同的随机噪声
                #     noise = torch.randn(1, 321) * epsilon
                #     perturbed_tensor[:, time_point, :] += noise.float().to(self.device)
                # batch_x += perturbed_tensor
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # use forced_alpha if set (dual_axis_sig two-stage stage-2 commit)
                _use_force = (self.args.model == 'dual_axis_sig'
                              and self.forced_alpha is not None)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)[0]
                                       if _use_force else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])
                        else:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)
                                       if _use_force else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                else:
                    if self.args.output_attention:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)[0]
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])

                    else:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            input = batch_x.detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # use forced_alpha if set (dual_axis_sig two-stage stage-2 commit)
                _use_force = (self.args.model == 'dual_axis_sig'
                              and self.forced_alpha is not None)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)[0]
                                       if _use_force else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])
                        else:
                            outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  force_alpha=self.forced_alpha)
                                       if _use_force else
                                       self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                else:
                    if self.args.output_attention:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)[0]
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0])
                    else:
                        outputs = (self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                              force_alpha=self.forced_alpha)
                                   if _use_force else
                                   self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark))
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return