# complete_multi_modal_pinn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示和数学符号
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']  # 修改字体设置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # 修改数学符号字体

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 多模态数据集类 ====================

class MultiModalMaterialDataset(Dataset):
    def __init__(self, displacement_csv, force_csv, normalize=True):
        """
        多模态材料数据集 - 同时处理位移场和力-位移数据
        """
        # 加载位移场数据
        self.disp_data = pd.read_csv(displacement_csv)
        
        # 提取位移场数据
        self.coords = self.disp_data[['x', 'y']].values.astype(np.float32)
        self.displacements = self.disp_data[['u', 'v']].values.astype(np.float32)
        self.point_types = self.disp_data['type'].values.astype(np.int64)
        self.group_ids = self.disp_data['group_id'].values.astype(np.int64)
        self.material_params = self.disp_data[['param_E', 'param_v', 'param_sigma_y', 'param_H']].values.astype(np.float32)
        
        # 加载力-位移数据并提取特征
        self.force_data = pd.read_csv(force_csv)
        self.force_features = self._extract_force_features()
        
        self.normalize = normalize
        if normalize:
            self._normalize_data()
            
        print(f"多模态数据集加载完成:")
        print(f"  位移场样本: {len(self.disp_data)} 个点")
        print(f"  力-位移曲线: {len(self.force_features)} 组材料")
        print(f"  材料参数范围 - E: [{np.min(self.material_params[:, 0]):.0f}, {np.max(self.material_params[:, 0]):.0f}] MPa")
        print(f"  材料参数范围 - H: [{np.min(self.material_params[:, 3]):.0f}, {np.max(self.material_params[:, 3]):.0f}] MPa")
    
    def _extract_force_features(self):
        """从力-位移数据中提取关键特征"""
        features_dict = {}
        unique_groups = self.force_data['group_id'].unique()
        
        for group_id in unique_groups:
            group_data = self.force_data[self.force_data['group_id'] == group_id].sort_values('load_step')
            
            displacements = group_data['displacement_y'].values
            forces = group_data['force_y'].values
            
            if len(displacements) < 5:
                continue
                
            # 1. 弹性刚度 (初始斜率)
            initial_points = min(5, len(displacements))
            elastic_stiffness = np.polyfit(displacements[:initial_points], forces[:initial_points], 1)[0]
            
            # 2. 最大载荷
            max_force = np.max(forces)
            max_displacement = displacements[np.argmax(forces)]
            
            # 3. 屈服点识别
            yield_point_idx = self._find_yield_point(displacements, forces)
            yield_force = forces[yield_point_idx] if yield_point_idx is not None else forces[min(3, len(forces)-1)]
            yield_displacement = displacements[yield_point_idx] if yield_point_idx is not None else displacements[min(3, len(displacements)-1)]
            
            # 4. 硬化行为特征 - 对H参数最关键！
            hardening_features = self._analyze_hardening_behavior(displacements, forces, yield_point_idx)
            
            # 5. 能量特征
            total_energy = np.trapz(forces, displacements)
            if yield_point_idx is not None and yield_point_idx < len(forces)-1:
                plastic_energy = np.trapz(forces[yield_point_idx:], displacements[yield_point_idx:])
            else:
                plastic_energy = total_energy * 0.7
                
            features_dict[group_id] = np.array([
                elastic_stiffness, yield_force, yield_displacement,
                max_force, max_displacement, hardening_features['slope'],
                hardening_features['ratio'], plastic_energy, total_energy
            ], dtype=np.float32)
            
        return features_dict
    
    def _find_yield_point(self, displacements, forces):
        """基于斜率变化识别屈服点"""
        if len(displacements) < 10:
            return None
            
        slopes = []
        for i in range(1, len(displacements)-1):
            slope = (forces[i+1] - forces[i-1]) / (displacements[i+1] - displacements[i-1] + 1e-8)
            slopes.append(slope)
        
        initial_slope = np.mean(slopes[:3])
        for i, slope in enumerate(slopes[5:], 5):
            if slope < initial_slope * 0.7:
                return i
        return None
    
    def _analyze_hardening_behavior(self, displacements, forces, yield_point_idx):
        """分析硬化行为 - 专门针对H参数优化"""
        if yield_point_idx is None or yield_point_idx >= len(displacements) - 3:
            start_idx = max(0, int(len(displacements) * 0.3))
        else:
            start_idx = yield_point_idx
            
        plastic_displacements = displacements[start_idx:]
        plastic_forces = forces[start_idx:]
        
        if len(plastic_displacements) < 3:
            return {'slope': 0.0, 'ratio': 0.0}
            
        hardening_slope = np.polyfit(plastic_displacements, plastic_forces, 1)[0]
        
        elastic_slope = np.polyfit(displacements[:min(5, len(displacements))], 
                                  forces[:min(5, len(forces))], 1)[0]
        hardening_ratio = hardening_slope / (elastic_slope + 1e-8)
        
        return {'slope': hardening_slope, 'ratio': hardening_ratio}
    
    def _normalize_data(self):
        """数据归一化"""
        # 位移场数据归一化
        self.coord_mean = np.mean(self.coords, axis=0)
        self.coord_std = np.std(self.coords, axis=0)
        self.coords = (self.coords - self.coord_mean) / (self.coord_std + 1e-8)
        
        self.disp_mean = np.mean(self.displacements, axis=0)
        self.disp_std = np.std(self.displacements, axis=0)
        self.displacements = (self.displacements - self.disp_mean) / (self.disp_std + 1e-8)
        
        # 材料参数归一化
        self.param_mean = np.mean(self.material_params, axis=0)
        self.param_std = np.std(self.material_params, axis=0)
        self.material_params = (self.material_params - self.param_mean) / (self.param_std + 1e-8)
        
        # 力-位移特征归一化
        all_features = np.array(list(self.force_features.values()))
        self.force_feature_mean = np.mean(all_features, axis=0)
        self.force_feature_std = np.std(all_features, axis=0)
        
        for group_id in self.force_features:
            self.force_features[group_id] = (self.force_features[group_id] - self.force_feature_mean) / (self.force_feature_std + 1e-8)
        
        print("多模态数据归一化完成")
    
    # ========== 添加缺失的反归一化方法 ==========
    def denormalize_coords(self, coords):
        """反归一化坐标"""
        if self.normalize:
            return coords * self.coord_std + self.coord_mean
        return coords
    
    def denormalize_displacements(self, displacements):
        """反归一化位移"""
        if self.normalize:
            return displacements * self.disp_std + self.disp_mean
        return displacements
    
    def denormalize_params(self, params):
        """反归一化材料参数"""
        if self.normalize:
            return params * self.param_std + self.param_mean
        return params
    
    def denormalize_force_features(self, features):
        """反归一化力-位移特征"""
        if self.normalize:
            return features * self.force_feature_std + self.force_feature_mean
        return features
    # ========== 反归一化方法结束 ==========
    
    def __len__(self):
        return len(self.disp_data)
    
    def __getitem__(self, idx):
        group_id = self.group_ids[idx]
        force_feature = self.force_features.get(group_id, np.zeros(9, dtype=np.float32))
        
        return {
            'coords': torch.tensor(self.coords[idx]),
            'displacements': torch.tensor(self.displacements[idx]),
            'point_type': torch.tensor(self.point_types[idx]),
            'group_id': torch.tensor(group_id),
            'material_params': torch.tensor(self.material_params[idx]),
            'force_features': torch.tensor(force_feature)
        }

# ==================== 2. 双分支多模态PINN模型 ====================

class MultiModalPINN(nn.Module):
    def __init__(self, disp_input_dim=4, force_input_dim=9, output_dim=4, 
                 disp_hidden_dim=128, force_hidden_dim=64, fusion_dim=256):
        """
        多模态PINN模型 - 双分支架构专门优化H参数学习
        """
        super(MultiModalPINN, self).__init__()
        
        # 分支1: 位移场编码器 (处理坐标和位移)
        self.disp_encoder = nn.Sequential(
            nn.Linear(disp_input_dim, disp_hidden_dim),
            nn.Tanh(),
            nn.Linear(disp_hidden_dim, disp_hidden_dim),
            nn.SiLU(),
            nn.Linear(disp_hidden_dim, disp_hidden_dim // 2),
            nn.Tanh()
        )
        
        # 分支2: 力-位移特征编码器 (专门学习H参数)
        self.force_encoder = nn.Sequential(
            nn.Linear(force_input_dim, force_hidden_dim),
            nn.Tanh(),
            nn.Linear(force_hidden_dim, force_hidden_dim),
            nn.SiLU(),
            nn.Linear(force_hidden_dim, force_hidden_dim // 2),
            nn.Tanh()
        )
        
        # 特征融合和参数预测
        self.fusion_net = nn.Sequential(
            nn.Linear((disp_hidden_dim // 2) + (force_hidden_dim // 2), fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.SiLU(),
            nn.Linear(fusion_dim // 2, output_dim)
        )
        
        self._initialize_weights()
        
        print(f"多模态PINN初始化完成:")
        print(f"  位移场分支: {disp_input_dim} → {disp_hidden_dim} → {disp_hidden_dim//2}")
        print(f"  力-位移分支: {force_input_dim} → {force_hidden_dim} → {force_hidden_dim//2}")
        print(f"  融合网络: {((disp_hidden_dim // 2) + (force_hidden_dim // 2))} → {fusion_dim} → {output_dim}")
    
    def _initialize_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, disp_inputs, force_inputs):
        disp_features = self.disp_encoder(disp_inputs)
        force_features = self.force_encoder(force_inputs)
        fused_features = torch.cat([disp_features, force_features], dim=1)
        return self.fusion_net(fused_features)

# ==================== 3. 多模态损失函数 ====================

class MultiModalLoss:
    def __init__(self, lambda_disp=1.0, lambda_force=0.8, lambda_physics=0.1, lambda_h=2.0):
        """
        多模态损失函数 - 平衡位移场和力-位移数据的监督
        """
        self.lambda_disp = lambda_disp
        self.lambda_force = lambda_force
        self.lambda_physics = lambda_physics
        self.lambda_h = lambda_h  # H参数的专项权重
        
        print(f"多模态损失权重 - 位移场: {lambda_disp}, 力-位移: {lambda_force}, 物理: {lambda_physics}, H专项: {lambda_h}")
    
    def compute_total_loss(self, pred_params, true_params, force_features, coords=None, displacements=None):
        """
        计算多模态总损失
        """
        # 位移场数据损失
        disp_loss = self._compute_disp_loss(pred_params, true_params)
        
        # 力-位移特征损失 (专门强化H参数学习)
        force_loss = self._compute_force_based_loss(pred_params, true_params, force_features)
        
        # 物理约束损失
        physics_loss = self._compute_physics_constraint(pred_params)
        
        # H参数专项损失
        h_special_loss = self._compute_h_special_loss(pred_params, true_params)
        
        total_loss = (self.lambda_disp * disp_loss + 
                     self.lambda_force * force_loss + 
                     self.lambda_physics * physics_loss +
                     self.lambda_h * h_special_loss)
        
        return {
            'total_loss': total_loss,
            'disp_loss': disp_loss,
            'force_loss': force_loss,
            'physics_loss': physics_loss,
            'h_special_loss': h_special_loss
        }
    
    def _compute_disp_loss(self, pred_params, true_params):
        """位移场数据损失"""
        return F.mse_loss(pred_params, true_params)
    
    def _compute_force_based_loss(self, pred_params, true_params, force_features):
        """基于力-位移特征的损失 - 专门针对H参数优化"""
        # 提取预测的H参数
        pred_H = pred_params[:, 3]
        true_H = true_params[:, 3]
        
        # 从力特征中提取硬化相关特征 (索引5:硬化斜率, 索引6:硬化比率, 索引7:塑性功)
        hardening_slope = force_features[:, 5]
        hardening_ratio = force_features[:, 6]
        plastic_work = force_features[:, 7]
        
        # 1. 直接H参数监督
        h_direct_loss = F.mse_loss(pred_H, true_H)
        
        # 2. 硬化特征一致性损失 (预测的H应该与硬化特征相关)
        # 这里我们鼓励pred_H与硬化斜率正相关
        h_feature_corr = -torch.mean(pred_H * hardening_slope)  # 负号因为我们要最大化相关性
        
        # 3. H参数范围约束 (基于物理先验)
        h_range_loss = torch.mean(F.relu(-pred_H) ** 2)  # H应该为正
        
        return h_direct_loss + 0.5 * h_feature_corr + 0.3 * h_range_loss
    
    def _compute_physics_constraint(self, pred_params):
        """物理约束损失"""
        E = pred_params[:, 0]
        v = pred_params[:, 1]
        sigma_y = pred_params[:, 2]
        H = pred_params[:, 3]
        
        # 正定性约束
        positive_loss = torch.mean(F.relu(-E) ** 2 + F.relu(-sigma_y) ** 2 + F.relu(-H) ** 2)
        
        # 泊松比约束
        v_loss = torch.mean(F.relu(0.15 - v) ** 2 + F.relu(v - 0.45) ** 2)
        
        # E-H关系约束 (H通常远小于E)
        e_h_ratio_loss = torch.mean(F.relu(H - 0.3 * E) ** 2)
        
        return positive_loss + v_loss + e_h_ratio_loss
    
    def _compute_h_special_loss(self, pred_params, true_params):
        """H参数专项损失"""
        pred_H = pred_params[:, 3]
        true_H = true_params[:, 3]
        
        # 组合多种损失函数来强化H参数学习
        mse_loss = F.mse_loss(pred_H, true_H)
        
        # 相对误差惩罚
        rel_error = torch.abs(pred_H - true_H) / (torch.abs(true_H) + 1e-6)
        rel_loss = torch.mean(rel_error)
        
        # 对数空间损失 (处理大范围变化)
        log_pred = torch.log(torch.abs(pred_H) + 1e-6)
        log_true = torch.log(torch.abs(true_H) + 1e-6)
        log_loss = F.mse_loss(log_pred, log_true)
        
        return mse_loss + 0.3 * rel_loss + 0.2 * log_loss

# ==================== 4. 渐进式训练器 ====================

class ProgressiveTrainer:
    def __init__(self, model, loss_function, train_loader, val_loader, test_loader, dataset):
        self.model = model
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.dataset = dataset
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=50,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.disp_losses = []
        self.force_losses = []
        self.physics_losses = []
        self.h_special_losses = []
        self.learning_rates = []
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience = 25
        self.patience_counter = 0
        
        # 渐进训练阶段
        self.training_phase = 1  # 1:基础训练, 2:H参数重点训练, 3:联合微调
        
        print("渐进式训练器初始化完成")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        disp_loss_total = 0
        force_loss_total = 0
        physics_loss_total = 0
        h_special_loss_total = 0
        
        # 动态调整训练策略
        if epoch < 50:
            self.training_phase = 1  # 基础训练阶段
        elif epoch < 100:
            self.training_phase = 2  # H参数重点训练
        else:
            self.training_phase = 3  # 联合微调
        
        pbar = tqdm(self.train_loader, desc=f'Phase {self.training_phase} - Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            coords = batch['coords']
            displacements = batch['displacements']
            point_types = batch['point_type']
            true_params = batch['material_params']
            force_features = batch['force_features']
            
            # 准备输入
            disp_inputs = torch.cat([coords, displacements], dim=1)
            
            # 前向传播
            pred_params = self.model(disp_inputs, force_features)
            
            self.optimizer.zero_grad()
            
            # 计算损失
            losses = self.loss_function.compute_total_loss(
                pred_params, true_params, force_features, coords, displacements
            )
            
            # 根据训练阶段调整损失权重
            if self.training_phase == 1:
                adjusted_loss = losses['disp_loss'] + 0.5 * losses['physics_loss']
            elif self.training_phase == 2:
                adjusted_loss = losses['force_loss'] + losses['h_special_loss'] + 0.3 * losses['disp_loss']
            else:
                adjusted_loss = losses['total_loss']
            
            adjusted_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录损失
            total_loss += losses['total_loss'].item()
            disp_loss_total += losses['disp_loss'].item()
            force_loss_total += losses['force_loss'].item()
            physics_loss_total += losses['physics_loss'].item()
            h_special_loss_total += losses['h_special_loss'].item()
            
            pbar.set_postfix({
                'Total': f"{losses['total_loss'].item():.6f}",
                'Disp': f"{losses['disp_loss'].item():.6f}",
                'Force': f"{losses['force_loss'].item():.6f}",
                'H_Special': f"{losses['h_special_loss'].item():.6f}"
            })
        
        # 更新学习率
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        self.learning_rates.append(current_lr)
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_disp_loss = disp_loss_total / num_batches
        avg_force_loss = force_loss_total / num_batches
        avg_physics_loss = physics_loss_total / num_batches
        avg_h_special_loss = h_special_loss_total / num_batches
        
        # 记录损失
        self.train_losses.append(avg_loss)
        self.disp_losses.append(avg_disp_loss)
        self.force_losses.append(avg_force_loss)
        self.physics_losses.append(avg_physics_loss)
        self.h_special_losses.append(avg_h_special_loss)
        
        return avg_loss, avg_disp_loss, avg_force_loss, avg_physics_loss, avg_h_special_loss
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                coords = batch['coords']
                displacements = batch['displacements']
                true_params = batch['material_params']
                force_features = batch['force_features']
                
                disp_inputs = torch.cat([coords, displacements], dim=1)
                pred_params = self.model(disp_inputs, force_features)
                
                # 验证时只计算数据损失
                loss = F.mse_loss(pred_params, true_params)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # 早停机制
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_model_state = self.model.state_dict().copy()
            self.patience_counter = 0
            print(f"验证损失改善: {avg_loss:.6f}")
        else:
            self.patience_counter += 1
        
        return avg_loss
    
    def train(self, num_epochs):
        """完整训练过程"""
        print("开始多模态渐进式训练...")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            train_loss, disp_loss, force_loss, physics_loss, h_special_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            if epoch % 10 == 0:
                phase_name = ["基础训练", "H参数重点", "联合微调"][min(self.training_phase-1, 2)]
                print(f"\nEpoch {epoch} ({phase_name}):")
                print(f"  训练损失: {train_loss:.6f}")
                print(f"    - 位移损失: {disp_loss:.6f}")
                print(f"    - 力特征损失: {force_loss:.6f}") 
                print(f"    - 物理损失: {physics_loss:.6f}")
                print(f"    - H专项损失: {h_special_loss:.6f}")
                print(f"  验证损失: {val_loss:.6f}")
                print(f"  学习率: {self.learning_rates[-1]:.2e}")
                print(f"  早停计数: {self.patience_counter}/{self.patience}")
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n早停触发于第 {epoch} 个epoch")
                break
        
        training_time = time.time() - start_time
        print(f"\n多模态训练完成! 总耗时: {training_time:.2f} 秒")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        axes[0, 0].plot(self.train_losses, label='Training Total Loss', linewidth=2)
        axes[0, 0].plot(self.val_losses, label='Validation Loss', linewidth=2)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Multi-modal Training: Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 多模态损失分量
        axes[0, 1].plot(self.disp_losses, label='Displacement Loss', alpha=0.8)
        axes[0, 1].plot(self.force_losses, label='Force Feature Loss', alpha=0.8)
        axes[0, 1].plot(self.physics_losses, label='Physics Loss', alpha=0.8)
        axes[0, 1].plot(self.h_special_losses, label='H Special Loss', alpha=0.8, linewidth=2)
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Multi-modal Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[1, 0].plot(self.learning_rates, color='purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule (with Warm Restarts)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 训练阶段标记
        phases = [50, 100]  # 阶段切换点
        for phase in phases:
            if phase < len(self.train_losses):
                axes[0, 0].axvline(x=phase, color='gray', linestyle='--', alpha=0.5)
                axes[0, 1].axvline(x=phase, color='gray', linestyle='--', alpha=0.5)
                axes[1, 0].axvline(x=phase, color='gray', linestyle='--', alpha=0.5)
        
        # H参数改进分析
        if len(self.h_special_losses) > 0:
            h_improvement = [self.h_special_losses[0] / (loss + 1e-8) for loss in self.h_special_losses]
            axes[1, 1].plot(h_improvement, color='red', linewidth=2, label='H Parameter Loss Improvement')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Improvement Factor')
            axes[1, 1].set_title('H Parameter Learning Progress')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_modal_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

# ==================== 5. 综合评估器 ====================

class ComprehensiveEvaluator:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
    
    def evaluate_on_test_set(self, test_loader):
        """在测试集上全面评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_group_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                coords = batch['coords']
                displacements = batch['displacements']
                true_params = batch['material_params']
                force_features = batch['force_features']
                group_ids = batch['group_id']
                
                disp_inputs = torch.cat([coords, displacements], dim=1)
                pred_params = self.model(disp_inputs, force_features)
                
                loss = F.mse_loss(pred_params, true_params)
                total_loss += loss.item()
                
                all_predictions.append(pred_params)
                all_targets.append(true_params)
                all_group_ids.append(group_ids)
        
        avg_loss = total_loss / len(test_loader)
        
        # 合并结果
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        group_ids = torch.cat(all_group_ids)
        
        # 反归一化参数 - 使用修复后的方法
        predictions_denorm = self.dataset.denormalize_params(predictions.numpy())
        targets_denorm = self.dataset.denormalize_params(targets.numpy())
        
        # 按组计算统计量
        unique_groups = torch.unique(group_ids)
        group_results = {}
        
        for group_id in unique_groups:
            group_mask = (group_ids == group_id)
            group_pred = predictions_denorm[group_mask.numpy()]
            group_target = targets_denorm[group_mask.numpy()]
            
            # 计算该组的平均预测和真实值
            avg_pred = np.mean(group_pred, axis=0)
            avg_target = np.mean(group_target, axis=0)
            
            # 计算相对误差
            rel_error = np.abs(avg_pred - avg_target) / (np.abs(avg_target) + 1e-8) * 100
            
            group_results[group_id.item()] = {
                'predicted': avg_pred,
                'true': avg_target,
                'error': rel_error
            }
        
        return avg_loss, predictions_denorm, targets_denorm, group_results
    
    def visualize_comprehensive_results(self, group_results, save_path='multi_modal_results.png'):
        """可视化多模态结果"""
        param_names = ['E (MPa)', 'v', 'σ_y (MPa)', 'H (MPa)']
        param_names_short = ['E', 'v', 'σ_y', 'H']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 参数反演散点图
        for i, param_name in enumerate(param_names):
            row, col = i // 2, i % 2
            true_vals = []
            pred_vals = []
            
            for results in group_results.values():
                true_vals.append(results['true'][i])
                pred_vals.append(results['predicted'][i])
            
            axes[row, col].scatter(true_vals, pred_vals, alpha=0.7, s=50)
            
            # 绘制理想线
            min_val = min(min(true_vals), min(pred_vals))
            max_val = max(max(true_vals), max(pred_vals))
            axes[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[row, col].set_xlabel('True Value')
            axes[row, col].set_ylabel('Predicted Value')
            axes[row, col].set_title(f'{param_name} Multi-modal Inversion Results')
            axes[row, col].grid(True, alpha=0.3)
            
            # 计算R²分数
            from sklearn.metrics import r2_score
            r2 = r2_score(true_vals, pred_vals)
            
            axes[row, col].text(0.05, 0.95, f'$R^2$ = {r2:.4f}', transform=axes[row, col].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. 误差分布对比
        all_errors = []
        param_colors = ['red', 'blue', 'green', 'orange']
        
        for i, param_name in enumerate(param_names_short):
            errors = []
            for results in group_results.values():
                errors.append(results['error'][i])
            all_errors.extend(errors)
            
            # 绘制每个参数的误差分布
            axes[1, 0].hist(errors, bins=20, alpha=0.6, color=param_colors[i], label=param_name)
        
        axes[1, 0].set_xlabel('Relative Error (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Parameter Error Distribution Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 3. H参数专项分析
        h_errors = []
        h_true_vals = []
        h_pred_vals = []
        
        for results in group_results.values():
            h_errors.append(results['error'][3])
            h_true_vals.append(results['true'][3])
            h_pred_vals.append(results['predicted'][3])
        
        axes[1, 1].scatter(h_true_vals, h_pred_vals, color='orange', alpha=0.7, s=50)
        min_h = min(min(h_true_vals), min(h_pred_vals))
        max_h = max(max(h_true_vals), max(h_pred_vals))
        axes[1, 1].plot([min_h, max_h], [min_h, max_h], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('H True Value (MPa)')
        axes[1, 1].set_ylabel('H Predicted Value (MPa)')
        axes[1, 1].set_title('H Parameter Special Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        
        h_r2 = r2_score(h_true_vals, h_pred_vals)
        axes[1, 1].text(0.05, 0.95, f'$R^2$ = {h_r2:.4f}', transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. 整体统计信息
        mean_errors = []
        for i in range(4):
            errors = [results['error'][i] for results in group_results.values()]
            mean_errors.append(np.mean(errors))
        
        bars = axes[1, 2].bar(range(4), mean_errors, color=param_colors, alpha=0.7)
        axes[1, 2].set_xlabel('Parameter')
        axes[1, 2].set_ylabel('Mean Relative Error (%)')
        axes[1, 2].set_title('Mean Error by Parameter')
        axes[1, 2].set_xticks(range(4))
        axes[1, 2].set_xticklabels(param_names_short)
        axes[1, 2].grid(True, alpha=0.3)
        
        # 在柱状图上添加数值
        for bar, error in zip(bars, mean_errors):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           f'{error:.1f}%', ha='center', va='bottom')
        
        # 添加整体统计
        overall_mean_error = np.mean(all_errors)
        success_rate = np.mean(np.array(all_errors) < 5.0) * 100
        
        fig.text(0.02, 0.02, f'Overall Mean Error: {overall_mean_error:.2f}%\nSuccess Rate (<5% error): {success_rate:.1f}%', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return overall_mean_error, success_rate

# ==================== 6. 主程序 - 多模态PINN ====================

def main_multi_modal_pinn():
    """多模态PINN主程序"""
    print("=" * 60)
    print("Multi-modal PINN: Displacement Field + Force-Displacement Data Fusion")
    print("=" * 60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 加载多模态数据
    print("\n1. Loading multi-modal data...")
    dataset = MultiModalMaterialDataset('all_training_data.csv', 'force_displacement_data.csv', normalize=True)
    
    # 数据集划分
    unique_groups = np.unique(dataset.group_ids)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    train_size = int(0.6 * len(unique_groups))
    val_size = int(0.2 * len(unique_groups))
    test_size = len(unique_groups) - train_size - val_size
    
    train_groups = unique_groups[:train_size]
    val_groups = unique_groups[train_size:train_size+val_size]
    test_groups = unique_groups[train_size+val_size:]
    
    print(f"Dataset split:")
    print(f"  Training set: {len(train_groups)} material groups")
    print(f"  Validation set: {len(val_groups)} material groups") 
    print(f"  Test set: {len(test_groups)} material groups")
    
    # 创建数据加载器
    train_indices = [i for i, group_id in enumerate(dataset.group_ids) if group_id in train_groups]
    val_indices = [i for i, group_id in enumerate(dataset.group_ids) if group_id in val_groups]
    test_indices = [i for i, group_id in enumerate(dataset.group_ids) if group_id in test_groups]
    
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=256,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_indices),
        batch_size=256,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Data loaders created:")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Test samples: {len(test_indices)}")
    
    # 2. 创建多模态PINN模型
    print("\n2. Creating multi-modal PINN model...")
    model = MultiModalPINN(
        disp_input_dim=4,      # [x, y, u, v]
        force_input_dim=9,     # 9个力-位移特征
        output_dim=4,          # [E, v, σ_y, H]
        disp_hidden_dim=128,
        force_hidden_dim=64,
        fusion_dim=256
    )
    
    # 3. 创建多模态损失函数
    print("\n3. Creating multi-modal loss function...")
    loss_function = MultiModalLoss(
        lambda_disp=1.0,      # 位移场数据权重
        lambda_force=0.8,     # 力-位移特征权重  
        lambda_physics=0.1,   # 物理约束权重
        lambda_h=2.0          # H参数专项权重
    )
    
    # 4. 创建渐进式训练器
    print("\n4. Creating progressive trainer...")
    trainer = ProgressiveTrainer(model, loss_function, train_loader, val_loader, test_loader, dataset)
    
    # 5. 训练模型
    print("\n5. Starting multi-modal training...")
    trainer.train(num_epochs=150)
    
    # 6. 加载最佳模型进行评估
    print("\n6. Evaluating best model...")
    if trainer.best_model_state is not None:
        model.load_state_dict(trainer.best_model_state)
    
    evaluator = ComprehensiveEvaluator(model, dataset)
    test_loss, predictions, targets, group_results = evaluator.evaluate_on_test_set(test_loader)
    
    # 7. 可视化结果
    print("\n7. Visualizing multi-modal results...")
    mean_error, success_rate = evaluator.visualize_comprehensive_results(group_results, 'multi_modal_final_results.png')
    
    # 8. 打印详细结果
    print("\n" + "=" * 60)
    print("Multi-modal PINN Results Summary")
    print("=" * 60)
    
    print(f"Test loss: {test_loss:.6f}")
    print(f"Mean relative error: {mean_error:.2f}%")
    print(f"Success rate (<5% error): {success_rate:.2f}%")
    
    # 详细参数误差
    param_names = ['E', 'v', 'σ_y', 'H']
    param_errors = {name: [] for name in param_names}
    
    for results in group_results.values():
        for i, name in enumerate(param_names):
            param_errors[name].append(results['error'][i])
    
    print("\nParameter error statistics:")
    for name in param_names:
        errors = param_errors[name]
        mean_err = np.mean(errors)
        max_err = np.max(errors)
        improvement = "" if name != 'H' else f" (vs Phase 2: {((62.35 - mean_err)/62.35*100):.1f}% improvement)"
        print(f"  {name}: Mean error = {mean_err:.2f}%, Max error = {max_err:.2f}%{improvement}")
    
    # 9. 保存模型和结果
    print("\n8. Saving model and results...")
    torch.save({
        'model_state_dict': trainer.best_model_state,
        'group_results': group_results,
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'disp_losses': trainer.disp_losses,
        'force_losses': trainer.force_losses,
        'physics_losses': trainer.physics_losses,
        'h_special_losses': trainer.h_special_losses,
        'learning_rates': trainer.learning_rates,
        'test_loss': test_loss,
        'mean_error': mean_error,
        'success_rate': success_rate
    }, 'multi_modal_pinn_model.pth')
    
    print("Multi-modal PINN completed! Model saved as 'multi_modal_pinn_model.pth'")
    
    return model, group_results, mean_error, success_rate

if __name__ == "__main__":
    model, results, mean_error, success_rate = main_multi_modal_pinn()