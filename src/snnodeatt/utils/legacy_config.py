import os
from copy import deepcopy

import numpy as np
import torch


# ====== 路径配置 ======
DATA_DIR = './origin_samples/preprocess_data'
MODEL_SAVE_DIR = './saved_models'
LOG_DIR = './logs'
RESULTS_DIR = './results'
N_FOLDS = 5


# ====== 创建必要目录 ======
def create_directories():
    """创建必要的目录结构"""
    directories = [MODEL_SAVE_DIR, LOG_DIR, RESULTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print(f"[CONFIG] 已创建目录: {[os.path.basename(d) for d in directories]}")


# ====== 通用基础常量 ======
HIDDEN_DIM = 64
THRESHOLD = 1.0
TAU_MEM = 20
TAU_SYN = 10
RESET_MECHANISM = 'subtract'
ADAPT_TAU = True
MULTI_SCALE = True
DROPOUT_RATE = 0.6
N_SAMPLES = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WIN_SIZE = 16
POSITION_ENCODING_DIMS = 2


# ====== 通用训练参数（适用于所有模型） ======
UNIFIED_TRAIN_PARAMS = {
    'num_epochs': 100,
    'batch_size': 16,
    'lr': 1e-4,
    'beta_time': 0.1,
    'early_stopping_patience': 12,
    'stopping_delta': 5e-5,
    'loss_explosion_threshold': 8.0,
    'grad_clip': 1.0,
    'weight_decay': 1e-4,
    'lr_scheduler_step_size': 20,
    'lr_scheduler_gamma': 0.75,
    'odefunc_weight_decay': 5e-4,
    'odefunc_grad_clip': 0.8,
    'num_runs': 1,
    'n_samples': 1,
    'n_splits': N_FOLDS,
    'kwargs': {}
}

UNIFIED_DETECTION_PARAMS = {
    'target_fpr': 0.01,
    'threshold_method': 'f1_optimized',
    'score_normalization': 'robust_iqr',
}

SNN_TRAIN_PARAMS = {
    **UNIFIED_TRAIN_PARAMS,
    'num_epochs': 100,
    'batch_size': 16,
    'lr': 1e-4,
    'early_stopping_patience': 10,
    'grad_clip': 1.0,
    'weight_decay': 1e-4
}


# ====== 损失函数参数 ======
ENHANCED_LOSS_PARAMS = {
    'alpha_recon': 0.6,
    'beta_pred': 0.3,
    'gamma_stability': 0.01,
    'exclude_positional_encoding_from_loss': True,
    'position_encoding_dims': POSITION_ENCODING_DIMS,
}

SNN_ODE_LOSS_PARAMS = {
    'alpha_recon': 0.6,
    'beta_pred': 0.3,
    'gamma_stability': 0.01,
    'exclude_positional_encoding_from_loss': True,
    'position_encoding_dims': POSITION_ENCODING_DIMS,
}


def get_enhanced_loss_params(model_name):
    """统一损失函数参数入口。"""
    base_params = clone_dict(ENHANCED_LOSS_PARAMS)

    print(
        f"[CONFIG] {model_name} 损失参数: "
        f"recon={base_params['alpha_recon']}, "
        f"pred={base_params['beta_pred']}, "
        f"stability={base_params['gamma_stability']}, "
        f"exclude_pe={base_params['exclude_positional_encoding_from_loss']}"
    )
    return base_params


# ====== 参数模板工具 ======
def merge_dicts(*dicts):
    """按顺序合并多个字典，后者覆盖前者"""
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def clone_dict(d):
    """深拷贝字典，避免共享引用"""
    return deepcopy(d)


# ====== 模型参数基类 ======
BASE_SNN_PARAMS = {
    'hidden_dim': HIDDEN_DIM,
    'threshold': THRESHOLD,
    'tau_mem': TAU_MEM,
    'tau_syn': TAU_SYN,
    'reset_mechanism': RESET_MECHANISM,
    'adapt_tau': ADAPT_TAU,
    'multi_scale': MULTI_SCALE,
    'dropout_rate': DROPOUT_RATE
}

BASE_RNN_PARAMS = {
    'hidden_dim': 32,
    'num_layers': 1,
    'dropout': 0.6,
    'output_dim': None
}


# ====== 模型特定参数 ======
RNN_GRU_PARAMS = clone_dict(BASE_RNN_PARAMS)
BI_LSTM_PARAMS = clone_dict(BASE_RNN_PARAMS)
SNN_PARAMS = clone_dict(BASE_SNN_PARAMS)

SNN_ODE_PARAMS = merge_dicts(
    BASE_SNN_PARAMS,
    {
        'hidden_dim': 256,
        'threshold': 0.3,
        'tau_mem': 10.0,
        'tau_syn': 8.0,
        'reset_mechanism': 'subtract',
        'adapt_tau': True,
        'multi_scale': True,
        'dropout_rate': 0.2,
        'use_soft_threshold': True,
        'solver': 'dopri5',
        'rtol': 1e-5,
        'atol': 1e-7,
        'num_layers': 3,
        'activation': 'sigmoid',
    }
)

# 注意：SNN_ODEATT 使用 tau_syn_base
SNN_ODEATT_PARAMS = {
    'hidden_dim': 256,
    'threshold': 0.3,
    'tau_mem': 10.0,
    'tau_syn_base': 8.0,
    'dropout_rate': 0.2,
    'beta_spike': 10.0,
}

LATENT_ODE_PARAMS = merge_dicts(
    BASE_SNN_PARAMS,
    {
        'latent_dim': 32,
        'reset_mechanism': 'subtract',
        'adapt_tau': False,
        'multi_scale': False,
    }
)

ODE_RNN_PARAMS = merge_dicts(
    BASE_SNN_PARAMS,
    {
        'solver_method': 'euler',
        'kwargs': {}
    }
)

DF2M_PARAMS = merge_dicts(
    BASE_SNN_PARAMS,
    {
        'output_dim': None,
        'num_clusters': 6,
        'num_inducing': 10,
        'dl_module_type': 'LSTM',
        'if_positive': True,
        'initial_alpha': 30.0,
        'jitter': 1e-4,
        'co_data': True,
        'hidden_size': HIDDEN_DIM,
        'out_size': HIDDEN_DIM
    }
)

STAT_MODEL_PARAMS = {
    'hidden_dim': 32
}


# ====== 统一模型注册表 ======
MODEL_REGISTRY = {
    'PredictiveSNN_Model': {
        'class_type': 'SNN',
        'params': SNN_PARAMS,
        'active': True,
    },
    'PredictiveSNN_ODE_Model': {
        'class_type': 'SNN_ODE',
        'params': SNN_ODE_PARAMS,
        'active': True,
    },
    'PredictiveSNN_ODEATT_Model': {
        'class_type': 'SNN_ODEATT',
        'params': SNN_ODEATT_PARAMS,
        'active': True,
    },
    'PredictiveLatentODE_Model': {
        'class_type': 'LatentODE',
        'params': LATENT_ODE_PARAMS,
        'active': True,
    },
    'PredictiveODERNN_Model': {
        'class_type': 'ODE_RNN',
        'params': ODE_RNN_PARAMS,
        'active': True,
    },
    'PredictiveRNN_Model': {
        'class_type': 'RNN_GRU',
        'params': RNN_GRU_PARAMS,
        'active': True,
    },
    'PredictiveBiLSTM_Model': {
        'class_type': 'BI_LSTM',
        'params': BI_LSTM_PARAMS,
        'active': True,
    },
    'PredictiveDF2M_Model': {
        'class_type': 'DF2M',
        'params': DF2M_PARAMS,
        'active': True,
    },
    'PredictiveStatLinearAR_Model': {
        'class_type': 'STAT_MODELS',
        'params': STAT_MODEL_PARAMS,
        'active': True,
    },
    'PredictiveStatLassoAR_Model': {
        'class_type': 'STAT_MODELS',
        'params': STAT_MODEL_PARAMS,
        'active': True,
    },
    'PredictiveStatKRR_Model': {
        'class_type': 'STAT_MODELS',
        'params': STAT_MODEL_PARAMS,
        'active': True,
    },
    'PredictiveStatFPCA_Model': {
        'class_type': 'STAT_MODELS',
        'params': STAT_MODEL_PARAMS,
        'active': True,
    },
    'PredictiveStatCumInt_Model': {
        'class_type': 'STAT_MODELS',
        'params': STAT_MODEL_PARAMS,
        'active': True,
    },
}

MODEL_CLASS_MAP = {
    model_name: meta['class_type']
    for model_name, meta in MODEL_REGISTRY.items()
}

MODEL_PARAMS_MAP = {
    model_name: meta['params']
    for model_name, meta in MODEL_REGISTRY.items()
}

ACTIVE_MODELS = [
    model_name
    for model_name, meta in MODEL_REGISTRY.items()
    if meta.get('active', False)
]


# ====== 训练参数映射 ======
SNN_SPECIFIC_TRAIN_PARAMS = {
    'PredictiveSNN_Model': merge_dicts(
        SNN_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveSNN_Model'),
    ),
    'PredictiveSNN_ODE_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        {
            'lr': 1e-4,
            'grad_clip': 1.0,
            'weight_decay': 1e-4,
            'early_stopping_patience': 12,
            'loss_explosion_threshold': 5.0,
        },
        get_enhanced_loss_params('PredictiveSNN_ODE_Model'),
    ),
    'PredictiveSNN_ODEATT_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        {
            'lr': 1e-4,
            'grad_clip': 1.0,
            'weight_decay': 1e-4,
            'odefunc_weight_decay': 5e-4,
            'odefunc_grad_clip': 0.8,
            'early_stopping_patience': 12,
            'loss_explosion_threshold': 5.0,
        },
        get_enhanced_loss_params('PredictiveSNN_ODEATT_Model'),
    ),
    'PredictiveLatentODE_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveLatentODE_Model'),
    ),
    'PredictiveODERNN_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveODERNN_Model'),
    ),
    'PredictiveRNN_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveRNN_Model'),
    ),
    'PredictiveBiLSTM_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveBiLSTM_Model'),
    ),
    'PredictiveDF2M_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveDF2M_Model'),
    ),
    'PredictiveStatLinearAR_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveStatLinearAR_Model'),
    ),
    'PredictiveStatLassoAR_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveStatLassoAR_Model'),
    ),
    'PredictiveStatKRR_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveStatKRR_Model'),
    ),
    'PredictiveStatFPCA_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveStatFPCA_Model'),
    ),
    'PredictiveStatCumInt_Model': merge_dicts(
        UNIFIED_TRAIN_PARAMS,
        get_enhanced_loss_params('PredictiveStatCumInt_Model'),
    ),
}


# ====== 实验模板 ======
EXPERIMENT_SPECS = {
    'exp1': {
        'name': '增强异常_无泊松',
        'abnormal_type': 'augmented_all',
        'poisson': False,
        'suffix': 'exp1',
    },
    'exp2': {
        'name': '原始异常_无泊松',
        'abnormal_type': 'original_6',
        'poisson': False,
        'suffix': 'exp2',
    },
    'exp3': {
        'name': '增强异常_有泊松',
        'abnormal_type': 'augmented_all',
        'poisson': True,
        'suffix': 'exp3',
    },
    'exp4': {
        'name': '原始异常_有泊松',
        'abnormal_type': 'original_6',
        'poisson': True,
        'suffix': 'exp4',
    }
}

MODEL_FILENAME_PREFIX = {
    'PredictiveSNN_Model': 'snn',
    'PredictiveSNN_ODE_Model': 'snn_ode',
    'PredictiveSNN_ODEATT_Model': 'snn_odeatt',
    'PredictiveLatentODE_Model': 'latent_ode',
    'PredictiveODERNN_Model': 'ode_rnn',
    'PredictiveRNN_Model': 'rnn',
    'PredictiveBiLSTM_Model': 'bilstm',
    'PredictiveDF2M_Model': 'df2m',
    'PredictiveStatLinearAR_Model': 'stat_linear_ar',
    'PredictiveStatLassoAR_Model': 'stat_lasso_ar',
    'PredictiveStatKRR_Model': 'stat_krr',
    'PredictiveStatFPCA_Model': 'stat_fpca',
    'PredictiveStatCumInt_Model': 'stat_cumint',
}


def build_experiment_models(exp_suffix, active_models=None):
    """根据实验后缀自动构建模型保存文件名映射"""
    selected_models = active_models if active_models is not None else ACTIVE_MODELS
    model_files = {}

    for model_name in selected_models:
        if model_name not in MODEL_FILENAME_PREFIX:
            raise ValueError(f"模型 {model_name} 未配置文件名前缀")
        prefix = MODEL_FILENAME_PREFIX[model_name]
        model_files[model_name] = f'{prefix}_{exp_suffix}.pth'

    return model_files


def build_experiments():
    """由实验模板自动生成完整实验配置"""
    experiments = {}
    for exp_key, spec in EXPERIMENT_SPECS.items():
        experiments[exp_key] = {
            'name': spec['name'],
            'abnormal_type': spec['abnormal_type'],
            'poisson': spec['poisson'],
            'models': build_experiment_models(spec['suffix'])
        }
    return experiments


EXPERIMENTS = build_experiments()


# ====== 评估参数 ======
EVAL_PARAMS = {
    'n_samples': 5,
    'anomaly_threshold': 0.99,
    'prediction_steps': 5,
    'confidence_level': 0.99,
    'target_fpr': 0.01,
    'score_clip_value': 100.0,
    'robust_scaling': True,
}


# ====== 数据预处理参数 ======
DATA_PARAMS = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'normalize': True,
    'scale_method': 'minmax',
    'sequence_length': WIN_SIZE,
    'stride': 8,
    'shuffle': True,
    'random_seed': 42
}


# ====== 日志配置 ======
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_log': True,
    'console_log': True
}


# ====== 初始化目录 ======
create_directories()


# ====== 数据维度检测 ======
def detect_input_dim(data_path, position_encoding_dims=2):
    """根据数据文件自动检测输入维度"""
    files = [
        f for f in os.listdir(data_path)
        if f.endswith('.csv') or f.endswith('.npy')
    ]
    if not files:
        raise FileNotFoundError(f"在 {data_path} 中未找到数据文件")

    first_file = os.path.join(data_path, files[0])
    print(f"[CONFIG] 检测输入维度，使用文件: {first_file}")

    if first_file.endswith('.csv'):
        first_row = np.loadtxt(first_file, delimiter=",", max_rows=1, dtype=str)
        feature_dim = int(np.atleast_1d(first_row).shape[0])
    elif first_file.endswith('.npy'):
        arr = np.load(first_file)
        feature_dim = arr.shape[-1]
    else:
        raise ValueError("不支持的数据格式")

    detected_dim = feature_dim + position_encoding_dims
    print(
        f"[CONFIG] 检测到原始特征维度: {feature_dim}，"
        f"位置编码: {position_encoding_dims}，"
        f"最终输入维度: {detected_dim}"
    )
    return feature_dim, detected_dim


try:
    RAW_FEATURE_DIM, EXPECTED_INPUT_DIM = detect_input_dim(
        DATA_DIR,
        position_encoding_dims=POSITION_ENCODING_DIMS
    )
except Exception as e:
    RAW_FEATURE_DIM = None
    EXPECTED_INPUT_DIM = None
    print(f"[CONFIG WARNING] 输入维度自动检测失败: {e}")


# ====== 模型类别集合 ======
SNN_BASED_MODELS = [
    'PredictiveSNN_Model',
    'PredictiveSNN_ODE_Model',
    'PredictiveSNN_ODEATT_Model',
    'PredictiveLatentODE_Model',
    'PredictiveODERNN_Model',
    'PredictiveDF2M_Model'
]

NON_SNN_REMOVE_KEYS = [
    'threshold',
    'tau_mem',
    'tau_syn',
    'tau_syn_base',
    'reset_mechanism',
    'adapt_tau',
    'multi_scale'
]


# ====== 工具函数 ======
def validate_and_adjust_params(model_name, params):
    """根据模型类型验证和调整参数"""
    if model_name not in SNN_BASED_MODELS:
        for param in NON_SNN_REMOVE_KEYS:
            params.pop(param, None)

    if model_name == 'PredictiveDF2M_Model':
        if 'num_clusters' in params and 'hidden_dim' in params:
            if params['num_clusters'] > params['hidden_dim']:
                params['num_clusters'] = min(params['num_clusters'], params['hidden_dim'])

    if model_name == 'PredictiveSNN_ODEATT_Model':
        if 'tau_syn_base' not in params and 'tau_syn' in params:
            params['tau_syn_base'] = params.pop('tau_syn')

    return params


def get_model_params(model_name, input_dim):
    """获取模型参数，使用传入的 input_dim"""
    if model_name not in MODEL_PARAMS_MAP:
        raise ValueError(f"未知模型名称: {model_name}")

    params = clone_dict(MODEL_PARAMS_MAP[model_name])
    params['input_dim'] = input_dim
    params = validate_and_adjust_params(model_name, params)

    print(f"[CONFIG] {model_name} 使用输入维度: {input_dim}")
    print(
        f"[CONFIG] {model_name} 参数: "
        f"input_dim={input_dim}, "
        f"hidden_dim={params.get('hidden_dim', 'N/A')}, "
        f"latent_dim={params.get('latent_dim', 'N/A')}"
    )
    return params


def get_training_params(model_name):
    params = clone_dict(SNN_SPECIFIC_TRAIN_PARAMS.get(model_name, UNIFIED_TRAIN_PARAMS))
    params['raw_feature_dim'] = RAW_FEATURE_DIM
    params['position_encoding_dims'] = POSITION_ENCODING_DIMS
    params['n_splits'] = params.get('n_splits', N_FOLDS)
    return params


def get_experiment_models(exp_key):
    """获取实验对应的模型配置"""
    if exp_key not in EXPERIMENTS:
        raise ValueError(f"未知实验: {exp_key}")

    experiment = EXPERIMENTS[exp_key]
    print(f"[CONFIG] 实验 {exp_key}: {experiment['name']}")
    print(f"[CONFIG] 异常类型: {experiment['abnormal_type']}, 泊松采样: {experiment['poisson']}")
    print(f"[CONFIG] 包含模型: {list(experiment['models'].keys())}")
    return experiment['models']


def update_model_params(model_name, **kwargs):
    """动态更新模型参数"""
    if model_name not in MODEL_PARAMS_MAP:
        raise ValueError(f"未知模型名称: {model_name}")

    MODEL_PARAMS_MAP[model_name].update(kwargs)
    MODEL_REGISTRY[model_name]['params'].update(kwargs)
    print(f"[CONFIG] 已更新 {model_name} 的参数: {list(kwargs.keys())}")


def get_experiment_config(exp_key):
    """获取完整实验配置"""
    if exp_key not in EXPERIMENTS:
        raise ValueError(f"未知实验: {exp_key}")
    return EXPERIMENTS[exp_key]


def get_model_save_path(model_name, exp_key):
    """获取模型保存路径"""
    experiment_models = get_experiment_models(exp_key)
    if model_name not in experiment_models:
        raise ValueError(f"模型 {model_name} 不在实验 {exp_key} 的配置中")

    model_filename = experiment_models[model_name]
    return os.path.join(MODEL_SAVE_DIR, model_filename)


def get_all_model_names():
    return list(MODEL_PARAMS_MAP.keys())


def get_model_type(model_name):
    return MODEL_CLASS_MAP.get(model_name, 'Unknown')


def validate_config():
    """验证配置完整性"""
    issues = []

    required_dirs = [DATA_DIR, MODEL_SAVE_DIR, LOG_DIR, RESULTS_DIR]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"目录不存在: {dir_path}")

    for model_name in ACTIVE_MODELS:
        if model_name not in MODEL_PARAMS_MAP:
            issues.append(f"模型 {model_name} 不在 MODEL_PARAMS_MAP 中")
        if model_name not in MODEL_CLASS_MAP:
            issues.append(f"模型 {model_name} 不在 MODEL_CLASS_MAP 中")

    for exp_key, exp_config in EXPERIMENTS.items():
        for model_name in exp_config['models']:
            if model_name not in MODEL_PARAMS_MAP:
                issues.append(f"实验 {exp_key} 中的模型 {model_name} 未配置参数")
            if model_name not in MODEL_FILENAME_PREFIX:
                issues.append(f"实验 {exp_key} 中的模型 {model_name} 未配置文件名前缀")

    if issues:
        print("[WARN] 配置验证发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("[OK] 配置验证通过")
        return True


def print_config_summary():
    """打印配置摘要"""
    print("\n" + "=" * 50)
    print("配置摘要")
    print("=" * 50)
    print(f"设备: {DEVICE}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型保存目录: {MODEL_SAVE_DIR}")
    print(f"激活模型数量: {len(ACTIVE_MODELS)}")
    print(f"激活模型列表: {ACTIVE_MODELS}")
    print(f"实验数量: {len(EXPERIMENTS)}")
    print(f"实验列表: {list(EXPERIMENTS.keys())}")
    print("=" * 50)


def get_training_runs(exp_key=None):
    return UNIFIED_TRAIN_PARAMS.get('num_runs', 1)


# demo1/2/3/4 与 exp1/2/3/4 显式对应，供入口与推理统一引用
DEMO_TO_EXP = {
    "demo1": "exp1",
    "demo2": "exp2",
    "demo3": "exp3",
    "demo4": "exp4",
}


if __name__ == "__main__":
    validate_config()
    print_config_summary()
