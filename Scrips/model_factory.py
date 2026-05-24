#model_factory

import os
import sys
import importlib
import inspect
from typing import Dict, Any, Optional, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_ROOT = os.path.join(PROJECT_ROOT, "Models")
NEURAL_MODELS_ROOT = os.path.join(MODELS_ROOT, "neural_networks")
STATISTICAL_MODELS_ROOT = os.path.join(MODELS_ROOT, "statisticals")

for path in (MODELS_ROOT, NEURAL_MODELS_ROOT, STATISTICAL_MODELS_ROOT):
    if path not in sys.path:
        sys.path.append(path)

try:
    from config import MODEL_PARAMS_MAP as CONFIG_MODEL_PARAMS_MAP
    MODEL_PARAMS_MAP = CONFIG_MODEL_PARAMS_MAP
    print("[OK] 成功从 config.py 导入 MODEL_PARAMS_MAP")
except ImportError:
    MODEL_PARAMS_MAP = {}
    print("[ERROR] 无法从 config.py 导入 MODEL_PARAMS_MAP")


MODEL_CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    'PredictiveRNN_Model': [
        ('rnn_model', 'PredictiveRNN_Model'),
        ('RNN_GRU', 'PredictiveRNN_Model'),
        ('rnn_gru', 'PredictiveRNN_Model'),
    ],
    'PredictiveBiLSTM_Model': [
        ('bilstm_model', 'PredictiveBiLSTM_Model'),
        ('BI_LSTM', 'PredictiveBiLSTM_Model'),
        ('bi_lstm', 'PredictiveBiLSTM_Model'),
    ],
    'PredictiveSNN_Model': [
        ('snn_model', 'PredictiveSNN_Model'),
        ('SNN', 'PredictiveSNN_Model'),
        ('predictive_snn_model', 'PredictiveSNN_Model'),
    ],
    'PredictiveSNN_ODE_Model': [
        ('snn_ode_model', 'PredictiveSNN_ODE_Model'),
        ('SNN_ODE', 'PredictiveSNN_ODE_Model'),
        ('predictive_snn_ode_model', 'PredictiveSNN_ODE_Model'),
    ],
    'PredictiveSNN_ODEATT_Model': [
        ('SNN_ODEATT', 'ContinuousAttentionSNNODE'),
        ('SNN_ODEATT', 'PredictiveSNNODEATT_Model'),
        ('SNN_ODEATT', 'PredictiveSNN_ODEATT_Model'),
    ],
    'PredictiveLatentODE_Model': [
        ('latent_ode_model', 'PredictiveLatentODE_Model'),
        ('SNN_LatentODE', 'PredictiveLatentODE_Model'),
        ('latent_ode', 'PredictiveLatentODE_Model'),
    ],
    'PredictiveODERNN_Model': [
        ('ode_rnn_model', 'PredictiveODERNN_Model'),
        ('ODE_RNN', 'PredictiveODERNN_Model'),
        ('ode_rnn', 'PredictiveODERNN_Model'),
    ],
    'PredictiveDF2M_Model': [
        ('df2m_model', 'PredictiveDF2M_Model'),
        ('DF2M', 'PredictiveDF2M_Model'),
        ('predictive_df2m_model', 'PredictiveDF2M_Model'),
    ],
    'PredictiveStatLinearAR_Model': [
        ('STAT_MODELS', 'PredictiveStatLinearAR_Model'),
    ],
    'PredictiveStatLassoAR_Model': [
        ('STAT_MODELS', 'PredictiveStatLassoAR_Model'),
    ],
    'PredictiveStatKRR_Model': [
        ('STAT_MODELS', 'PredictiveStatKRR_Model'),
    ],
    'PredictiveStatFPCA_Model': [
        ('STAT_MODELS', 'PredictiveStatFPCA_Model'),
    ],
    'PredictiveStatCumInt_Model': [
        ('STAT_MODELS', 'PredictiveStatCumInt_Model'),
    ],
}


PARAM_NAME_COMPAT = {
    'hidden_dim': 'hidden_size',
    'output_dim': 'out_size',
    'latent_dim': 'latent_size',
    'dropout_rate': 'dropout',
    'tau_mem': 'tau_membrane',
    'tau_syn': 'tau_synaptic',
    'tau_syn_base': 'tau_synaptic',
    'num_layers': 'n_layers',
    'num_clusters': 'n_clusters',
    'num_inducing': 'n_inducing_points',
    'dl_module_type': 'dynamic_module_type',
}


MODEL_DEFAULT_PARAMS = {
    'PredictiveDF2M_Model': {
        'num_clusters': 20,
        'num_inducing': 40,
        'dl_module_type': 'LSTM',
        'if_positive': True,
        'initial_alpha': 10.0,
        'jitter': 1e-6,
        'co_data': True,
    },
    'PredictiveLatentODE_Model': {
        'latent_dim': 64,
    },
    'PredictiveODERNN_Model': {
        'solver_method': 'euler',
    },
}


def _dynamic_import_model(module_name: str, class_name: str):
    module_candidates = [
        f"models.{module_name}",
        module_name,
        f"neural_networks.{module_name}",
        f"statisticals.{module_name}",
    ]

    errors = []
    for import_name in module_candidates:
        try:
            module = importlib.import_module(import_name)
            model_class = getattr(module, class_name)
            print(f"[OK] 成功导入: {class_name} from {import_name}")
            return model_class
        except ImportError as e:
            errors.append(f"{import_name}: {e}")
        except AttributeError as e:
            errors.append(f"{import_name}: {e}")

    print(f"[ERROR] 导入模型失败: {class_name}; tried {errors}")
    return None


def _resolve_model_class(model_name: str):
    if model_name not in MODEL_CANDIDATES:
        raise ValueError(f"未知模型: {model_name}。可用模型: {list(MODEL_CANDIDATES.keys())}")

    errors = []
    for module_name, class_name in MODEL_CANDIDATES[model_name]:
        model_class = _dynamic_import_model(module_name, class_name)
        if model_class is not None:
            return model_class, module_name, class_name
        errors.append(f"models.{module_name}.{class_name}")

    raise ImportError(f"无法导入模型 {model_name}，已尝试: {'；'.join(errors)}")


def _get_valid_params_from_signature(model_class):
    sig = inspect.signature(model_class.__init__)
    return list(sig.parameters.keys())[1:]


def _try_get_value_for_param(param_name: str, source_dict: Dict[str, Any]) -> Any:
    if param_name in source_dict:
        return source_dict[param_name]

    if param_name in PARAM_NAME_COMPAT:
        compat_name = PARAM_NAME_COMPAT[param_name]
        if compat_name in source_dict:
            return source_dict[compat_name]

    for src_name, dst_name in PARAM_NAME_COMPAT.items():
        if dst_name == param_name and src_name in source_dict:
            return source_dict[src_name]

    return None


def _apply_special_parameter_handling(model_name: str,
                                      model_params: Dict[str, Any],
                                      input_dim: int,
                                      kwargs: Dict[str, Any]) -> None:
    if model_name == 'PredictiveDF2M_Model':
        if 'output_dim' not in model_params and 'out_size' not in model_params:
            model_params['output_dim'] = input_dim

        if 'hidden_dim' in kwargs and 'hidden_size' not in model_params:
            model_params['hidden_size'] = kwargs['hidden_dim']
        elif 'hidden_size' in kwargs and 'hidden_dim' not in model_params:
            model_params['hidden_dim'] = kwargs['hidden_size']

        if 'num_clusters' in model_params and 'hidden_dim' in model_params:
            model_params['num_clusters'] = min(model_params['num_clusters'], model_params['hidden_dim'])

    elif model_name == 'PredictiveLatentODE_Model':
        if 'latent_dim' not in model_params:
            model_params['latent_dim'] = 64

    elif model_name == 'PredictiveODERNN_Model':
        if 'solver_method' not in model_params:
            model_params['solver_method'] = 'euler'

    elif model_name == 'PredictiveSNN_ODEATT_Model':
        if 'tau_syn_base' not in model_params:
            if 'tau_syn_base' in kwargs:
                model_params['tau_syn_base'] = kwargs['tau_syn_base']
            elif 'tau_syn' in kwargs:
                model_params['tau_syn_base'] = kwargs['tau_syn']
            elif 'tau_syn' in model_params:
                model_params['tau_syn_base'] = model_params.pop('tau_syn')

        for noisy_key in [
            'solver', 'rtol', 'atol', 'use_soft_threshold',
            'activation', 'reset_mechanism', 'adapt_tau', 'multi_scale',
            'num_layers'
        ]:
            model_params.pop(noisy_key, None)


def _build_model_params(model_class,
                        model_name: str,
                        input_dim: int,
                        kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        valid_params = _get_valid_params_from_signature(model_class)
        model_params: Dict[str, Any] = {}

        clean_kwargs = dict(kwargs)
        clean_kwargs.pop('input_dim', None)

        if 'input_dim' in valid_params:
            model_params['input_dim'] = input_dim

        predefined_params = MODEL_PARAMS_MAP.get(model_name, {})
        for param_name in valid_params:
            if param_name == 'input_dim':
                continue
            value = _try_get_value_for_param(param_name, predefined_params)
            if value is not None and param_name not in clean_kwargs:
                model_params[param_name] = value

        default_params = MODEL_DEFAULT_PARAMS.get(model_name, {})
        for param_name in valid_params:
            if param_name == 'input_dim' or param_name in model_params:
                continue
            value = _try_get_value_for_param(param_name, default_params)
            if value is not None and param_name not in clean_kwargs:
                model_params[param_name] = value

        for param_name in valid_params:
            if param_name == 'input_dim':
                continue
            value = _try_get_value_for_param(param_name, clean_kwargs)
            if value is not None:
                model_params[param_name] = value

        _apply_special_parameter_handling(model_name, model_params, input_dim, clean_kwargs)

        final_params = {k: v for k, v in model_params.items() if k in valid_params}
        print(f"[PARAMS] {model_name} 最终参数: {final_params}")
        return final_params

    except Exception as e:
        print(f"[WARNING] 构建参数失败: {e}")
        return {'input_dim': input_dim}


def _validate_model_params(model_class, model_params: Dict[str, Any]) -> List[str]:
    try:
        sig = inspect.signature(model_class.__init__)
        required_params = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.default == inspect.Parameter.empty and param_name not in model_params:
                required_params.append(param_name)
        return required_params
    except Exception:
        return []


def _create_model_with_fallback(model_class, model_name, model_params, input_dim):
    strategies = [
        lambda: model_class(**model_params),
        lambda: model_class(**{
            k: v for k, v in model_params.items()
            if k not in ['threshold', 'tau_mem', 'tau_syn', 'tau_syn_base']
        }),
        lambda: model_class(
            input_dim=input_dim,
            hidden_dim=model_params.get('hidden_dim', model_params.get('hidden_size', 256))
        ),
        lambda: model_class(input_dim=input_dim),
    ]

    errors = []
    for i, strategy in enumerate(strategies, start=1):
        try:
            print(f"[FALLBACK] 尝试策略 {i}")
            model = strategy()
            print(f"[OK] 使用策略 {i} 成功创建模型: {model_name}")
            return model
        except Exception as e:
            errors.append(f"策略 {i}: {e}")

    raise ValueError(f"无法创建模型 {model_name}: " + "\n".join(errors))


def create_model(model_name: str, input_dim: int, **kwargs) -> Any:
    """Create a model instance from the project registry.

    Parameters
    ----------
    model_name:
        Public model key, for example ``PredictiveSNN_ODEATT_Model``.
    input_dim:
        Feature dimension after data-loader augmentation.
    **kwargs:
        Optional model-specific parameter overrides.

    Returns
    -------
    Any
        Instantiated model object.  The concrete type depends on
        ``model_name`` and is kept dynamic for backward compatibility.
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")

    model_class, module_name, class_name = _resolve_model_class(model_name)

    print(f"\n[MODEL_FACTORY] 创建模型: {model_name}")
    print(f"[MODEL_FACTORY] 模块: models.{module_name}")
    print(f"[MODEL_FACTORY] 类名: {class_name}")
    print(f"[MODEL_FACTORY] 输入维度: {input_dim}")
    print(f"[MODEL_FACTORY] 接收参数: {list(kwargs.keys())}")

    try:
        model_params = _build_model_params(model_class, model_name, input_dim, kwargs)
        missing_params = _validate_model_params(model_class, model_params)
        if missing_params:
            print(f"[WARNING] 模型 {model_name} 缺少必要参数: {missing_params}")

        print(f"[CREATION] 尝试创建 {model_name} ...")
        model = model_class(**model_params)
        print(f"[OK] 成功创建模型: {model_name}")
        return model

    except (TypeError, ValueError) as e:
        print(f"[ERROR] 创建模型 {model_name} 失败: {e}")
        return _create_model_with_fallback(model_class, model_name, model_params, input_dim)

    except Exception as e:
        print(f"[ERROR] 创建模型 {model_name} 出现未知错误: {e}")
        return _create_model_with_fallback(model_class, model_name, model_params, input_dim)


def get_available_models() -> List[str]:
    return list(MODEL_CANDIDATES.keys())


def get_model_requirements(model_name: str) -> Optional[Dict[str, Any]]:
    try:
        model_class, module_name, class_name = _resolve_model_class(model_name)
    except Exception:
        return None

    try:
        sig = inspect.signature(model_class.__init__)
        requirements = {
            'module': f'models.{module_name}',
            'class_name': class_name,
            'required': [],
            'optional': [],
        }

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_info = {
                'name': param_name,
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any',
                'default': None if param.default == inspect.Parameter.empty else param.default,
            }

            if param.default == inspect.Parameter.empty:
                requirements['required'].append(param_info)
            else:
                requirements['optional'].append(param_info)

        return requirements
    except Exception:
        return None


def check_model_availability() -> Dict[str, bool]:
    print("\n[MODEL_FACTORY] 模型可用性检查:")
    available = {}
    for model_name in MODEL_CANDIDATES.keys():
        try:
            _resolve_model_class(model_name)
            available[model_name] = True
            print(f"  [OK] {model_name}")
        except Exception:
            available[model_name] = False
            print(f"  [ERROR] {model_name}")
    return available
