import os
import argparse
import json
from datetime import datetime
import numpy as np

from train import train_all_models_kfold
from config import (
    DATA_DIR,
    EXPERIMENTS,
    ACTIVE_MODELS,
    get_experiment_models,
    UNIFIED_TRAIN_PARAMS,
    MODEL_SAVE_DIR,
    N_FOLDS,
)
from utils import (
    setup_experiment_logging,
    save_experiment_config,
    setup_results_directory,
    save_training_metrics
)


def check_model_exists(exp_key, model_name, poisson=False, use_cross_validation=True, n_splits=None):
    """
    检查模型是否已经训练并保存
    适配交叉验证：检查所有折的模型文件
    """
    model_dir = MODEL_SAVE_DIR

    if n_splits is None:
        n_splits = UNIFIED_TRAIN_PARAMS.get('n_splits', N_FOLDS)

    if not os.path.exists(model_dir):
        print(f"[INFO] 模型目录不存在: {model_dir}")
        return False

    try:
        experiment_models = get_experiment_models(exp_key)
    except ValueError as e:
        print(f"[ERROR] 获取实验模型配置失败: {e}")
        return False

    if model_name not in experiment_models:
        print(f"[WARN] 模型 {model_name} 不在实验 {exp_key} 配置中")
        return False

    base_filename = experiment_models[model_name]

    if use_cross_validation:
        all_folds_exist = True
        for fold in range(1, n_splits + 1):
            fold_filename = base_filename.replace('.pth', f'_fold{fold}.pth')
            model_path = os.path.join(model_dir, fold_filename)

            if not os.path.exists(model_path):
                print(f"[CHECK] 模型折 {fold} 不存在: {model_path}")
                all_folds_exist = False
                break
            else:
                print(f"[CHECK] 模型折 {fold} 存在: {model_path}")

        return all_folds_exist
    else:
        model_path = os.path.join(model_dir, base_filename)
        exists = os.path.exists(model_path)
        print(f"[CHECK] 模型检查: {model_path} -> {'存在' if exists else '不存在'}")
        return exists


def get_existing_models(exp_key, model_names, poisson=False, use_cross_validation=True, n_splits=None):
    """
    获取已经存在的模型列表
    """
    existing_models = []
    models_to_train = []

    for model_name in model_names:
        if check_model_exists(
            exp_key,
            model_name,
            poisson=poisson,
            use_cross_validation=use_cross_validation,
            n_splits=n_splits
        ):
            existing_models.append(model_name)
        else:
            models_to_train.append(model_name)

    print(f"[STATUS] 现有模型: {existing_models}")
    print(f"[STATUS] 需要训练: {models_to_train}")

    return existing_models, models_to_train


def validate_experiment_setup(exp_key, data_dir, models_to_run):
    """
    验证实验设置
    """
    issues = []

    if exp_key not in EXPERIMENTS:
        issues.append(f"未知实验标识: {exp_key}")
        return issues

    if not os.path.exists(data_dir):
        issues.append(f"数据目录不存在: {data_dir}")

    experiment_models = get_experiment_models(exp_key)

    if models_to_run:
        for model_name in models_to_run:
            if model_name not in experiment_models:
                issues.append(f"模型 {model_name} 不在实验 {exp_key} 的配置中")

    return issues


def process_cross_validation_results(training_results):
    """
    处理交叉验证训练结果，提取关键指标
    """
    cv_summary = {}

    for model_name, cv_result in training_results.items():
        if isinstance(cv_result, dict) and 'all_results' in cv_result:
            fold_results = []
            best_val_losses = []
            final_epochs = []

            for fold_key, fold_data in cv_result['all_results'].items():
                training_info = fold_data.get('training_results', {})
                model_info = fold_data.get('model_info', {})
                fold_results.append({
                    'fold': fold_key,
                    'fold_id': model_info.get('fold_id', 'unknown'),
                    'best_val_loss': training_info.get('best_val_loss', 0),
                    'final_epoch': training_info.get('final_epoch', 0),
                    'train_loss_history': training_info.get('train_loss_history', []),
                    'val_loss_history': training_info.get('val_loss_history', [])
                })
                best_val_losses.append(training_info.get('best_val_loss', 0))
                final_epochs.append(training_info.get('final_epoch', 0))

            cv_summary[model_name] = {
                'cv_results': fold_results,
                'mean_best_val_loss': float(np.mean(best_val_losses)) if best_val_losses else 0,
                'std_best_val_loss': float(np.std(best_val_losses)) if best_val_losses else 0,
                'mean_final_epoch': float(np.mean(final_epochs)) if final_epochs else 0,
                'total_folds': len(fold_results),
                'n_folds': cv_result.get('n_folds', 0),
                'completed_folds': cv_result.get('completed_folds', 0)
            }
        else:
            cv_summary[model_name] = {
                'single_training': True,
                'best_val_loss': cv_result.get('training_results', {}).get('best_val_loss', 0),
                'final_epoch': cv_result.get('training_results', {}).get('final_epoch', 0)
            }

    return cv_summary


def generate_training_summary(results):
    """
    生成训练汇总报告
    """
    summary = {
        'experiment_key': results['experiment_info']['experiment_key'],
        'total_models_trained': 0,
        'total_models_failed': 0,
        'failed_models': {},
        'training_success': False,
        'use_cross_validation': results['experiment_info'].get('use_cross_validation', False),
        'n_splits': results['experiment_info'].get('n_splits'),
        'timestamp': datetime.now().isoformat()
    }

    training_results = results.get('training_results', {})
    if training_results:
        summary['total_models_trained'] = training_results.get('total_models_trained', 0)
        summary['total_models_failed'] = training_results.get('total_models_failed', 0)
        summary['failed_models'] = training_results.get('failed_models', {})
        summary['training_success'] = summary['total_models_trained'] > 0

        if 'cv_summary' in training_results:
            summary['cross_validation_summary'] = training_results['cv_summary']

    return summary


def print_training_summary(results):
    """
    打印训练汇总报告
    """
    if not results or 'summary' not in results:
        print("[ERROR] 训练结果为空")
        return

    summary = results['summary']
    exp_info = results.get('experiment_info', {})
    exp_config = exp_info.get('experiment_config', {})

    print(f"\n{'='*60}")
    print(f"训练汇总报告: {exp_config.get('name', '未知实验')}")
    print(f"{'='*60}")
    print(f"实验标识: {summary.get('experiment_key', '未知')}")
    print(f"使用交叉验证: {'[YES]' if summary.get('use_cross_validation') else '[NO]'}")
    if summary.get('use_cross_validation'):
        print(f"交叉验证折数: {summary.get('n_splits', '未知')}")
    print(f"训练成功: {'[YES]' if summary.get('training_success') else '[NO]'} ({summary.get('total_models_trained', 0)} 个模型)")
    print(f"训练失败模型数: {summary.get('total_models_failed', 0)}")

    if summary.get('failed_models'):
        print("\n失败模型:")
        for model_name, err in summary['failed_models'].items():
            print(f"  - {model_name}: {err}")

    if summary.get('cross_validation_summary'):
        print(f"\n{'='*60}")
        print("交叉验证训练摘要:")
        print(f"{'='*60}")

        for model_name, cv_info in summary['cross_validation_summary'].items():
            if 'cv_results' in cv_info:
                print(f"\n[CV] {model_name} (K折交叉验证):")
                print(f"   平均最佳验证损失: {cv_info['mean_best_val_loss']:.6f} ± {cv_info['std_best_val_loss']:.6f}")
                print(f"   平均训练轮数: {cv_info['mean_final_epoch']:.1f}")
                print(f"   成功折数: {cv_info['completed_folds']}/{cv_info['n_folds']}")


def run_single_experiment(
    exp_key,
    data_dir=DATA_DIR,
    models_to_run=None,
    results_base_dir='./results',
    force_retrain=False,
    use_cross_validation=True,
    n_splits=None
):
    """
    运行单个实验（仅训练）
    """
    if n_splits is None:
        n_splits = UNIFIED_TRAIN_PARAMS.get('n_splits', N_FOLDS)

    exp_config = EXPERIMENTS[exp_key]
    apply_poisson_sampling = exp_config['poisson']
    base_lambda = exp_config.get('base_lambda', 0.5)

    logger = setup_experiment_logging(exp_key)
    results_dir = setup_results_directory(exp_key, results_base_dir)

    logger.info(f"开始实验训练: {exp_key}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"结果目录: {results_dir}")
    logger.info(f"使用交叉验证: {use_cross_validation}")
    if use_cross_validation:
        logger.info(f"交叉验证折数: {n_splits}")

    validation_issues = validate_experiment_setup(exp_key, data_dir, models_to_run)
    if validation_issues:
        for issue in validation_issues:
            logger.error(issue)
        return None

    logger.info(f"实验配置: {exp_config['name']}")
    logger.info(f"异常类型: {exp_config['abnormal_type']}")
    logger.info(f"泊松采样: {exp_config['poisson']}")

    # 默认目标模型：该实验配置中的全部模型
    if models_to_run is None:
        models_to_run = list(get_experiment_models(exp_key).keys())

    logger.info(f"指定的模型: {models_to_run}")

    existing_models, missing_models = get_existing_models(
        exp_key,
        models_to_run,
        exp_config['poisson'],
        use_cross_validation,
        n_splits=n_splits
    )

    # ===== force_retrain 核心修复 =====
    if force_retrain:
        logger.info("force_retrain=True，忽略已有模型检查，强制重新训练指定模型")
        models_to_train = list(models_to_run)
    else:
        models_to_train = list(missing_models)

    print(f"[STATUS] force_retrain={force_retrain}")
    print(f"[STATUS] 最终需要训练: {models_to_train}")

    if not force_retrain and not models_to_train:
        logger.info("所有指定模型已存在，跳过训练阶段")
        return {
            'experiment_info': {
                'experiment_key': exp_key,
                'experiment_config': exp_config,
                'models_to_run': models_to_run,
                'existing_models': existing_models,
                'data_dir': data_dir,
                'force_retrain': force_retrain,
                'use_cross_validation': use_cross_validation,
                'n_splits': n_splits if use_cross_validation else None,
                'timestamp': datetime.now().isoformat()
            },
            'training_results': {
                'training_summary': {},
                'failed_models': {},
                'experiment_key': exp_key,
                'total_models_trained': 0,
                'total_models_failed': 0,
                'skip_reason': 'models_already_exist',
                'use_cross_validation': use_cross_validation
            },
            'summary': {
                'experiment_key': exp_key,
                'total_models_trained': 0,
                'total_models_failed': 0,
                'failed_models': {},
                'training_success': False,
                'use_cross_validation': use_cross_validation,
                'n_splits': n_splits if use_cross_validation else None,
                'timestamp': datetime.now().isoformat(),
                'message': '所有模型已存在，跳过训练'
            }
        }

    config_to_save = {
        'experiment_key': exp_key,
        'experiment_config': exp_config,
        'models_to_run': models_to_run,
        'existing_models': existing_models,
        'models_for_training': models_to_train,
        'data_dir': data_dir,
        'force_retrain': force_retrain,
        'use_cross_validation': use_cross_validation,
        'n_splits': n_splits if use_cross_validation else None,
        'timestamp': datetime.now().isoformat()
    }

    config_file = save_experiment_config(config_to_save, exp_key, results_dir)
    logger.info(f"实验配置已保存: {config_file}")

    results = {
        'experiment_info': config_to_save,
        'training_results': {},
        'summary': {}
    }

    logger.info("开始训练阶段...")
    try:
        training_summary = train_all_models_kfold(
            data_dir=data_dir,
            exp_key=exp_key,
            model_names=models_to_train,
            num_runs=1,
            n_folds=n_splits,
            apply_poisson_sampling=apply_poisson_sampling,
            base_lambda=base_lambda
        )

        if training_summary and 'training_summary' in training_summary:
            results['training_results'] = training_summary
            trained_count = training_summary.get('total_models_trained', 0)
            failed_count = training_summary.get('total_models_failed', 0)
            logger.info(f"训练阶段完成，成功训练 {trained_count} 个模型，失败 {failed_count} 个模型")

            cv_summary = process_cross_validation_results(training_summary['training_summary'])
            results['training_results']['cv_summary'] = cv_summary

            for model_name, training_result in training_summary['training_summary'].items():
                if isinstance(training_result, dict) and 'all_results' in training_result:
                    for fold_key, fold_data in training_result['all_results'].items():
                        training_results_data = fold_data.get('training_results', {})
                        if training_results_data:
                            metrics_file = save_training_metrics(
                                training_results_data,
                                exp_key,
                                f"{model_name}_{fold_key}",
                                results_dir
                            )
                            logger.info(f"训练指标已保存: {metrics_file}")
                else:
                    training_results_data = training_result.get('training_results', {})
                    if training_results_data:
                        metrics_file = save_training_metrics(
                            training_results_data,
                            exp_key,
                            model_name,
                            results_dir
                        )
                        logger.info(f"训练指标已保存: {metrics_file}")
        else:
            logger.error("训练阶段返回了空结果")
            results['training_results'] = {}

    except Exception as e:
        logger.error(f"训练阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return results

    results['summary'] = generate_training_summary(results)

    results_file = os.path.join(results_dir, f'training_results_{exp_key}.json')
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"训练完成，结果已保存: {results_file}")
    except Exception as e:
        logger.error(f"保存结果文件失败: {e}")

    print_training_summary(results)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='实验键，例如 exp1')
    parser.add_argument('--models', nargs='*', default=None, help='指定模型名列表')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--force_retrain', action='store_true')
    parser.add_argument('--no_cv', action='store_true')
    parser.add_argument('--n_splits', type=int, default=N_FOLDS)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_key = args.exp
    models_to_run = args.models if args.models else None
    use_cross_validation = not args.no_cv

    if exp_key not in EXPERIMENTS:
        raise ValueError(f"未知实验标识: {exp_key}。可选实验: {list(EXPERIMENTS.keys())}")

    run_single_experiment(
        exp_key=exp_key,
        data_dir=args.data_dir,
        models_to_run=models_to_run,
        force_retrain=args.force_retrain,
        use_cross_validation=use_cross_validation,
        n_splits=args.n_splits,
    )


if __name__ == '__main__':
    main()
