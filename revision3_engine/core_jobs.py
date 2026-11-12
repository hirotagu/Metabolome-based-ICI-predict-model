"""One-unit workers for Reviewer 6-6: only Dataset 3/4/5 are refitted.

The coordinator must execute these jobs serially in fresh Python processes and
validate/quarantine only each job's declared owned paths before retrying.
Aggregate files are written once at explicit barriers so downstream source
hashes remain stable while later jobs are running.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

ENGINE = Path(__file__).resolve().parent
ALL_KEYS = (
    'dataset1_pre', 'dataset1_post1',
    'dataset2_pre', 'dataset2_post1', 'dataset2_post2',
    'dataset3_pre', 'dataset3_post1', 'dataset3_post2',
    'dataset4_pre', 'dataset4_post1', 'dataset4_post2',
    'dataset5_pre', 'dataset5_post1', 'dataset5_post2',
    'dataset6_pre', 'dataset6_post1', 'dataset7_pre',
)
CANONICAL_KEYS = tuple(k for k in ALL_KEYS if k.startswith(('dataset3_', 'dataset4_', 'dataset5_')))
PRE_KEYS = tuple(f'dataset{i}_pre' for i in (3, 4, 5))
MAIN_TAG = 'All_data_primary'
COMPARE_TAG = 'Compare_method_v1'
NEST_TAG = 'Compare_nest_v1'
HOLDOUT_TAG = 'Holdout_v1'
RETRY_TAG = 'Holdout_d3-2_V2'
CALIBRATION_TAG = 'Metrics_Calibration_v1'
RETRY_KEY = 'dataset3_post2'


def _primary_relative(key: str) -> str:
    prefix = 'supplement/dataset6_small_sample/primary' if key.startswith('dataset6_') else 'main/primary'
    return f'{MAIN_TAG}/{prefix}/{key}'


def _unit_relative(stage: str, key: str) -> str:
    if stage == 'nested':
        return _primary_relative(key)
    tags = {'compare': COMPARE_TAG, 'nest': NEST_TAG, 'calibration': CALIBRATION_TAG,
            'holdout': RETRY_TAG if key == RETRY_KEY else HOLDOUT_TAG}
    if stage not in tags:
        raise ValueError(f'Unknown core stage: {stage}')
    return f'{tags[stage]}/datasets/{key}'


def _aggregate_paths(tag: str, filenames: tuple[str, ...]) -> list[str]:
    return [f'{tag}/{name}' for name in filenames]


def plan_jobs() -> list[dict[str, Any]]:
    """Return only the 6-6 affected core jobs; no unaffected cohort can be fitted."""
    jobs: list[dict[str, Any]] = []
    def unit(stage: str, key: str, deps: list[str]) -> None:
        jobs.append({'id': f'core:{stage}:{key}', 'family': 'core', 'stage': stage,
                     'dataset': key, 'depends_on': deps,
                     'owned_paths': [_unit_relative(stage, key)]})
    def barrier(stage: str, deps: list[str], owned: list[str]) -> None:
        jobs.append({'id': f'core:{stage}', 'family': 'core', 'stage': stage,
                     'depends_on': deps, 'owned_paths': owned})
    common_files = ('run_manifest.csv', 'settings_used.json', 'run_receipt.json')
    for key in CANONICAL_KEYS:
        unit('nested', key, [])
    barrier('main_summary', [f'core:nested:{k}' for k in CANONICAL_KEYS],
            _aggregate_paths(MAIN_TAG, common_files))
    for key in CANONICAL_KEYS:
        unit('compare', key, ['core:main_summary'])
    barrier('compare_summary', [f'core:compare:{k}' for k in CANONICAL_KEYS],
            _aggregate_paths(COMPARE_TAG, common_files + ('comparison_all_datasets.csv', 'pre_primary_inference.csv')))
    for key in CANONICAL_KEYS:
        unit('holdout', key, [])
    barrier('holdout_summary', [f'core:holdout:{k}' for k in CANONICAL_KEYS],
            _aggregate_paths(HOLDOUT_TAG, common_files + ('holdout_summary_all_datasets.csv',)) +
            _aggregate_paths(RETRY_TAG, common_files + ('holdout_summary_all_datasets.csv',)))
    for key in PRE_KEYS:
        unit('nest', key, ['core:compare_summary'])
    barrier('nest_summary', [f'core:nest:{k}' for k in PRE_KEYS],
            _aggregate_paths(NEST_TAG, common_files + ('nesting_comparison_all_pre.csv',)))
    for key in CANONICAL_KEYS:
        unit('calibration', key, ['core:compare_summary', 'core:holdout_summary'])
    barrier('calibration_summary', [f'core:calibration:{k}' for k in CANONICAL_KEYS],
            _aggregate_paths(CALIBRATION_TAG, common_files + (
                'calibration_metrics_all_datasets.csv', 'calibration_primary_metrics_all_datasets.csv',
                'calibration_group_summary.csv')))
    return jobs


def _modules(inputs: dict[str, Path], results: Path):
    """Load exactly the bundled current modules and connect explicit inputs."""
    if set(inputs) not in (set(CANONICAL_KEYS), set(ALL_KEYS)):
        raise ValueError('Core execution requires the nine affected inputs, optionally accompanied by the eight unchanged canonical inputs')
    for key in CANONICAL_KEYS:
        path = inputs[key]
        path = Path(path)
        if not path.is_file() or path.name != f'{key}.xlsx':
            raise ValueError(f'Invalid canonical input: {key}: {path}')
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
    base = importlib.import_module('ICI_predict_v3')
    common = importlib.import_module('ICI_revision_common')
    for module in (base, common):
        if Path(module.__file__).resolve().parent != ENGINE:
            raise RuntimeError(f'Wrong source module already imported: {module.__file__}')
    base.INPUT_FILES = [str(Path(inputs[key]).resolve()) for key in CANONICAL_KEYS]
    base.INPUT_DIR = Path(inputs[CANONICAL_KEYS[0]]).resolve().parent
    base.DATASETS_TO_RUN = list(CANONICAL_KEYS)
    base.PRIMARY_TARGETS = list(CANONICAL_KEYS)
    base.OUTPUT_ROOT = results
    base.RUN_TAG = MAIN_TAG
    base.ALLOW_OVERWRITE = False
    base.EN_MAX_ITER_LIMIT = 20000
    specs = base.discover_datasets()
    if set(specs) != set(CANONICAL_KEYS):
        raise ValueError('Discovered input dataset set differs from the nine affected D3/D4/D5 matrices')
    return base, common, specs


def _load(name: str):
    if name != 'Holdout_d3-2':
        module = importlib.import_module(name)
    else:
        label = '_r3_holdout_d3_2'
        spec = importlib.util.spec_from_file_location(label, ENGINE / 'Holdout_d3-2.py')
        if spec is None or spec.loader is None:
            raise ImportError('Cannot load D3 Post2 holdout producer')
        module = importlib.util.module_from_spec(spec)
        sys.modules[label] = module
        spec.loader.exec_module(module)
    if Path(module.__file__).resolve().parent != ENGINE:
        raise RuntimeError(f'Wrong producer imported: {module.__file__}')
    return module


def _configure(module, results: Path, tag: str) -> None:
    module.OUTPUT_ROOT = results
    module.RUN_TAG = tag
    module.ALLOW_OVERWRITE = False
    if hasattr(module, 'STRICT_BASE_SCRIPT_HASH'):
        module.STRICT_BASE_SCRIPT_HASH = True
    if hasattr(module, 'CONTINUE_ON_DATASET_ERROR'):
        module.CONTINUE_ON_DATASET_ERROR = False
    if hasattr(module, 'MAIN_RUN_ROOT'):
        module.MAIN_RUN_ROOT = results / MAIN_TAG
    if hasattr(module, 'COMPARE_METHOD_RUN_ROOT'):
        module.COMPARE_METHOD_RUN_ROOT = results / COMPARE_TAG
    if hasattr(module, 'HOLDOUT_DEFAULT_RUN_ROOT'):
        module.HOLDOUT_DEFAULT_RUN_ROOT = results / HOLDOUT_TAG
        module.HOLDOUT_DEFAULT_PRODUCER = ENGINE / 'Holdout.py'
        module.HOLDOUT_RETRY_RUN_ROOT = results / RETRY_TAG
        module.HOLDOUT_RETRY_PRODUCER = ENGINE / 'Holdout_d3-2.py'


def _checked_receipt(directory: Path, key: str, producer: Path, base, common,
                     input_path: Path) -> dict[str, Any]:
    receipt = common.read_json(directory / 'run_receipt.json')
    if receipt.get('status') != 'completed' or receipt.get('dataset') != key:
        raise ValueError(f'Incomplete or mismatched result: {directory}')
    if receipt.get('script_sha256') != common.sha256_file(producer):
        raise ValueError(f'Producer hash mismatch: {directory}')
    if receipt.get('input_sha256') != common.sha256_file(input_path):
        raise ValueError(f'Input hash mismatch: {directory}')
    for field, path in (('base_script_sha256', Path(base.__file__)),
                        ('common_module_sha256', Path(common.__file__))):
        if field in receipt and receipt[field] != common.sha256_file(path):
            raise ValueError(f'{field} mismatch: {directory}')
    hashes = receipt.get('output_sha256')
    if isinstance(hashes, dict):
        common.validate_output_sha256_manifest(directory, receipt, list(hashes))
    return receipt


def _manifest_row(key: str, directory: Path, specs, common) -> dict[str, Any]:
    return {'dataset': key, 'status': 'completed',
            'analysis_role': common.dataset_role(specs[key]),
            'output_dir': str(directory.resolve()), 'message': ''}


def _assert_free_paths(job: dict[str, Any], results: Path) -> None:
    for relative in job['owned_paths']:
        destination = results / relative
        if destination.exists():
            raise FileExistsError(f'Job output already exists; coordinator must validate resume or archive partial output: {destination}')


def _summary(stage: str, inputs: dict[str, Path], results: Path, base, common, specs) -> dict[str, Any]:
    import pandas as pd
    started = time.time()
    started_utc = common.utc_now()
    if stage == 'main_summary':
        rows = []
        for key in CANONICAL_KEYS:
            directory = results / _primary_relative(key)
            artifact = common.validate_main_artifacts(specs[key], directory, strict_script_hash=True)
            receipt = common.read_json(directory / 'run_receipt.json')
            params = artifact.parameters
            # Source-defined master-manifest fields, plus explicit completion.
            row = _manifest_row(key, directory, specs, common)
            row.update({'scenario': 'primary', 'outer_splits_used': artifact.outer_plan.n_splits,
                        'max_selected_k': int(params['k_selected'].max()),
                        'primary_k_ceiling_hit': bool((params['k_selected'] >= base.PRIMARY_MAX_K).any()),
                        **{field: receipt['metrics'][field] for field in (
                            'roc_auc_oof', 'average_precision_oof', 'brier_oof')}})
            rows.append(row)
        root = results / MAIN_TAG
        manifest = pd.DataFrame(rows)
        common.write_csv(manifest, root / 'run_manifest.csv')
        common.finalize_run_receipt(root, Path(base.__file__), started_utc, started, manifest,
            {'analysis': 'reviewer_6_6_affected_primary_nested', 'datasets_to_run': list(CANONICAL_KEYS),
             'scientific_producer_functions_unchanged': True, 'wrapper_sha256': common.sha256_file(Path(__file__))})
        return {'status': 'completed', 'n_datasets': len(rows), 'output_dir': str(root)}

    if stage == 'holdout_summary':
        groups = ((HOLDOUT_TAG, [k for k in CANONICAL_KEYS if k != RETRY_KEY], 'Holdout'),
                  (RETRY_TAG, [RETRY_KEY], 'Holdout_d3-2'))
    else:
        config = {'compare_summary': (COMPARE_TAG, CANONICAL_KEYS, 'Compare_method'),
                  'nest_summary': (NEST_TAG, PRE_KEYS, 'Compare_nest'),
                  'calibration_summary': (CALIBRATION_TAG, CANONICAL_KEYS, 'Metrics_Calibration')}
        if stage not in config:
            raise ValueError(f'Unknown summary stage: {stage}')
        groups = (config[stage],)
    finished = []
    for tag, keys, producer_name in groups:
        module = _load(producer_name)
        _configure(module, results, tag)
        root = results / tag
        rows, tables, primary_tables = [], [], []
        for key in keys:
            directory = root / 'datasets' / key
            receipt = _checked_receipt(directory, key, Path(module.__file__), base, common, inputs[key])
            rows.append(_manifest_row(key, directory, specs, common))
            name = 'calibration_metrics_all_units.csv' if stage == 'calibration_summary' else 'metrics_summary.csv'
            frame = common.read_csv(directory / name)
            if set(frame['dataset'].astype(str)) != {key}:
                raise ValueError(f'Wrong dataset in {directory / name}')
            tables.append(frame)
            if stage == 'calibration_summary':
                primary_tables.append(common.read_csv(directory / 'calibration_primary_metrics.csv'))
        manifest = pd.DataFrame(rows)
        combined = pd.concat(tables, ignore_index=True)
        common.write_csv(manifest, root / 'run_manifest.csv')
        settings: dict[str, Any] = {'datasets_to_run': list(keys), 'wrapper_sha256': common.sha256_file(Path(__file__)),
                                   'collection_complete': True, 'scientific_producer_functions_unchanged': True,
                                   'scope': 'Only Dataset 3/4/5 affected by Reviewer 6-6',
                                   'unaffected_cohorts_refitted': False}
        if stage == 'compare_summary':
            common.write_csv(combined, root / 'comparison_all_datasets.csv')
            common.write_csv(pd.DataFrame([{'test': 'not_run',
                'inferential_role': 'primary', 'status': 'not_performed_scoped_update',
                'required_datasets': '|'.join(module.FORMAL_PRE_KEYS),
                'available_updated_datasets': '|'.join(PRE_KEYS),
                'reason': 'Only the three affected public pre-ICI cohorts are included; no across-cohort hypothesis test is performed.'}]),
                root / 'pre_primary_inference.csv')
            settings.update({'analysis': 'fully_nested_vs_fully_non_nested',
                             'main_run_root': str(results / MAIN_TAG),
                             'formal_pre_keys': list(module.FORMAL_PRE_KEYS), 'bootstrap_n': module.BOOTSTRAP_N,
                             'max_k': module.MAX_K, 'random_state': module.RANDOM_STATE,
                             'formal_primary_test': 'not performed on the three updated cohorts',
                             'formal_sensitivity_test': 'not performed on the three updated cohorts',
                             'post_tests': 'not performed', 'dataset6_test': 'not performed; descriptive supplement'})
        elif stage == 'nest_summary':
            common.write_csv(combined, root / 'nesting_comparison_all_pre.csv')
            settings.update({'analysis': 'four_way_nesting_diagnostic', 'main_run_root': str(results / MAIN_TAG),
                             'compare_method_run_root': str(results / COMPARE_TAG),
                             'methods': list(module.METHOD_ORDER), 'hybrids_are_valid_performance_estimators': False,
                             'formal_tests': 'not performed', 'max_k': module.MAX_K})
        elif stage == 'holdout_summary':
            common.write_csv(combined, root / 'holdout_summary_all_datasets.csv')
            settings.update({'analysis': 'prespecified_and_repeated_stratified_holdout',
                             'test_size': module.TEST_SIZE, 'holdout_splits': module.HOLDOUT_SPLITS,
                             'random_state': module.RANDOM_STATE, 'fixed_split': 1, 'max_k': module.MAX_K,
                             'fixed_split_bootstrap_n': module.FIXED_SPLIT_BOOTSTRAP_N,
                             'inferential_tests': 'not performed'})
            if tag == RETRY_TAG:
                settings['en_max_iter_limit'] = module.HOLDOUT_EN_MAX_ITER_LIMIT
        elif stage == 'calibration_summary':
            primary = pd.concat(primary_tables, ignore_index=True)
            common.write_csv(combined, root / 'calibration_metrics_all_datasets.csv')
            common.write_csv(primary, root / 'calibration_primary_metrics_all_datasets.csv')
            common.write_csv(module._group_summary(primary), root / 'calibration_group_summary.csv')
            settings.update({'analysis': 'saved_prediction_calibration_and_prior_sensitivity',
                             'main_run_root': str(results / MAIN_TAG), 'compare_method_run_root': str(results / COMPARE_TAG),
                             'holdout_default_run_root': str(results / HOLDOUT_TAG),
                             'holdout_retry_run_root': str(results / RETRY_TAG),
                             'holdout_retry_key': RETRY_KEY, 'holdout_retry_en_max_iter_limit': 80000,
                             'methods': list(module.METHOD_ORDER), 'probability_variants': list(module.PROBABILITY_VARIANTS),
                             'bootstrap_enabled': module.RUN_BOOTSTRAP_CI, 'bootstrap_n': module.BOOTSTRAP_N,
                             'raw_probability_role': 'primary', 'prior_adjusted_probability_role': 'Supplement sensitivity',
                             'formal_tests': 'not performed'})
        common.finalize_run_receipt(root, Path(module.__file__), started_utc, started, manifest, settings)
        finished.append(str(root))
    return {'status': 'completed', 'output_dirs': finished}


def execute_job(job: dict[str, Any], inputs: dict[str, Path], results: Path,
                context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one declared job; coordinator owns resume/locking and run-level identity."""
    expected = {j['id']: j for j in plan_jobs()}
    if job.get('id') not in expected:
        raise ValueError(f'Unknown core job: {job.get("id")}')
    canonical_job = expected[job['id']]
    for field in ('stage', 'dataset', 'owned_paths', 'depends_on'):
        if job.get(field) != canonical_job.get(field):
            raise ValueError(f'Modified job contract: {job["id"]}: {field}')
    results = Path(results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    _assert_free_paths(job, results)
    base, common, specs = _modules(inputs, results)
    stage = job['stage']
    if stage.endswith('_summary'):
        return _summary(stage, inputs, results, base, common, specs)
    key = job['dataset']
    directory = results / _unit_relative(stage, key)
    if stage == 'nested':
        result = base.run_nested_cv(specs[key], base._primary_scenario(specs[key]))
        if Path(result.output_dir).resolve() != directory:
            raise RuntimeError('Nested producer output escaped declared job path')
        common.validate_main_artifacts(specs[key], directory, strict_script_hash=True)
        return {'status': 'completed', 'dataset': key, 'output_dir': str(directory)}
    names = {'compare': 'Compare_method', 'nest': 'Compare_nest', 'calibration': 'Metrics_Calibration',
             'holdout': 'Holdout_d3-2' if key == RETRY_KEY else 'Holdout'}
    module = _load(names[stage])
    tag = directory.parent.parent.name
    _configure(module, results, tag)
    module.DATASETS_TO_RUN = [key]
    run_root = results / tag
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / 'datasets').mkdir(exist_ok=True)
    (run_root / 'failed').mkdir(exist_ok=True)
    (run_root / '_staging').mkdir(exist_ok=True)
    if stage == 'holdout':
        if module.TEST_SIZE != 0.30 or module.HOLDOUT_SPLITS != 25 or module.RANDOM_STATE != 42:
            raise ValueError('Holdout protocol changed')
        if key == RETRY_KEY:
            if base.EN_MAX_ITER_LIMIT != 20000 or module.HOLDOUT_EN_MAX_ITER_LIMIT != 80000:
                raise ValueError('D3 Post2 holdout convergence-limit protocol changed')
            base.EN_MAX_ITER_LIMIT = 80000
        with common.staged_dataset_dir(run_root, key) as output:
            module.run_dataset(specs[key], output)
    else:
        main_dir = common.resolve_main_run_dirs(results / MAIN_TAG, specs, [key])[key]
        artifact = common.validate_main_artifacts(specs[key], main_dir, strict_script_hash=True)
        with common.staged_dataset_dir(run_root, key) as output:
            if stage == 'compare':
                module.run_non_nested(artifact, output)
            elif stage == 'nest':
                comparison = module._load_compare_method(artifact)
                module.run_dataset(artifact, comparison, output)
            elif stage == 'calibration':
                module.analyze_dataset(artifact, output)
    _checked_receipt(directory, key, Path(module.__file__), base, common, inputs[key])
    return {'status': 'completed', 'dataset': key, 'output_dir': str(directory)}
