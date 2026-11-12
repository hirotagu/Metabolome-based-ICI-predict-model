"""Comment 6-6 sensitivity jobs: refit Datasets 3, 4, and 5 only.

The R2 mathematical functions are retained in ICI_predict_v3.py and
Compare_sensitivity.py. This module only selects the prespecified scenarios,
sets their original grids, and supplies per-unit orchestration/provenance.
Run each execute_job call in a fresh process, as the full-run launcher does.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
CANONICAL_KEYS = (
    'dataset3_pre', 'dataset3_post1', 'dataset3_post2',
    'dataset4_pre', 'dataset4_post1', 'dataset4_post2',
    'dataset5_pre', 'dataset5_post1', 'dataset5_post2',
)
FORMAL_PRE_KEYS = tuple(f'dataset{i}_pre' for i in (3, 4, 5))
DEFAULT_EN = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1, 3]
DEFAULT_L2 = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3]
EXPANDED_EN = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1, 3, 10, 30]
EXPANDED_L2 = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10, 30]
# (family, root, sheet, imputation, max_k, outer_splits, source, keys)
SCENARIOS = {
    'knn': ('knn', 'All_data_KNN', 'Sheet1', 'knn', 10, 5,
            'imputation_sensitivity', FORMAL_PRE_KEYS),
    'kmax30': ('kmax', 'All_data_Kmax30', 'Sheet1', 'half_minimum', 30, 5,
               'kmax_sensitivity', CANONICAL_KEYS),
    'outer3': ('outer_folds', 'All_data_outer3', 'Sheet1', 'half_minimum', 10, 3,
               'fold_sensitivity', CANONICAL_KEYS),
    'reggrid': ('regularization_grid', 'All_data_RegGrid', 'Sheet1', 'half_minimum', 10, 5,
                'regularization_grid_sensitivity', FORMAL_PRE_KEYS),
}
SUMMARY_FILES = ('comparison_all_scenarios.csv', 'scenario_summary.csv',
                 'primary_delta_reference.csv', 'run_manifest.csv', 'warnings_all.csv',
                 'source_artifact_hashes.csv', 'settings_used.json', 'run_receipt.json')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_path(scenario: str, dataset: str) -> str:
    return f'{SCENARIOS[scenario][1]}/supplement/sensitivity/{scenario}/{dataset}'


def _compare_path(scenario: str, dataset: str) -> str:
    return f'Compare_sensitivity_v1/datasets/{scenario}/{dataset}'


def plan_jobs() -> list[dict[str, Any]]:
    nested = []
    compare = []
    for scenario, values in SCENARIOS.items():
        for dataset in values[7]:
            nested.append({'id': f'sensitivity_nested:{scenario}:{dataset}',
                           'kind': 'sensitivity_nested', 'scenario': scenario, 'dataset': dataset,
                           'depends_on': [], 'owned_paths': [_nested_path(scenario, dataset)]})
            compare.append({'id': f'sensitivity_compare:{scenario}:{dataset}',
                            'kind': 'sensitivity_compare', 'scenario': scenario, 'dataset': dataset,
                            'depends_on': ['sensitivity:nested_manifest', 'core:compare_summary'],
                            'owned_paths': [_compare_path(scenario, dataset)]})
    roots = sorted({values[1] for values in SCENARIOS.values()})
    barrier = {'id': 'sensitivity:nested_manifest', 'kind': 'sensitivity_manifest',
               'depends_on': [j['id'] for j in nested],
               'owned_paths': [f'{root}/run_manifest.csv' for root in roots]}
    summary = {'id': 'sensitivity:summary', 'kind': 'sensitivity_summary',
               'depends_on': [j['id'] for j in compare],
               'owned_paths': [f'Compare_sensitivity_v1/{name}' for name in SUMMARY_FILES]}
    return nested + [barrier] + compare + [summary]


def _validate_job(job: Mapping[str, Any]) -> None:
    known = {j['id']: j for j in plan_jobs()}
    if job.get('id') not in known:
        raise ValueError(f'Unknown sensitivity job: {job.get("id")}')
    expected = known[job['id']]
    for key in ('kind', 'scenario', 'dataset'):
        if job.get(key) != expected.get(key):
            raise ValueError(f'Sensitivity job specification mismatch: {key}')


def _configure_engine(scenario: str, inputs: Mapping[str, Path], results: Path):
    import ICI_predict_v3 as engine
    missing = set(CANONICAL_KEYS) - set(inputs)
    if missing:
        raise ValueError(f'Comment 6-6 inputs missing: {sorted(missing)}')
    values = SCENARIOS[scenario]
    # Explicit paths prevent unchanged datasets present in inputs from entering any fit.
    engine.INPUT_FILES = [str(Path(inputs[k]).resolve()) for k in CANONICAL_KEYS]
    engine.INPUT_DIR = Path(inputs[CANONICAL_KEYS[0]]).resolve().parent
    engine.SEARCH_INPUT_RECURSIVELY = False
    engine.DATASETS_TO_RUN = list(CANONICAL_KEYS)
    engine.INPUT_METADATA_OVERRIDES = {}
    engine.PRIMARY_SHEET_OVERRIDES = {}
    engine.OUTPUT_ROOT = Path(results).resolve()
    engine.RUN_TAG = values[1]
    engine.ALLOW_OVERWRITE = False
    engine.EN_C_GRID = list(EXPANDED_EN if scenario == 'reggrid' else DEFAULT_EN)
    engine.L2_C_GRID = list(EXPANDED_L2 if scenario == 'reggrid' else DEFAULT_L2)
    # This was explicitly False in the submitted expanded-grid source.
    # Keep the warnings; do not silently remove this sensitivity or tighten it.
    engine.HARD_FAIL_ON_NONCONVERGENCE = scenario != 'reggrid'
    return engine


def _scenario(engine, scenario: str):
    values = SCENARIOS[scenario]
    return engine.ScenarioSpec(name=scenario, sheet_name=values[2], imputation=values[3],
                               max_k=values[4], outer_splits_target=values[5], source=values[6])


def _configure_compare(engine, scenario: str, results: Path, context: Mapping[str, Any]):
    import Compare_sensitivity as compare
    values = SCENARIOS[scenario]
    compare.PRIMARY_COMPARE_RUN_ROOT = Path(context.get('primary_compare_root', results / 'Compare_method_v1'))
    compare.OUTPUT_ROOT = results
    compare.RUN_TAG = 'Compare_sensitivity_v1'
    # Limit the primary reference to the nine newly corrected matrices.
    # The original strict input/code/output checks remain active.
    compare.CANONICAL_KEYS = CANONICAL_KEYS
    compare.FORMAL_PRE_KEYS = FORMAL_PRE_KEYS
    compare.KNN_KEYS = FORMAL_PRE_KEYS
    compare.STRICT_PRODUCER_SCRIPT_HASH = True
    compare.STRICT_PRIMARY_REFERENCE_HASH = True
    compare.WRITE_EXCEL = False  # One final workbook is produced by report_results.py.
    compare.PERFORM_HYPOTHESIS_TESTS = False
    compare.COMPUTE_BOOTSTRAP_INTERVALS = False
    engine_path = Path(engine.__file__).resolve()
    compare._ENGINE_CACHE.clear()
    compare._ENGINE_CACHE[engine_path] = engine
    config = compare.ScenarioConfig(
        family=values[0], name=scenario, run_root=results / values[1],
        producer_script=engine_path, expected_keys=values[7], sheet_name=values[2],
        imputation=values[3], max_k=values[4], outer_splits_target=values[5],
        source_label=values[6], same_analysis_population_as_primary=not scenario.startswith('sd_'))
    # Ceiling flags are derived anew, rather than carrying R2's historical 14/17.
    import pandas as pd
    triggered = []
    for key in CANONICAL_KEYS:
        section = 'supplement/dataset6_small_sample' if key.startswith('dataset6_') else 'main'
        path = results / 'All_data_primary' / section / 'primary' / key / 'outer_parameters.csv'
        if not path.is_file():
            raise FileNotFoundError(f'Missing corrected primary parameters: {path}')
        params = pd.read_csv(path)
        if 'k_selected' not in params or params.empty:
            raise ValueError(f'Invalid primary parameters: {path}')
        if (params['k_selected'].astype(int) >= 10).any():
            triggered.append(key)
    compare.KMAX_PRIMARY_TRIGGER_KEYS = tuple(triggered)
    return compare, config


def _record_dispatcher(directory: Path, job: Mapping[str, Any], engine) -> None:
    path = directory / 'run_receipt.json'
    payload = json.loads(path.read_text())
    payload['execution_dispatcher'] = str(Path(__file__).resolve())
    payload['execution_dispatcher_sha256'] = _sha(Path(__file__))
    payload['execution_job'] = job['id']
    payload['single_scientific_engine_sha256'] = _sha(Path(engine.__file__).resolve())
    payload['hard_fail_on_nonconvergence'] = bool(engine.HARD_FAIL_ON_NONCONVERGENCE)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n')


def _write_nested_manifests(inputs: Mapping[str, Path], results: Path) -> dict[str, Any]:
    import pandas as pd
    import ICI_revision_common as common
    by_root: dict[str, list[dict[str, Any]]] = {}
    count = 0
    for scenario, values in SCENARIOS.items():
        engine = _configure_engine(scenario, inputs, results)
        for key in values[7]:
            directory = results / _nested_path(scenario, key)
            receipt = common.read_json(directory / 'run_receipt.json')
            settings = common.read_json(directory / 'settings_used.json')
            if receipt.get('status') != 'completed' or receipt.get('scenario') != scenario or receipt.get('dataset') != key:
                raise ValueError(f'Nested sensitivity receipt mismatch: {scenario}/{key}')
            if receipt.get('script_sha256') != _sha(Path(engine.__file__)):
                raise ValueError(f'Nested sensitivity engine changed: {scenario}/{key}')
            if receipt.get('execution_dispatcher_sha256') != _sha(Path(__file__)):
                raise ValueError(f'Nested sensitivity dispatcher changed: {scenario}/{key}')
            if receipt.get('input_sha256') != _sha(Path(inputs[key])):
                raise ValueError(f'Nested sensitivity input changed: {scenario}/{key}')
            wanted = _scenario(engine, scenario)
            from dataclasses import asdict
            if settings.get('scenario') != asdict(wanted):
                raise ValueError(f'Nested sensitivity scenario differs: {scenario}/{key}')
            for field, expected in [('en_C_grid', engine.EN_C_GRID), ('l2_C_grid', engine.L2_C_GRID)]:
                if settings.get(field) != expected:
                    raise ValueError(f'Nested sensitivity grid differs: {scenario}/{key}/{field}')
            params = common.read_csv(directory / 'outer_parameters.csv')
            metric = common.read_csv(directory / 'metrics_summary.csv')
            if len(metric) != 1:
                raise ValueError(f'Nested metrics must have one row: {scenario}/{key}')
            row = metric.iloc[0]
            by_root.setdefault(values[1], []).append({
                'dataset': key, 'scenario': scenario, 'status': 'completed',
                'outer_splits_used': receipt['outer_splits_used'],
                'max_selected_k': int(params['k_selected'].max()),
                'primary_k_ceiling_hit': bool((params['k_selected'] >= wanted.max_k).any()),
                'roc_auc_oof': row['roc_auc_oof'], 'average_precision_oof': row['average_precision_oof'],
                'brier_oof': row['brier_oof'], 'output_dir': str(directory.resolve())})
            count += 1
    if count != 24:
        raise AssertionError(f'Expected 24 scoped nested sensitivity units, got {count}')
    for root, rows in by_root.items():
        common.write_csv(pd.DataFrame(rows), results / root / 'run_manifest.csv')
    return {'completed_nested_units': count, 'family_roots': list(by_root)}


def _summary(inputs: Mapping[str, Path], results: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd
    import ICI_revision_common as common
    started_time, started_utc = time.time(), common.utc_now()
    comparisons, manifests, warnings, hashes = [], [], [], []
    primary_reference = None
    for scenario, values in SCENARIOS.items():
        engine = _configure_engine(scenario, inputs, results)
        compare, config = _configure_compare(engine, scenario, results, context)
        if primary_reference is None:
            primary_reference, primary_hashes = compare._validate_primary_reference()
            hashes.append(primary_hashes)
        for key in values[7]:
            complete = compare._load_completed_unit(results / 'Compare_sensitivity_v1', config, key)
            if complete is None:
                raise FileNotFoundError(f'Missing comparison: {scenario}/{key}')
            row, source_hashes, warning_frame = complete
            receipt = common.read_json(results / _compare_path(scenario, key) / 'run_receipt.json')
            if receipt.get('execution_dispatcher_sha256') != _sha(Path(__file__)):
                raise ValueError(f'Comparison dispatcher changed: {scenario}/{key}')
            comparisons.append(row)
            hashes.append(source_hashes)
            if not warning_frame.empty:
                warnings.append(warning_frame.assign(scenario=scenario, source_dataset=key))
            manifests.append({'family': values[0], 'scenario': scenario, 'dataset': key,
                              'status': 'completed', 'execution_action': 'validated',
                              'output_dir': str((results / _compare_path(scenario, key)).resolve()),
                              'message': '', 'nested_roc_auc': row['nested_roc_auc'],
                              'nonnested_roc_auc': row['nonnested_roc_auc'],
                              'delta_auc': row['delta_auc_nonnested_minus_nested'],
                              'change_delta_auc_vs_primary': row['change_delta_auc_vs_primary']})
    if len(comparisons) != 24:
        raise AssertionError('Expected all 24 scoped sensitivity comparisons')
    root = results / 'Compare_sensitivity_v1'
    _, manifest, _, _ = compare._write_top_outputs(root, comparisons, manifests, primary_reference, warnings, hashes)
    common.finalize_run_receipt(root, Path(compare.__file__), started_utc, started_time, manifest, {
        'analysis': 'scenario_matched_nested_vs_fully_non_nested_sensitivity',
        'single_engine': str(Path(engine.__file__).resolve()),
        'execution_dispatcher_sha256': _sha(Path(__file__)),
        'n_expected_units': 24, 'scenario_names': list(SCENARIOS),
        'hypothesis_tests': 'not performed', 'bootstrap_intervals': 'not computed by comparison',
        'post_analysis': 'descriptive; repeated time points are not independent cohorts',
        'unaffected_datasets': 'not included in this public D3-D5 sensitivity run',
        'sd_analysis': 'not performed; Dataset 1 is outside this public sensitivity run',
        'kmax30_scope': 'D3-D5 nine matrices; ceiling flags derived from their current primary outer fits',
        'kmax_primary_triggered_keys': list(compare.KMAX_PRIMARY_TRIGGER_KEYS),
        'expanded_grid_nonconvergence': 'original HARD_FAIL=False retained; inspect warnings',
        'publication_values_status': 'public_reproduction_run',
        'scope': 'Comment 6-6: Datasets 3, 4, and 5 only; no three-cohort hypothesis test'})
    return {'completed_comparison_units': 24, 'output_root': str(root)}


def execute_job(job: Mapping[str, Any], inputs: Mapping[str, Path], results: Path,
                context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _validate_job(job)
    results = Path(results).resolve()
    inputs = {str(k): Path(v).resolve() for k, v in inputs.items()}
    context = {} if context is None else context
    kind = job['kind']
    if kind == 'sensitivity_manifest':
        return _write_nested_manifests(inputs, results)
    if kind == 'sensitivity_summary':
        return _summary(inputs, results, context)
    scenario, key = job['scenario'], job['dataset']
    engine = _configure_engine(scenario, inputs, results)
    if kind == 'sensitivity_nested':
        specs = engine.discover_datasets()
        result = engine.run_nested_cv(specs[key], _scenario(engine, scenario))
        directory = results / _nested_path(scenario, key)
        _record_dispatcher(directory, job, engine)
        return {'output_dir': str(directory), 'dataset': key, 'scenario': scenario,
                'metrics': result.metrics}
    if kind == 'sensitivity_compare':
        compare, config = _configure_compare(engine, scenario, results, context)
        reference, primary_hashes = compare._validate_primary_reference()
        artifact = compare._validate_sensitivity_artifacts(config, key)
        import pandas as pd
        artifact.source_hashes = pd.concat(
            [artifact.source_hashes, primary_hashes], ignore_index=True
        ).drop_duplicates(['path', 'sha256']).reset_index(drop=True)
        directory = results / _compare_path(scenario, key)
        directory.mkdir(parents=True, exist_ok=False)
        metrics = compare._run_matching_non_nested(artifact, reference, directory)
        _record_dispatcher(directory, job, engine)
        return {'output_dir': str(directory), 'dataset': key, 'scenario': scenario,
                'metrics': metrics}
    raise ValueError(f'Unknown sensitivity job kind: {kind}')
