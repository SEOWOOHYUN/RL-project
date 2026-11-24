import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.torque_experiment import summarize, config_for_preset


def row(success, steps, time, effort):
    return dict(is_success=success, elapsed_steps=steps, elapsed_time_s=time,
                release_time_s=.1, success_time_s=time if success else None,
                torque_squared_integral=effort, absolute_work_j=2.,
                normalized_effort=.3, target_x=1.25, target_z=.9)


def test_efficiency_summary_keeps_failures_out_of_success_time_but_in_success_rate():
    result = summarize([row(True, 10, .2, 1), row(False, 2, .04, .1), row(True, 20, .4, 3)])
    assert result['success_rate'] == pytest.approx(2/3)
    assert result['successful_episodes']['elapsed_steps']['mean'] == 15
    assert result['successful_episodes']['success_time_s']['mean'] == pytest.approx(.3)
    assert result['all_episodes']['torque_squared_integral']['mean'] == pytest.approx(4.1/3)


def test_no_success_reports_null_not_zero_efficiency():
    result = summarize([row(False, 2, .04, .1)])
    assert result['success_rate'] == 0
    assert result['successful_episodes']['success_time_s']['mean'] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        summarize([])


def test_presets_only_change_efficiency_weights():
    from dataclasses import asdict
    baseline = asdict(config_for_preset('baseline'))
    time = asdict(config_for_preset('time'))
    efficient = asdict(config_for_preset('efficient'))
    assert baseline.pop('time_weight') == 0
    assert baseline.pop('effort_weight') == 0
    assert time.pop('time_weight') > 0
    assert time.pop('effort_weight') == 0
    assert efficient.pop('time_weight') > 0
    assert efficient.pop('effort_weight') > 0
    assert baseline == time == efficient


def test_real_ppo_train_save_reload_and_paired_evaluate(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'torque_experiment.py'
    run = tmp_path / 'run'
    subprocess.run([sys.executable, str(script), 'train', '--out', str(run),
                    '--timesteps', '64', '--n-steps', '32', '--batch-size', '16',
                    '--seed', '3'], check=True, capture_output=True, text=True)
    assert (run / 'final_model.zip').exists()
    assert (run / 'vecnormalize.pkl').exists()
    config = json.loads((run / 'config.json').read_text())
    assert config['environment']['torque_limits'] == [8., 5.]
    outputs = []
    for name in ('eval1', 'eval2'):
        out = tmp_path / name
        subprocess.run([sys.executable, str(script), 'evaluate', '--run', str(run),
                        '--episodes', '2', '--seed', '100', '--targets', '1.1,.8', '1.3,1.',
                        '--out', str(out)], check=True, capture_output=True, text=True)
        outputs.append(json.loads((out / 'evaluation.json').read_text()))
    assert outputs[0] == outputs[1]
    assert outputs[0]['summary']['episodes'] == 4
    assert len(outputs[0]['per_target']) == 2
    assert len(outputs[0]['episodes']) == 4
