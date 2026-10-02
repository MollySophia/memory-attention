"""Check controller gate ordering with fake subprocesses; no GPU work."""
import json
from pathlib import Path
import sys
import pytest
import continue_validation as controller
import reuse_correctness


@pytest.mark.parametrize('outcome', ['confirmation_failed','correctness_failed','passed'])
def test_reused_reference_does_not_skip_candidate_or_gate(outcome,tmp_path,monkeypatch):
    root=tmp_path/'campaign';attempt=root/'A0016';attempt.mkdir(parents=True)
    confirmation=root/'confirmation';confirmation.mkdir()
    (confirmation/'baseline.json').write_text('{}')
    manifest=confirmation/'manifest.json'
    manifest.write_text(json.dumps(dict(controller_pid=-1,status='completed',jobs=[
        dict(name='baseline',implementation='baseline',variant='ma_gpu')])) )
    references={(b,s):root/f'reference-b{b}-s{s}.json' for b in (1,8) for s in (1234,4321)}
    monkeypatch.setattr(controller,'ROOT',root)
    monkeypatch.setattr(reuse_correctness,'reference_index',lambda *args:references)
    commands=[]
    class Process:
        pid=12345
        def __init__(self,command,**kwargs):
            commands.append(command)
            self.result=0
            script=Path(command[1]).name
            if script=='analyze_paired.py':
                Path(command[command.index('--output')+1]).write_text(json.dumps(
                    dict(confirmation_promising=outcome!='confirmation_failed')))
            elif script=='compare_correctness.py' and outcome=='correctness_failed':
                self.result=1
            elif script=='run_paired.py':
                output=Path(command[command.index('--output')+1]);output.mkdir()
                (output/'manifest.json').write_text(json.dumps(dict(status='completed',jobs=[dict(status='completed')])))
        def wait(self):return self.result
    monkeypatch.setattr(controller.subprocess,'Popen',Process)
    monkeypatch.setattr(sys,'argv',['continue_validation.py','--attempt','A0016',
        '--confirmation',str(manifest),'--baseline-root',str(root/'baseline'),
        '--candidate-root',str(root/'candidate'),'--screen-manifest',str(root/'screen.json'),
        '--correctness-first','--reuse-resident-correctness',str(root/'references'),
        '--first-run-index','4'])
    if outcome=='correctness_failed':
        with pytest.raises(RuntimeError):controller.main()
    else:
        assert controller.main()==0
    state=json.loads((attempt/'validation-controller.json').read_text())
    assert state['accepted'] is False
    scripts=[Path(command[1]).name for command in commands]
    if outcome!='passed':
        assert 'run_paired.py' not in scripts
        assert state['status']==('confirmation_needs_review' if outcome=='confirmation_failed' else 'failed_needs_review')
    else:
        assert state['status']=='validation_complete_needs_audit'
        candidate=[command for command in commands if Path(command[1]).name=='full_correctness.py']
        assert len(candidate)==4
        assert all(command[command.index('--variant')+1]=='ma_offload' for command in candidate)
        assert scripts[:9]==['analyze_paired.py']+['full_correctness.py','compare_correctness.py']*4
        assert scripts.count('run_paired.py')==4
        assert len([job for job in state['jobs'] if job['status']=='reused'])==4
