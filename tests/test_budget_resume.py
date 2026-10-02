import pytest
from scripts.resume_training_on_budget import action_for


def test_budget_resume_only_restarts_budget_stops():
    assert action_for({'status':'training'}) == 'wait'
    assert action_for({'status':'failed','last_checkpoint':'checkpoint-001400'}) == 'failed'
    assert action_for({'status':'complete'}) == 'complete'
    assert action_for({'status':'budget_exhausted','last_checkpoint':'checkpoint-001400'}) == 'resume'
    with pytest.raises(ValueError):
        action_for({'status':'budget_exhausted'})
