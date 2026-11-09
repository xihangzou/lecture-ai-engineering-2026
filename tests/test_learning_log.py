"""第3回 宿題: save_learning_log の承認まわりのテスト。TestModel を使うので API キーは要らない。"""

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import ask_approvals, log_agent


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path, monkeypatch):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LOG_DIR", log_dir)
    return log_dir


def _first_run():
    model = TestModel(call_tools=["save_learning_log"])
    with log_agent.override(model=model):
        result = log_agent.run_sync("今日の学びを記録して")
    return model, result


def test_save_is_offered_and_waits_for_approval(tmp_log_dir):
    model, result = _first_run()
    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert "save_learning_log" in offered
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not tmp_log_dir.exists()


def _resume(model, result, answer):
    approvals = ask_approvals(result.output, ask=lambda _prompt: answer)
    with log_agent.override(model=model):
        return log_agent.run_sync(message_history=result.all_messages(), deferred_tool_results=approvals)


def test_approved_save_writes_file(tmp_log_dir):
    model, result = _first_run()
    resumed = _resume(model, result, "y")
    assert isinstance(resumed.output, str)
    files = list(tmp_log_dir.glob("*.md"))
    assert len(files) == 1
    args = result.output.approvals[0].args_as_dict()
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith(f"# {args['title']}") and args["content"] in text


def test_denied_save_writes_nothing(tmp_log_dir):
    model, result = _first_run()
    resumed = _resume(model, result, "n")
    assert isinstance(resumed.output, str)
    assert not tmp_log_dir.exists()


def test_save_rejects_empty_input(tmp_log_dir):
    assert "両方を指定" in tools.save_learning_log(" ", "本文")
    assert not tmp_log_dir.exists()
