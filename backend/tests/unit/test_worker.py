from raglab.db.repository import JobCancelled
from raglab.worker import is_cancelled, leaves, root_message


def test_cancellation_is_found_inside_nested_groups() -> None:
    nested = ExceptionGroup("outer", [ExceptionGroup("inner", [JobCancelled(5)])])
    assert is_cancelled(nested)
    assert is_cancelled(JobCancelled(1))
    assert not is_cancelled(ExceptionGroup("g", [ValueError("x")]))


def test_root_message_unwraps_task_groups() -> None:
    group = ExceptionGroup("unhandled errors in a TaskGroup", [RuntimeError("Not logged in")])
    assert root_message(group) == "Not logged in"
    assert root_message(TimeoutError()) == "TimeoutError"
    assert len(leaves(ExceptionGroup("g", [ValueError(), KeyError()]))) == 2
