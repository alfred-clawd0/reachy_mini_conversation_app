"""Torque safety through the real run() cleanup and registered signal handler."""

import signal
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import main, moves, config, console, handler_factory
from reachy_mini_conversation_app.tools import core_tools


@pytest.fixture
def runtime(monkeypatch):
    """Replace hardware and serving boundaries while retaining run() control flow."""
    events = []
    handlers = {}
    delays = []
    robot = MagicMock()
    camera = MagicMock()
    movement = MagicMock()
    stream = MagicMock()
    robot.enable_motors.side_effect = lambda: events.append("enable")
    robot.wake_up.side_effect = lambda: events.append("wake")
    robot.disable_motors.side_effect = lambda: events.append("disable")
    robot.client.disconnect.side_effect = lambda: events.append("disconnect")
    robot.media.close.side_effect = lambda: events.append("media_close")
    movement.stop.side_effect = lambda **kwargs: events.append("movement_stop")
    camera.stop.side_effect = lambda: events.append("camera_stop")
    monkeypatch.setattr(main, "setup_logger", lambda debug: logging.getLogger("test.main_shutdown"))
    monkeypatch.setattr(main.time, "sleep", delays.append)
    monkeypatch.setattr(main.signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr(config, "config", SimpleNamespace(BACKEND_PROVIDER=config.LOCAL_BACKEND, MODEL_NAME="test"))
    monkeypatch.setattr(main, "initialize_camera_and_vision", lambda *args: (camera, None))
    monkeypatch.setattr(moves, "MovementManager", lambda **kwargs: movement)
    monkeypatch.setattr(core_tools, "initialize_tools", lambda **kwargs: None)
    monkeypatch.setattr(handler_factory, "build_conversation_handler", lambda *args, **kwargs: MagicMock())
    monkeypatch.setattr(main.gr, "Chatbot", MagicMock())
    monkeypatch.setattr(console, "LocalStream", lambda *args, **kwargs: stream)
    monkeypatch.setenv("AGENT_SPEECH_WOBBLE", "0")
    args = SimpleNamespace(
        debug=False, robot_host=None, robot_name=None, no_camera=False, head_tracker=None, gradio=False
    )
    stop_event = threading.Event()
    stop_event.set()  # the poll thread exits immediately rather than outliving the test

    def run():
        main.run(args, robot=robot, app_stop_event=stop_event)

    return SimpleNamespace(
        run=run,
        robot=robot,
        camera=camera,
        movement=movement,
        stream=stream,
        events=events,
        handlers=handlers,
        delays=delays,
    )


def _sleep_outcomes(runtime, failures, *, reenter_signal=False):
    attempts = []

    def sleep():
        runtime.events.append("sleep")
        attempts.append(None)
        if reenter_signal and len(attempts) == 1:
            runtime.handlers[signal.SIGTERM](signal.SIGTERM, None)
        if len(attempts) <= failures:
            raise TimeoutError("move still pending")
        runtime.events.append("sleep_success")

    runtime.robot.goto_sleep.side_effect = sleep


@pytest.mark.parametrize("failures", [0, 1, 2])
@pytest.mark.parametrize("exit_path", ["normal", "keyboard", "launch_error", "signal"])
def test_shutdown_requires_completed_sleep(runtime, caplog, failures, exit_path):
    """All exits share retry, torque safety, disconnect, and single-cleanup semantics."""
    _sleep_outcomes(runtime, failures, reenter_signal=exit_path == "signal")
    if exit_path == "keyboard":
        runtime.stream.launch.side_effect = KeyboardInterrupt
    elif exit_path == "launch_error":
        runtime.stream.launch.side_effect = RuntimeError("launch failed")
    elif exit_path == "signal":
        runtime.stream.launch.side_effect = lambda: runtime.handlers[signal.SIGTERM](signal.SIGTERM, None)

    if exit_path == "signal":
        with pytest.raises(SystemExit) as stopped:
            runtime.run()
        assert stopped.value.code == 128 + signal.SIGTERM
    elif exit_path == "launch_error":
        with pytest.raises(RuntimeError, match="launch failed"):
            runtime.run()
    else:
        runtime.run()

    assert runtime.robot.goto_sleep.call_count == min(failures + 1, 2)
    assert runtime.delays.count(0.5) == int(failures > 0)
    assert runtime.robot.disable_motors.call_count == int(failures < 2)
    if failures < 2:
        assert runtime.events.index("sleep_success") < runtime.events.index("disable")
        assert "leaving motors enabled" not in caplog.text
    else:
        assert caplog.text.count("sleep move did not complete; leaving motors enabled to avoid a head drop") == 1
    runtime.robot.client.disconnect.assert_called_once()
    runtime.camera.stop.assert_called_once()
    runtime.movement.stop.assert_called_once_with(skip_neutral=True)
    assert runtime.events.index("sleep") < runtime.events.index("disconnect")


@pytest.mark.parametrize("component", ["camera", "movement"])
def test_pre_robot_cleanup_failure_cannot_skip_sleep(runtime, caplog, component):
    """Log only a pre-cleanup failure's type and still complete robot shutdown."""
    _sleep_outcomes(runtime, 0)
    getattr(runtime, component).stop.side_effect = RuntimeError("private cleanup details")
    runtime.run()
    runtime.robot.goto_sleep.assert_called_once()
    runtime.robot.disable_motors.assert_called_once()
    runtime.robot.client.disconnect.assert_called_once()
    assert "RuntimeError" in caplog.text
    assert "private cleanup details" not in caplog.text


@pytest.mark.parametrize("failures", [0, 2])
def test_failed_wake_requires_sleep_before_disabling(runtime, failures):
    """A timed-out startup move must never trigger an unconditional torque cut."""
    runtime.robot.wake_up.side_effect = TimeoutError("wake still pending")
    _sleep_outcomes(runtime, failures)
    with pytest.raises(TimeoutError, match="wake still pending"):
        runtime.run()
    assert runtime.robot.enable_motors.call_count == 2
    assert runtime.robot.wake_up.call_count == 2
    assert runtime.delays.count(2.0) == 1
    assert runtime.robot.disable_motors.call_count == int(failures == 0)
    if failures == 0:
        assert runtime.events.index("sleep_success") < runtime.events.index("disable")
    runtime.movement.start.assert_not_called()
    runtime.stream.launch.assert_not_called()
    runtime.robot.client.disconnect.assert_called_once()


def test_handler_setup_failure_still_cleans_robot(runtime, monkeypatch):
    """Initialization failures before launch also pass through the shared finally."""
    _sleep_outcomes(runtime, 0)

    def fail(*args, **kwargs):
        raise RuntimeError("handler setup failed")

    monkeypatch.setattr(handler_factory, "build_conversation_handler", fail)
    with pytest.raises(RuntimeError, match="handler setup failed"):
        runtime.run()
    runtime.robot.goto_sleep.assert_called_once()
    runtime.robot.disable_motors.assert_called_once()
    runtime.robot.client.disconnect.assert_called_once()
