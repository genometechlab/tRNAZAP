import io
import logging

import pytest
from tqdm import tqdm

from trnazap.cli.logging_setup import setup_logging


@pytest.fixture
def cli_logging():
    root = logging.getLogger()
    saved = (root.handlers[:], root.level, logging.getLogger("trnazap").level)
    setup_logging()
    yield
    root.handlers[:] = saved[0]
    root.setLevel(saved[1])
    logging.getLogger("trnazap").setLevel(saved[2])
    logging.captureWarnings(False)


def test_levels(cli_logging, capsys):
    logging.getLogger("trnazap.something").info("ours")
    logging.getLogger("thirdparty").info("theirs")
    logging.getLogger("thirdparty").warning("their warning")
    err = capsys.readouterr().err
    assert "ours" in err and "their warning" in err
    assert "theirs" not in err


def test_log_lines_do_not_join_progress_bar(cli_logging, monkeypatch):
    # tqdm.write clears the bar, prints the line, and redraws the bar, so the log
    # line always starts on a fresh line instead of after the bar's text.
    out = io.StringIO()
    monkeypatch.setattr("sys.stderr", out)
    bar = tqdm(total=10, file=out, desc="Reading POD5", ncols=80)
    bar.update(3)
    logging.getLogger("trnazap.x").info("hello")
    bar.close()
    assert any(line.endswith("INFO: hello")
               for line in out.getvalue().replace("\r", "\n").split("\n"))
