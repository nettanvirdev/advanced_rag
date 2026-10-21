import pytest

from advanced_rag import config, main


def test_unknown_command_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["advanced-rag"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    assert "build" in capsys.readouterr().out


def test_build_indexes_the_data_folder(monkeypatch, capsys, tmp_path):
    (tmp_path / "cat.txt").write_text("Cats sleep a lot.", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["advanced-rag", "build"])
    main()
    assert "1 chunks in the index, 1 changed" in capsys.readouterr().out
