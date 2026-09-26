'''
The interactive epoch table must show the unweighted content and style reconstruction
losses separately (rec_c, rec_s), not only their w_s-weighted sum.
'''
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "cardiac_motion")
from utils.lightning_helpers import EpochMetricsTableCallback


def _fake_trainer(epoch, metrics):
    return SimpleNamespace(sanity_checking=False, current_epoch=epoch, callback_metrics=metrics, optimizers=[])


def test_table_shows_unweighted_rec_c_and_rec_s(capsys):
    from rich.console import Console
    callback = EpochMetricsTableCallback(console=Console(width=200))  # wide enough not to truncate headers
    metrics = {
        "training_loss": torch.tensor(2.0),
        "val_loss": torch.tensor(1.5),
        "val_recon_loss": torch.tensor(1.2),  # = rec_c + w_s * rec_s with w_s = 0.1
        "val_recon_loss_c": torch.tensor(0.2),
        "val_recon_loss_s": torch.tensor(10.0),
    }
    callback.on_validation_end(_fake_trainer(0, metrics), None)

    ep, train, val, rec_c, rec_s = callback.rows[-1][:5]
    assert (ep, train, val, rec_c, rec_s) == ("0", "2.0000", "1.5000", "0.2000", "10.0000")

    callback._print_table()
    out = capsys.readouterr().out
    assert "rec_c" in out and "rec_s" in out


def _run_epochs(callback, n_epochs):
    trainer = None
    callback.on_fit_start(trainer, None)
    for epoch in range(n_epochs):
        callback.on_train_epoch_start(trainer, None)
        metrics = {"val_loss": torch.tensor(1.0 / (epoch + 1)), "val_recon_loss_c": torch.tensor(0.1),
                   "val_recon_loss_s": torch.tensor(0.2)}
        callback.on_validation_end(_fake_trainer(epoch, metrics), None)
    callback.on_fit_end(trainer, None)


def test_table_redrawn_in_place_on_terminal():
    import io
    import logging
    from rich.console import Console

    buffer = io.StringIO()
    callback = EpochMetricsTableCallback(console=Console(file=buffer, force_terminal=True, width=120))

    handler = logging.StreamHandler(sys.__stderr__)
    logging.getLogger().addHandler(handler)
    try:
        callback.on_fit_start(None, None)
        assert callback._live is not None
        assert handler.stream is not sys.__stderr__  # log lines go through Live, above the table
        callback.on_fit_end(None, None)
        assert callback._live is None
        assert handler.stream is sys.__stderr__      # restored
    finally:
        logging.getLogger().removeHandler(handler)

    buffer.truncate(0)
    _run_epochs(callback, 3)
    out = buffer.getvalue()
    assert "\x1b[2K" in out  # previous table erased before redrawing
    assert callback._live is None


def test_table_printed_each_epoch_when_not_a_terminal():
    import io
    from rich.console import Console

    buffer = io.StringIO()
    callback = EpochMetricsTableCallback(console=Console(file=buffer, force_terminal=False, width=120))
    _run_epochs(callback, 3)
    out = buffer.getvalue()
    assert callback._live is None
    assert out.count("Epoch metrics") == 3
    assert "\x1b[" not in out


def test_live_stopped_on_exception():
    import io
    from rich.console import Console

    callback = EpochMetricsTableCallback(console=Console(file=io.StringIO(), force_terminal=True))
    callback.on_fit_start(None, None)
    callback.on_exception(None, None, RuntimeError("boom"))
    assert callback._live is None


def test_table_shows_effective_w_s():
    from types import SimpleNamespace as NS
    callback = EpochMetricsTableCallback()
    metrics = {"val_loss": torch.tensor(1.0), "w_s_effective": torch.tensor(0.1)}

    ramping_module = NS(_effective_w_s=lambda: 0.325)  # value used in this epoch's validation
    callback.on_validation_end(_fake_trainer(66, metrics), ramping_module)
    assert callback.rows[-1][7] == "0.325"

    callback.on_validation_end(_fake_trainer(67, metrics), None)  # no ramp info: logged metric
    assert callback.rows[-1][7] == "0.100"


def test_table_shows_thickness_nrmse_only_when_logged():
    callback = EpochMetricsTableCallback()
    callback.on_validation_end(_fake_trainer(0, {"val_loss": torch.tensor(1.0), "val_thickness_nrmse": torch.tensor(0.7312)}), None)
    callback.on_validation_end(_fake_trainer(1, {"val_loss": torch.tensor(1.0)}), None)  # run without --w_thickness
    thk_nrmse = 11  # ep, train, val, rec_c, rec_s, s_trans, s_shape, w_s, ratio_t, ratio_tp, vdev, thk_nrmse
    assert callback.rows[0][thk_nrmse] == "0.731"
    assert callback.rows[1][thk_nrmse] == ""


def test_table_shows_rec_s_translation_and_shape():
    callback = EpochMetricsTableCallback()
    metrics = {"val_loss": torch.tensor(1.0), "val_recon_loss_s": torch.tensor(12.0),
               "val_recon_loss_s_translation": torch.tensor(5.0), "val_recon_loss_s_shape": torch.tensor(7.0)}
    callback.on_validation_end(_fake_trainer(0, metrics), None)
    assert callback.rows[-1][4:7] == ("12.0000", "5.0000", "7.0000")


def test_default_console_on_a_file_is_wide_and_not_a_terminal(monkeypatch):
    # FORCE_COLOR makes rich treat any stream as a terminal; SLURM logs must still get a plain, wide table
    import io
    monkeypatch.setenv("FORCE_COLOR", "3")
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    callback = EpochMetricsTableCallback()
    callback.on_fit_start(None, None)
    assert callback._live is None
    console = callback._get_console()
    assert not console.is_terminal and console.width == EpochMetricsTableCallback.FILE_WIDTH

    metrics = {"training_loss": torch.tensor(28.1234), "val_loss": torch.tensor(33.5678), "val_recon_loss_c": torch.tensor(25.4321),
               "val_recon_loss_s": torch.tensor(37.8765), "val_recon_loss_s_translation": torch.tensor(10.1234),
               "val_recon_loss_s_shape": torch.tensor(27.7531), "val_rec_ratio_to_time_mean": torch.tensor(3.1234),
               "val_rec_ratio_to_time_mean_pooled": torch.tensor(1.6123), "val_mean_vertex_dev": torch.tensor(5.4321),
               "val_thickness_nrmse": torch.tensor(1.1234)}
    callback.on_validation_end(_fake_trainer(0, metrics), None)
    callback._print_table()
    out = sys.stdout.getvalue()
    assert "…" not in out and "\x1b[" not in out
    assert "28.1234" in out and "37.8765" in out
