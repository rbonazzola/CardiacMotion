"""
Tests for config loading and CLI arg merging.

Run with:  python -m unittest tests/test_config.py -v
"""
import sys
import unittest
from easydict import EasyDict

sys.path.insert(0, "cardiac_motion")
from config.load_config import load_yaml_config
from config.cli_args import overwrite_config_items


from argparse import Namespace

def _make_ref_config():
    """Mimics the EasyDict structure produced by load_yaml_config."""
    return EasyDict({
        "network_architecture": EasyDict({"latent_dim_c": 8, "latent_dim_s": 8}),
        "optimizer": EasyDict({"parameters": EasyDict({"lr": 1e-4})}),
        "batch_size": 16,
    })

def _make_cli(latent_dim_c):
    """Mimics args.config — a Namespace with nested structure."""
    return Namespace(network_architecture=Namespace(latent_dim_c=latent_dim_c))


class TestOverwriteConfigItems(unittest.TestCase):

    def test_cli_value_overwrites_default(self):
        result = overwrite_config_items(_make_ref_config(), _make_cli(32))
        self.assertEqual(result.network_architecture.latent_dim_c, 32)

    def test_unspecified_keys_keep_defaults(self):
        result = overwrite_config_items(_make_ref_config(), _make_cli(32))
        self.assertEqual(result.network_architecture.latent_dim_s, 8)

    def test_empty_cli_returns_ref_unchanged(self):
        result = overwrite_config_items(_make_ref_config(), {})
        self.assertEqual(result.network_architecture.latent_dim_c, 8)
        self.assertEqual(result.batch_size, 16)

    def test_ref_config_is_not_mutated(self):
        ref = _make_ref_config()
        overwrite_config_items(ref, _make_cli(64))
        self.assertEqual(ref.network_architecture.latent_dim_c, 8)

    def test_second_call_on_ref_loses_intermediate_changes(self):
        """Regression: documents the old bug where calling overwrite_config_items
        a second time on ref_config discarded changes made to config in between."""
        ref = _make_ref_config()
        cli = _make_cli(64)

        config = overwrite_config_items(ref, cli)
        config.batch_size = 999          # change made between the two calls

        # Old buggy pattern: second call was on ref, not config
        config_bugged = overwrite_config_items(ref, cli)
        self.assertNotEqual(config_bugged.batch_size, 999,
                            "Second call on ref_config should lose intermediate changes")


if __name__ == "__main__":
    unittest.main()
