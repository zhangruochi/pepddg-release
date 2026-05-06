"""Unit tests for v23 PAE token mapping.

Uses synthetic fixtures (checked into repo) — no NFS dependency.
Tests both 2-chain (1F47-like) and 3-chain (3EQY-like) cases.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from research.pepddg_v5.scripts.v23_pae_token_mapping import (
    BoltzChainInfo,
    TokenMapping,
    build_token_mapping,
    extract_pae_features,
    parse_manifest,
)

# ── Synthetic fixtures ──────────────────────────────────────────────


def _make_manifest(chains: list[dict], name: str = "test") -> str:
    """Create a temporary manifest.json and return its path."""
    manifest = {
        "records": [{
            "id": name,
            "structure": {"num_chains": len(chains)},
            "chains": chains,
            "interfaces": [],
            "inference_options": {"pocket_constraints": [], "contact_constraints": []},
            "templates": [],
        }]
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix=f"manifest_{name}_"
    )
    json.dump(manifest, tmp)
    tmp.flush()
    return tmp.name


def _two_chain_manifest() -> str:
    """1F47-like: receptor (144) + peptide (17) = 161 tokens."""
    return _make_manifest([
        {"chain_id": 0, "chain_name": "A", "num_residues": 144,
         "entity_id": 0, "mol_type": 0, "cluster_id": -1,
         "msa_id": "test_0", "valid": True},
        {"chain_id": 1, "chain_name": "B", "num_residues": 17,
         "entity_id": 1, "mol_type": 0, "cluster_id": -1,
         "msa_id": "test_1", "valid": True},
    ], name="1F47_WT")


def _three_chain_manifest() -> str:
    """3EQY-like: homodimer receptor (84+84) + peptide (12) = 180 tokens."""
    return _make_manifest([
        {"chain_id": 0, "chain_name": "A", "num_residues": 84,
         "entity_id": 0, "mol_type": 0, "cluster_id": -1,
         "msa_id": "test_0", "valid": True},
        {"chain_id": 1, "chain_name": "B", "num_residues": 84,
         "entity_id": 0, "mol_type": 0, "cluster_id": -1,
         "msa_id": "test_0", "valid": True},
        {"chain_id": 2, "chain_name": "C", "num_residues": 12,
         "entity_id": 1, "mol_type": 0, "cluster_id": -1,
         "msa_id": "test_1", "valid": True},
    ], name="3EQY_WT")


# ── Tests: parse_manifest ───────────────────────────────────────────


class TestParseManifest:
    def test_two_chain(self):
        path = _two_chain_manifest()
        chains = parse_manifest(path)
        assert len(chains) == 2
        assert chains[0].chain_name == "A"
        assert chains[0].num_residues == 144
        assert chains[0].token_start == 0
        assert chains[0].token_end == 144
        assert chains[1].chain_name == "B"
        assert chains[1].num_residues == 17
        assert chains[1].token_start == 144
        assert chains[1].token_end == 161

    def test_three_chain(self):
        path = _three_chain_manifest()
        chains = parse_manifest(path)
        assert len(chains) == 3
        assert chains[2].chain_name == "C"
        assert chains[2].token_start == 168
        assert chains[2].token_end == 180


# ── Tests: build_token_mapping ──────────────────────────────────────


class TestBuildTokenMapping:
    def test_two_chain_peptide_found(self):
        path = _two_chain_manifest()
        tm = build_token_mapping(path, peptide_length=17, target="1F47")
        assert tm.peptide_chain.chain_name == "B"
        assert tm.peptide_chain.token_start == 144
        assert len(tm.receptor_chains) == 1
        assert tm.receptor_chains[0].chain_name == "A"
        assert tm.total_tokens == 161

    def test_three_chain_peptide_found(self):
        path = _three_chain_manifest()
        tm = build_token_mapping(path, peptide_length=12, target="3EQY")
        assert tm.peptide_chain.chain_name == "C"
        assert tm.peptide_chain.token_start == 168
        assert len(tm.receptor_chains) == 2
        assert tm.total_tokens == 180

    def test_no_matching_chain_raises(self):
        path = _two_chain_manifest()
        with pytest.raises(ValueError, match="No chain with 99 residues"):
            build_token_mapping(path, peptide_length=99, target="BAD")

    def test_receptor_token_indices(self):
        path = _three_chain_manifest()
        tm = build_token_mapping(path, peptide_length=12, target="3EQY")
        rec_idx = tm.receptor_token_indices
        assert len(rec_idx) == 168  # 84 + 84
        assert rec_idx[0] == 0
        assert rec_idx[-1] == 167


# ── Tests: mutation_token_index ─────────────────────────────────────


class TestMutationTokenIndex:
    def test_1f47_resid_4(self):
        """DB4A on 1F47: chain B, resid 4 → token 147."""
        path = _two_chain_manifest()
        tm = build_token_mapping(path, peptide_length=17, target="1F47")
        assert tm.mutation_token_index(4) == 147

    def test_1f47_resid_1(self):
        path = _two_chain_manifest()
        tm = build_token_mapping(path, peptide_length=17, target="1F47")
        assert tm.mutation_token_index(1) == 144

    def test_3eqy_resid_5(self):
        """EB5A on 3EQY: resid 5 → token 172."""
        path = _three_chain_manifest()
        tm = build_token_mapping(path, peptide_length=12, target="3EQY")
        assert tm.mutation_token_index(5) == 172

    def test_3eqy_resid_1(self):
        path = _three_chain_manifest()
        tm = build_token_mapping(path, peptide_length=12, target="3EQY")
        assert tm.mutation_token_index(1) == 168

    def test_out_of_range_raises(self):
        path = _two_chain_manifest()
        tm = build_token_mapping(path, peptide_length=17, target="1F47")
        with pytest.raises(ValueError, match="out of range"):
            tm.mutation_token_index(0)
        with pytest.raises(ValueError, match="out of range"):
            tm.mutation_token_index(18)


# ── Tests: extract_pae_features ─────────────────────────────────────


class TestExtractPaeFeatures:
    def test_basic_feature_extraction(self):
        """Synthetic PAE: peptide residue 1 has known values to receptor."""
        path = _two_chain_manifest()
        tm = build_token_mapping(path, peptide_length=17, target="1F47")

        # Create synthetic PAE: 161x161
        pae = np.full((161, 161), 10.0, dtype=np.float32)
        # Set peptide res 1 (token 144) → receptor tokens 0-143
        # Make 5 low values and rest high
        pae[144, 0] = 0.5
        pae[144, 1] = 0.6
        pae[144, 2] = 0.7
        pae[144, 3] = 0.8
        pae[144, 4] = 0.9
        # Rest of receptor stays at 10.0

        features = extract_pae_features(pae, tm, resid=1)

        assert abs(features["pae_mut_to_rec_top5"] - 0.7) < 0.01  # mean(0.5,0.6,0.7,0.8,0.9)
        assert features["pae_mut_to_rec_mean"] > 5.0  # mostly 10s
        assert features["pae_mut_row_std"] > 0  # non-uniform

    def test_three_chain_feature_extraction(self):
        """3EQY-like: peptide res 5 (token 172) → 168 receptor tokens."""
        path = _three_chain_manifest()
        tm = build_token_mapping(path, peptide_length=12, target="3EQY")

        pae = np.full((180, 180), 8.0, dtype=np.float32)
        # Set top-5 lowest for token 172 → receptor
        for i in range(5):
            pae[172, i] = 1.0 + i * 0.1

        features = extract_pae_features(pae, tm, resid=5)

        expected_top5 = np.mean([1.0, 1.1, 1.2, 1.3, 1.4])
        assert abs(features["pae_mut_to_rec_top5"] - expected_top5) < 0.01
        assert "pae_mut_to_rec_mean" in features
        assert "pae_mut_row_std" in features
