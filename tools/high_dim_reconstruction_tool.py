from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from sklearn.cross_decomposition import PLSRegression
except Exception:  # pragma: no cover - optional dependency fallback
    PLSRegression = None

try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    GradientBoostingRegressor = None

try:
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    MLPRegressor = None


@dataclass
class ReconstructionArtifact:
    image_path: str
    description: str
    success: bool
    error_message: Optional[str] = None


class HighDimReconstructionTool:
    def __init__(
        self,
        unary_bins: int = 48,
        pair_bins: int = 24,
        max_unary_views: int = 6,
        max_pair_views: int = 4,
        rank: int = 3,
        min_bin_count: int = 3,
        enable_surrogate_overview: bool = True,
        overview_grid_size: int = 42,
        overview_top_variables: int = 4,
        overview_top_pairs: int = 3,
        overview_max_points: int = 1200,
    ):
        self.unary_bins = max(16, int(unary_bins))
        self.pair_bins = max(12, int(pair_bins))
        self.max_unary_views = max(2, int(max_unary_views))
        self.max_pair_views = max(1, int(max_pair_views))
        self.rank = max(1, int(rank))
        self.min_bin_count = max(1, int(min_bin_count))
        self.enable_surrogate_overview = bool(enable_surrogate_overview)
        self.overview_grid_size = max(24, int(overview_grid_size))
        self.overview_top_variables = max(2, int(overview_top_variables))
        self.overview_top_pairs = max(1, int(overview_top_pairs))
        self.overview_max_points = max(128, int(overview_max_points))

    def ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_name: str,
        output_dir: str,
        prefix: str = "recon",
        structure_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        feature_names = [str(x) for x in (feature_names or []) if str(x).strip()]
        if len(feature_names) < 3 or target_name not in df.columns:
            return {
                "tokens": None,
                "trace": {"mode": "masked_low_rank_multiview_v1", "status": "skipped"},
                "image_paths": [],
                "descriptions": [],
            }

        self.ensure_dir(output_dir)
        structure_profile = dict(structure_profile or {})

        selected_variables = self._select_variables(df, feature_names, target_name, structure_profile)
        selected_pairs = self._select_pairs(df, feature_names, target_name, structure_profile, selected_variables)

        unary_views = [self._build_unary_view(df, name, target_name) for name in selected_variables]
        pair_views = [self._build_pair_view(df, a, b, target_name) for a, b in selected_pairs]

        unary_views = [x for x in unary_views if x is not None]
        pair_views = [x for x in pair_views if x is not None]

        unary_recon, unary_spectrum = self._reconstruct_unary_bank(unary_views)
        pair_recon, pair_spectrum = self._reconstruct_pair_bank(pair_views)

        unary_tokens = self._build_unary_tokens(unary_views, unary_recon)
        pair_tokens = self._build_pair_tokens(pair_views, pair_recon)

        overview_bundle = self._build_surrogate_overview(
            df=df,
            feature_names=feature_names,
            target_name=target_name,
            output_dir=output_dir,
            prefix=prefix,
            structure_profile=structure_profile,
            unary_views=unary_views,
            unary_recon=unary_recon,
            unary_tokens=unary_tokens,
            pair_views=pair_views,
            pair_recon=pair_recon,
            pair_tokens=pair_tokens,
        )

        artifacts = []
        artifacts.extend(self._export_unary_artifacts(unary_views, unary_recon, output_dir, prefix))
        artifacts.extend(self._export_pair_artifacts(pair_views, pair_recon, output_dir, prefix))
        artifacts.extend(list(overview_bundle.get("artifacts", []) or []))

        image_paths = [x.image_path for x in artifacts if x.success]
        descriptions = [x.description for x in artifacts if x.success]

        role_info = dict(structure_profile.get("variable_roles", {}) or {})
        dominant_unary = [
            item["variable"]
            for item in sorted(unary_tokens, key=lambda x: (-float(x["amplitude_score"]), -float(x["curvature_score"]), x["variable"]))[:4]
        ]
        dominant_pairs = [
            item["variables"]
            for item in sorted(pair_tokens, key=lambda x: (-float(x["interaction_score"]), -float(x["nonseparable_score"]), x["pair_name"]))[:4]
        ]
        periodic_vars = [
            item["variable"]
            for item in sorted(unary_tokens, key=lambda x: (-float(x["periodic_score"]), x["variable"]))
            if float(item["periodic_score"]) >= 0.16
        ][:3]
        symmetry_vars = [
            item["variable"]
            for item in sorted(unary_tokens, key=lambda x: (-float(x["mirror_symmetry_score"]), x["variable"]))
        ][:3]

        tokens = {
            "mode": "masked_low_rank_multiview_v1",
            "selected_variables": selected_variables,
            "selected_pairs": [list(x) for x in selected_pairs],
            "estimated_unary_rank": int(min(self.rank, max(1, len(unary_views)))),
            "estimated_pair_rank": int(min(self.rank, max(1, len(pair_views)))) if pair_views else 0,
            "unary_spectrum": [round(float(x), 6) for x in unary_spectrum[:4]],
            "pair_spectrum": [round(float(x), 6) for x in pair_spectrum[:4]],
            "dominant_unary_variables": dominant_unary,
            "dominant_pairs": dominant_pairs,
            "periodic_variables": periodic_vars,
            "symmetry_variables": symmetry_vars,
            "denominator_like_variables": list(role_info.get("denominator_core", []) or [])[:3],
            "active_variables": list(structure_profile.get("active_variables", []) or [])[:5],
            "unary_tokens": unary_tokens[: min(6, len(unary_tokens))],
            "pair_tokens": pair_tokens[: min(6, len(pair_tokens))],
        }
        overview_tokens = dict(overview_bundle.get("tokens", {}) or {})
        if overview_tokens:
            tokens["mode"] = "hybrid_multiview_surrogate_projection_v1"
            tokens.update(overview_tokens)
        trace = {
            "mode": "masked_low_rank_multiview_v1",
            "status": "ok",
            "num_unary_views": len(unary_views),
            "num_pair_views": len(pair_views),
            "selected_variables": selected_variables,
            "selected_pairs": [list(x) for x in selected_pairs],
            "artifact_count": len(image_paths),
        }
        overview_trace = dict(overview_bundle.get("trace", {}) or {})
        if overview_trace:
            trace["mode"] = "hybrid_multiview_surrogate_projection_v1"
            trace.update(overview_trace)
        return {
            "tokens": tokens,
            "trace": trace,
            "image_paths": image_paths,
            "descriptions": descriptions,
        }

    def _safe_float_array(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.asarray([], dtype=float)
        arr = arr.copy()
        finite = np.isfinite(arr)
        if int(finite.sum()) <= 0:
            return np.zeros_like(arr)
        mean = float(np.nanmean(arr[finite]))
        std = float(np.nanstd(arr[finite]))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        arr[finite] = (arr[finite] - mean) / std
        arr[~finite] = 0.0
        return np.clip(arr, -6.0, 6.0)

    def _rank_variable_score(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.size != y.size or x.size < 8:
            return 0.0
        x_std = float(np.nanstd(x))
        y_std = float(np.nanstd(y))
        if x_std < 1e-8 or y_std < 1e-8:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        corr = 0.0 if not np.isfinite(corr) else abs(float(corr))
        quad = np.corrcoef(x * x, y)[0, 1]
        quad = 0.0 if not np.isfinite(quad) else abs(float(quad))
        recip = np.corrcoef(1.0 / (1.0 + np.abs(x)), y)[0, 1]
        recip = 0.0 if not np.isfinite(recip) else abs(float(recip))
        return float(max(corr, quad, recip))

    def _rank_pair_score(self, x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
        if x1.size != y.size or x2.size != y.size or y.size < 8:
            return 0.0
        features = [
            x1 * x2,
            x1 / (1.0 + np.abs(x2)),
            x2 / (1.0 + np.abs(x1)),
            x1 * x1 - x2 * x2,
        ]
        best = 0.0
        for feat in features:
            std = float(np.nanstd(feat))
            y_std = float(np.nanstd(y))
            if std < 1e-8 or y_std < 1e-8:
                continue
            corr = np.corrcoef(feat, y)[0, 1]
            if np.isfinite(corr):
                best = max(best, abs(float(corr)))
        return float(best)

    def _select_variables(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_name: str,
        structure_profile: Dict[str, Any],
    ) -> List[str]:
        y = self._standardize(df[target_name].to_numpy(dtype=float))
        role_active = [str(x) for x in list(structure_profile.get("active_variables", []) or []) if str(x) in feature_names]
        scored = []
        for idx, name in enumerate(feature_names):
            x = self._standardize(df[name].to_numpy(dtype=float))
            score = self._rank_variable_score(x, y)
            if name in role_active:
                score += 0.4
            scored.append((score, -idx, name))
        scored.sort(reverse=True)
        ordered = [name for _, _, name in scored]
        for name in role_active[::-1]:
            if name in ordered:
                ordered.remove(name)
            ordered.insert(0, name)
        dedup = []
        seen = set()
        for name in ordered:
            if name not in seen:
                dedup.append(name)
                seen.add(name)
        return dedup[: self.max_unary_views]

    def _select_pairs(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_name: str,
        structure_profile: Dict[str, Any],
        selected_variables: List[str],
    ) -> List[Tuple[str, str]]:
        y = self._standardize(df[target_name].to_numpy(dtype=float))
        pair_candidates = []
        for item in list(structure_profile.get("top_pair_patterns", []) or [])[:6]:
            variables = [str(x) for x in list(item.get("variables", []) or []) if str(x) in feature_names]
            if len(variables) >= 2:
                pair_candidates.append((variables[0], variables[1], float(item.get("score", 0.0) or 0.0) + 0.5))
        for i in range(min(len(selected_variables), self.max_unary_views)):
            for j in range(i + 1, min(len(selected_variables), self.max_unary_views)):
                a = selected_variables[i]
                b = selected_variables[j]
                x1 = self._standardize(df[a].to_numpy(dtype=float))
                x2 = self._standardize(df[b].to_numpy(dtype=float))
                pair_candidates.append((a, b, self._rank_pair_score(x1, x2, y)))
        pair_candidates.sort(key=lambda x: (-float(x[2]), x[0], x[1]))
        out = []
        seen = set()
        for a, b, _ in pair_candidates:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, b))
            if len(out) >= self.max_pair_views:
                break
        return out

    def _build_unary_view(self, df: pd.DataFrame, x_name: str, y_name: str) -> Optional[Dict[str, Any]]:
        x = np.asarray(df[x_name], dtype=float).reshape(-1)
        y = np.asarray(df[y_name], dtype=float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < max(16, self.unary_bins // 2):
            return None
        x = self._standardize(x[mask])
        y = y[mask]
        edges = np.linspace(float(np.min(x)), float(np.max(x)), num=self.unary_bins + 1)
        if not np.all(np.isfinite(edges)) or float(edges[-1] - edges[0]) < 1e-8:
            return None
        bin_idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, self.unary_bins - 1)
        mean = np.zeros(self.unary_bins, dtype=float)
        count = np.zeros(self.unary_bins, dtype=float)
        for idx in range(self.unary_bins):
            local = y[bin_idx == idx]
            if local.size > 0:
                mean[idx] = float(np.mean(local))
                count[idx] = float(local.size)
        observed = count >= float(self.min_bin_count)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_norm = mean.copy()
        if np.any(observed):
            mu = float(np.mean(mean_norm[observed]))
            sigma = float(np.std(mean_norm[observed]))
            if not np.isfinite(sigma) or sigma < 1e-8:
                sigma = 1.0
            mean_norm = (mean_norm - mu) / sigma
        count_norm = count / max(1.0, float(np.max(count) or 1.0))
        vector = np.concatenate([mean_norm, count_norm], axis=0)
        mask_vec = np.concatenate([observed, np.ones_like(observed, dtype=bool)], axis=0)
        return {
            "type": "unary",
            "variable": x_name,
            "centers": centers.tolist(),
            "mean_curve": mean.tolist(),
            "count_curve": count.tolist(),
            "vector": vector.astype(float),
            "mask": mask_vec.astype(bool),
        }

    def _build_pair_view(self, df: pd.DataFrame, x1_name: str, x2_name: str, y_name: str) -> Optional[Dict[str, Any]]:
        x1 = np.asarray(df[x1_name], dtype=float).reshape(-1)
        x2 = np.asarray(df[x2_name], dtype=float).reshape(-1)
        y = np.asarray(df[y_name], dtype=float).reshape(-1)
        mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
        if int(mask.sum()) < max(48, self.pair_bins * 2):
            return None
        x1 = self._standardize(x1[mask])
        x2 = self._standardize(x2[mask])
        y = y[mask]
        edges1 = np.linspace(float(np.min(x1)), float(np.max(x1)), num=self.pair_bins + 1)
        edges2 = np.linspace(float(np.min(x2)), float(np.max(x2)), num=self.pair_bins + 1)
        if float(edges1[-1] - edges1[0]) < 1e-8 or float(edges2[-1] - edges2[0]) < 1e-8:
            return None
        idx1 = np.clip(np.digitize(x1, edges1[1:-1], right=False), 0, self.pair_bins - 1)
        idx2 = np.clip(np.digitize(x2, edges2[1:-1], right=False), 0, self.pair_bins - 1)
        mean = np.zeros((self.pair_bins, self.pair_bins), dtype=float)
        count = np.zeros((self.pair_bins, self.pair_bins), dtype=float)
        for r, c, val in zip(idx1, idx2, y):
            mean[r, c] += float(val)
            count[r, c] += 1.0
        observed = count >= float(self.min_bin_count)
        safe = np.where(count > 0, count, 1.0)
        mean = mean / safe
        vector = mean.reshape(-1).astype(float)
        mask_vec = observed.reshape(-1).astype(bool)
        return {
            "type": "pair",
            "variables": [x1_name, x2_name],
            "heatmap": mean.tolist(),
            "count_map": count.tolist(),
            "vector": vector,
            "mask": mask_vec,
        }

    def _masked_low_rank_reconstruct(
        self,
        matrix: np.ndarray,
        mask: np.ndarray,
        rank: int,
        n_iter: int = 6,
    ) -> Tuple[np.ndarray, List[float]]:
        if matrix.size == 0:
            return matrix, []
        filled = np.asarray(matrix, dtype=float).copy()
        observed = np.asarray(mask, dtype=bool)
        col_means = np.zeros(filled.shape[1], dtype=float)
        for j in range(filled.shape[1]):
            local = filled[observed[:, j], j]
            if local.size > 0:
                col_means[j] = float(np.mean(local))
        filled[~observed] = np.take(col_means, np.where(~observed)[1])
        final_energy = []
        for _ in range(max(2, int(n_iter))):
            u, s, vt = np.linalg.svd(filled, full_matrices=False)
            if s.size == 0:
                break
            r = max(1, min(int(rank), int(np.sum(s > 1e-8)) or 1))
            recon = (u[:, :r] * s[:r]) @ vt[:r, :]
            filled[~observed] = recon[~observed]
            filled[observed] = matrix[observed]
            denom = float(np.sum(s * s))
            if denom > 0:
                final_energy = ((s[:r] * s[:r]) / denom).astype(float).tolist()
        return filled, final_energy

    def _reconstruct_unary_bank(self, unary_views: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], List[float]]:
        if not unary_views:
            return {}, []
        matrix = np.stack([np.asarray(x["vector"], dtype=float) for x in unary_views], axis=0)
        mask = np.stack([np.asarray(x["mask"], dtype=bool) for x in unary_views], axis=0)
        recon, spectrum = self._masked_low_rank_reconstruct(matrix, mask, rank=min(self.rank, len(unary_views)))
        out = {}
        for idx, item in enumerate(unary_views):
            n = self.unary_bins
            out[item["variable"]] = recon[idx, :n].astype(float)
        return out, spectrum

    def _reconstruct_pair_bank(self, pair_views: List[Dict[str, Any]]) -> Tuple[Dict[str, np.ndarray], List[float]]:
        if not pair_views:
            return {}, []
        matrix = np.stack([np.asarray(x["vector"], dtype=float) for x in pair_views], axis=0)
        mask = np.stack([np.asarray(x["mask"], dtype=bool) for x in pair_views], axis=0)
        recon, spectrum = self._masked_low_rank_reconstruct(matrix, mask, rank=min(self.rank, len(pair_views)))
        out = {}
        for idx, item in enumerate(pair_views):
            key = "__".join(item["variables"])
            out[key] = recon[idx].reshape(self.pair_bins, self.pair_bins).astype(float)
        return out, spectrum

    def _build_unary_tokens(self, unary_views: List[Dict[str, Any]], unary_recon: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        out = []
        for item in unary_views:
            var = item["variable"]
            curve = np.asarray(unary_recon.get(var, np.asarray(item["mean_curve"], dtype=float)), dtype=float)
            if curve.size <= 2:
                continue
            amplitude = float(np.nanmax(curve) - np.nanmin(curve))
            second = np.diff(curve, n=2) if curve.size >= 3 else np.asarray([], dtype=float)
            curvature = float(np.mean(np.abs(second))) if second.size > 0 else 0.0
            first = np.diff(curve)
            sign_changes = 0.0
            if first.size > 1:
                signs = np.sign(first)
                signs = signs[np.abs(signs) > 1e-8]
                if signs.size > 1:
                    sign_changes = float(np.sum(signs[:-1] * signs[1:] < 0)) / float(max(1, signs.size - 1))
            monotonic = float(max(0.0, 1.0 - sign_changes))
            rev = curve[::-1]
            symmetry = 0.0
            if float(np.std(curve)) >= 1e-8 and float(np.std(rev)) >= 1e-8:
                corr = np.corrcoef(curve, rev)[0, 1]
                if np.isfinite(corr):
                    symmetry = abs(float(corr))
            fft_vals = np.fft.rfft(curve - float(np.mean(curve)))
            periodic = 0.0
            if fft_vals.size > 1:
                amps = np.abs(fft_vals[1:])
                denom = float(np.sum(amps))
                if denom > 1e-8:
                    periodic = float(np.max(amps) / denom)
            out.append({
                "variable": var,
                "amplitude_score": round(amplitude, 6),
                "curvature_score": round(curvature, 6),
                "monotonic_score": round(monotonic, 6),
                "mirror_symmetry_score": round(symmetry, 6),
                "periodic_score": round(periodic, 6),
            })
        out.sort(key=lambda x: (-float(x["amplitude_score"]), -float(x["curvature_score"]), x["variable"]))
        return out

    def _build_pair_tokens(self, pair_views: List[Dict[str, Any]], pair_recon: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        out = []
        for item in pair_views:
            variables = list(item["variables"])
            key = "__".join(variables)
            heatmap = np.asarray(pair_recon.get(key, np.asarray(item["heatmap"], dtype=float)), dtype=float)
            if heatmap.size <= 0:
                continue
            row = np.mean(heatmap, axis=1, keepdims=True)
            col = np.mean(heatmap, axis=0, keepdims=True)
            global_mean = float(np.mean(heatmap))
            additive = row + col - global_mean
            resid = heatmap - additive
            denom = float(np.linalg.norm(heatmap)) + 1e-8
            interaction = float(np.linalg.norm(resid) / denom)
            u, s, vt = np.linalg.svd(heatmap, full_matrices=False)
            nonseparable = 0.0
            if s.size > 1:
                denom_s = float(np.sum(s * s))
                if denom_s > 1e-8:
                    nonseparable = float(1.0 - (s[0] * s[0] / denom_s))
            swap_symmetry = 0.0
            flat = heatmap.reshape(-1)
            flat_t = heatmap.T.reshape(-1)
            if float(np.std(flat)) >= 1e-8 and float(np.std(flat_t)) >= 1e-8:
                corr = np.corrcoef(flat, flat_t)[0, 1]
                if np.isfinite(corr):
                    swap_symmetry = abs(float(corr))
            diag_mean = float(np.mean(np.diag(heatmap)))
            anti_diag_mean = float(np.mean(np.diag(np.fliplr(heatmap))))
            diag_contrast = abs(diag_mean - anti_diag_mean) / (float(np.std(heatmap)) + 1e-8)
            out.append({
                "pair_name": key,
                "variables": variables,
                "interaction_score": round(interaction, 6),
                "nonseparable_score": round(nonseparable, 6),
                "swap_symmetry_score": round(swap_symmetry, 6),
                "diagonal_contrast_score": round(float(diag_contrast), 6),
            })
        out.sort(key=lambda x: (-float(x["interaction_score"]), -float(x["nonseparable_score"]), x["pair_name"]))
        return out

    def _standardize_matrix(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        mean = np.nanmean(arr, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.nanstd(arr, axis=0)
        std = np.where(np.isfinite(std) & (std >= 1e-8), std, 1.0)
        out = (arr - mean) / std
        out[~np.isfinite(out)] = 0.0
        return out.astype(float), mean.astype(float), std.astype(float)

    def _safe_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true.size == 0 or y_true.size != y_pred.size:
            return 0.0
        denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        if denom <= 1e-12:
            return 1.0
        numer = float(np.sum((y_true - y_pred) ** 2))
        return float(max(-1.0, min(1.0, 1.0 - numer / denom)))

    def _prepare_xy_matrix(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        cols = [x for x in feature_names if x in df.columns]
        if target_name not in df.columns or not cols:
            return None, None
        safe = df[cols + [target_name]].apply(pd.to_numeric, errors="coerce")
        arr = safe.to_numpy(dtype=float)
        mask = np.all(np.isfinite(arr), axis=1)
        if int(np.sum(mask)) < max(48, len(cols) * 6):
            return None, None
        filtered = arr[mask]
        return filtered[:, : len(cols)].astype(float), filtered[:, len(cols)].astype(float)

    def _normalize_positive_scores(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr[~np.isfinite(arr)] = 0.0
        arr = np.maximum(arr, 0.0)
        total = float(np.sum(arr))
        if total <= 1e-12:
            if arr.size <= 0:
                return arr
            return np.full(arr.shape, 1.0 / float(arr.size), dtype=float)
        return arr / total

    def _compute_ablation_importance(self, predict_fn, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] <= 0:
            return np.asarray([], dtype=float)
        if x.shape[0] > 256:
            idx = np.linspace(0, x.shape[0] - 1, num=256, dtype=int)
            x_eval = x[idx]
        else:
            x_eval = x
        baseline = np.asarray(predict_fn(x_eval), dtype=float).reshape(-1)
        out = np.zeros(x_eval.shape[1], dtype=float)
        for j in range(x_eval.shape[1]):
            altered = np.array(x_eval, copy=True)
            altered[:, j] = float(np.median(x_eval[:, j]))
            shifted = np.asarray(predict_fn(altered), dtype=float).reshape(-1)
            out[j] = float(np.mean(np.abs(shifted - baseline)))
        return self._normalize_positive_scores(out)

    def _fit_surrogate_bundle(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        x_std, x_mean, x_scale = self._standardize_matrix(x)
        y_mean = float(np.mean(y)) if y.size > 0 else 0.0
        y_scale = float(np.std(y)) if y.size > 0 else 1.0
        if not np.isfinite(y_scale) or y_scale < 1e-8:
            y_scale = 1.0
        y_std = (y - y_mean) / y_scale

        candidates = []

        if GradientBoostingRegressor is not None and x.shape[0] >= 32:
            try:
                gb = GradientBoostingRegressor(
                    n_estimators=180,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42,
                )
                gb.fit(x, y)
                gb_pred = np.asarray(gb.predict(x), dtype=float).reshape(-1)
                candidates.append({
                    "name": "gradient_boosting_regressor",
                    "predict_fn": lambda arr, model=gb: np.asarray(model.predict(np.asarray(arr, dtype=float)), dtype=float).reshape(-1),
                    "predictions": gb_pred,
                    "raw_importance": self._normalize_positive_scores(np.asarray(getattr(gb, "feature_importances_", np.zeros(x.shape[1])), dtype=float)),
                })
            except Exception:
                pass

        if MLPRegressor is not None and x.shape[0] >= 32:
            try:
                mlp = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                    learning_rate_init=0.01,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mlp.fit(x_std, y_std)
                mlp_pred = np.asarray(mlp.predict(x_std), dtype=float).reshape(-1) * y_scale + y_mean
                candidates.append({
                    "name": "mlp_regressor",
                    "predict_fn": lambda arr, model=mlp, mean=x_mean, scale=x_scale, ym=y_mean, ys=y_scale: (
                        np.asarray(model.predict((np.asarray(arr, dtype=float) - mean) / scale), dtype=float).reshape(-1) * ys + ym
                    ),
                    "predictions": mlp_pred,
                    "raw_importance": None,
                })
            except Exception:
                pass

        try:
            design = np.concatenate([np.ones((x_std.shape[0], 1), dtype=float), x_std], axis=1)
            coef, _, _, _ = np.linalg.lstsq(design, y_std, rcond=None)
            lin_pred = (design @ coef) * y_scale + y_mean
            lin_coef = np.abs(np.asarray(coef[1:], dtype=float))
            candidates.append({
                "name": "linear_fallback",
                "predict_fn": lambda arr, coef=coef, mean=x_mean, scale=x_scale, ym=y_mean, ys=y_scale: (
                    np.concatenate(
                        [np.ones((np.asarray(arr, dtype=float).shape[0], 1), dtype=float), (np.asarray(arr, dtype=float) - mean) / scale],
                        axis=1,
                    ) @ coef
                ) * ys + ym,
                "predictions": lin_pred,
                "raw_importance": self._normalize_positive_scores(lin_coef),
            })
        except Exception:
            pass

        if not candidates:
            constant_pred = np.full(y.shape, y_mean, dtype=float)
            candidates.append({
                "name": "constant_mean",
                "predict_fn": lambda arr, value=y_mean: np.full((np.asarray(arr, dtype=float).shape[0],), value, dtype=float),
                "predictions": constant_pred,
                "raw_importance": self._normalize_positive_scores(np.zeros(x.shape[1], dtype=float)),
            })

        best = None
        best_r2 = -1e9
        for item in candidates:
            preds = np.asarray(item["predictions"], dtype=float).reshape(-1)
            score = self._safe_r2(y, preds)
            item["train_r2"] = float(score)
            item["train_mae"] = float(np.mean(np.abs(preds - y))) if preds.size else None
            if score > best_r2:
                best_r2 = score
                best = item

        ablation = self._compute_ablation_importance(best["predict_fn"], x)
        raw_importance = best.get("raw_importance", None)
        if raw_importance is None or np.asarray(raw_importance, dtype=float).size != x.shape[1]:
            combined = ablation
        else:
            combined = 0.65 * self._normalize_positive_scores(raw_importance) + 0.35 * ablation
            combined = self._normalize_positive_scores(combined)

        records = []
        for idx, name in enumerate(feature_names):
            records.append({
                "variable": str(name),
                "importance": round(float(combined[idx]), 6),
            })
        records.sort(key=lambda item: (-float(item["importance"]), item["variable"]))

        return {
            "model_name": best["name"],
            "predict_fn": best["predict_fn"],
            "predictions": np.asarray(best["predictions"], dtype=float).reshape(-1),
            "train_r2": float(best.get("train_r2", 0.0) or 0.0),
            "train_mae": float(best.get("train_mae", 0.0) or 0.0),
            "importance_records": records,
        }

    def _build_projection_bundle(
        self,
        x: np.ndarray,
        y_signal: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        y_signal = np.asarray(y_signal, dtype=float).reshape(-1)
        x_std, _, _ = self._standardize_matrix(x)
        if y_signal.size != x_std.shape[0]:
            y_signal = np.zeros(x_std.shape[0], dtype=float)

        scores = None
        weight_matrix = None
        projection_name = "correlation_fallback"

        if PLSRegression is not None and x_std.shape[0] > 8:
            try:
                n_components = 2 if x_std.shape[1] >= 2 else 1
                pls = PLSRegression(n_components=n_components, scale=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pls.fit(x_std, y_signal.reshape(-1, 1))
                scores = np.asarray(pls.transform(x_std), dtype=float)
                weight_matrix = np.asarray(pls.x_weights_, dtype=float)
                projection_name = "pls_regression"
            except Exception:
                scores = None
                weight_matrix = None

        if scores is None or weight_matrix is None:
            corr = []
            for j in range(x_std.shape[1]):
                col = x_std[:, j]
                if float(np.std(col)) < 1e-8 or float(np.std(y_signal)) < 1e-8:
                    corr.append(0.0)
                    continue
                value = np.corrcoef(col, y_signal)[0, 1]
                corr.append(0.0 if not np.isfinite(value) else float(value))
            w1 = np.asarray(corr, dtype=float)
            if float(np.linalg.norm(w1)) < 1e-8:
                w1 = np.ones(x_std.shape[1], dtype=float)
            w1 = w1 / (float(np.linalg.norm(w1)) + 1e-8)
            z1 = x_std @ w1
            if x_std.shape[1] >= 2:
                resid = x_std - np.outer(z1, w1)
                _, _, vt = np.linalg.svd(resid, full_matrices=False)
                w2 = np.asarray(vt[0], dtype=float) if vt.size > 0 else np.zeros_like(w1)
                z2 = resid @ w2
            else:
                w2 = np.zeros_like(w1)
                z2 = np.zeros(x_std.shape[0], dtype=float)
            scores = np.column_stack([z1, z2]).astype(float)
            weight_matrix = np.column_stack([w1, w2]).astype(float)

        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape[1] < 2:
            scores = np.concatenate([scores, np.zeros((scores.shape[0], 1), dtype=float)], axis=1)
        if weight_matrix.ndim == 1:
            weight_matrix = weight_matrix.reshape(-1, 1)
        if weight_matrix.shape[1] < 2:
            weight_matrix = np.concatenate([weight_matrix, np.zeros((weight_matrix.shape[0], 1), dtype=float)], axis=1)

        projection_records = []
        for idx, name in enumerate(feature_names):
            z1_w = float(weight_matrix[idx, 0]) if idx < weight_matrix.shape[0] else 0.0
            z2_w = float(weight_matrix[idx, 1]) if idx < weight_matrix.shape[0] else 0.0
            projection_records.append({
                "variable": str(name),
                "z1_weight": round(z1_w, 6),
                "z2_weight": round(z2_w, 6),
                "projection_strength": round(max(abs(z1_w), abs(z2_w)), 6),
            })
        projection_records.sort(key=lambda item: (-float(item["projection_strength"]), item["variable"]))

        return {
            "projection_name": projection_name,
            "scores": np.asarray(scores[:, :2], dtype=float),
            "projection_records": projection_records,
        }

    def _build_smoothed_surface(self, scores: np.ndarray, signal: np.ndarray) -> Optional[Dict[str, Any]]:
        scores = np.asarray(scores, dtype=float)
        signal = np.asarray(signal, dtype=float).reshape(-1)
        if scores.ndim != 2 or scores.shape[0] < 16 or scores.shape[0] != signal.size:
            return None
        if scores.shape[1] < 2:
            scores = np.concatenate([scores, np.zeros((scores.shape[0], 1), dtype=float)], axis=1)

        if scores.shape[0] > self.overview_max_points:
            idx = np.linspace(0, scores.shape[0] - 1, num=self.overview_max_points, dtype=int)
            scores = scores[idx]
            signal = signal[idx]

        z1 = scores[:, 0]
        z2 = scores[:, 1]
        lo1, hi1 = np.percentile(z1, [2, 98])
        lo2, hi2 = np.percentile(z2, [2, 98])
        if not np.isfinite(hi1 - lo1) or float(hi1 - lo1) < 1e-8:
            lo1, hi1 = float(np.min(z1) - 1.0), float(np.max(z1) + 1.0)
        if not np.isfinite(hi2 - lo2) or float(hi2 - lo2) < 1e-8:
            lo2, hi2 = float(np.min(z2) - 1.0), float(np.max(z2) + 1.0)
        pad1 = max(1e-4, 0.08 * float(hi1 - lo1))
        pad2 = max(1e-4, 0.08 * float(hi2 - lo2))
        grid_z1 = np.linspace(float(lo1 - pad1), float(hi1 + pad1), num=self.overview_grid_size)
        grid_z2 = np.linspace(float(lo2 - pad2), float(hi2 + pad2), num=self.overview_grid_size)
        zz1, zz2 = np.meshgrid(grid_z1, grid_z2)

        sigma1 = max(float(np.std(z1)) * 0.55, float(grid_z1[-1] - grid_z1[0]) / 12.0, 1e-4)
        sigma2 = max(float(np.std(z2)) * 0.55, float(grid_z2[-1] - grid_z2[0]) / 12.0, 1e-4)

        flat1 = zz1.reshape(-1, 1)
        flat2 = zz2.reshape(-1, 1)
        dist2 = ((flat1 - z1.reshape(1, -1)) / sigma1) ** 2 + ((flat2 - z2.reshape(1, -1)) / sigma2) ** 2
        dist2 = np.clip(dist2, 0.0, 60.0)
        weights = np.exp(-0.5 * dist2)
        density = np.sum(weights, axis=1)
        surface = (weights @ signal.reshape(-1, 1)).reshape(-1) / np.maximum(density, 1e-8)
        surface = surface.reshape(zz1.shape)
        density = density.reshape(zz1.shape)

        finite_density = density[np.isfinite(density)]
        if finite_density.size > 0:
            threshold = float(np.quantile(finite_density, 0.18))
            surface = np.where(density >= threshold, surface, np.nan)

        return {
            "z1_grid": zz1,
            "z2_grid": zz2,
            "surface": surface,
            "density": density,
            "sample_scores": scores,
            "sample_signal": signal,
        }

    def _build_overview_summary(
        self,
        feature_records: List[Dict[str, Any]],
        unary_tokens: List[Dict[str, Any]],
        pair_tokens: List[Dict[str, Any]],
        structure_profile: Dict[str, Any],
        surrogate_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        top_variables = [str(item.get("variable", "")).strip() for item in feature_records[: self.overview_top_variables] if str(item.get("variable", "")).strip()]
        top_pairs = [list(item.get("variables", [])[:2]) for item in pair_tokens[: self.overview_top_pairs] if len(list(item.get("variables", []) or [])) >= 2]
        denominator_like = list(((structure_profile.get("variable_roles", {}) or {}).get("denominator_core", []) or [])[:4])
        periodic = [
            str(item.get("variable", "")).strip()
            for item in unary_tokens
            if float(item.get("periodic_score", 0.0) or 0.0) >= 0.16
        ][:3]

        summary_lines = []
        if top_variables:
            summary_lines.append("top variables in the proxy view: " + ", ".join(top_variables))
        if top_pairs:
            summary_lines.append(
                "strong pair interactions in the local views: " +
                ", ".join(f"{a}-{b}" for a, b in top_pairs[: self.overview_top_pairs])
            )
        if denominator_like:
            summary_lines.append("variables that look denominator-like: " + ", ".join(str(x) for x in denominator_like))
        if periodic:
            summary_lines.append("variables with periodic or modulated clues: " + ", ".join(periodic))
        summary_lines.append(
            f"proxy model fit on train: R2={float(surrogate_bundle.get('train_r2', 0.0) or 0.0):.3f}, "
            f"MAE={float(surrogate_bundle.get('train_mae', 0.0) or 0.0):.3g}"
        )
        summary_lines.append("z1 and z2 are supervised compressed coordinates built from all input variables.")

        return {
            "summary_lines": summary_lines[:6],
            "top_variables": top_variables,
            "top_pairs": top_pairs,
            "denominator_like_variables": denominator_like,
            "periodic_variables": periodic,
        }

    def _build_surrogate_overview(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        target_name: str,
        output_dir: str,
        prefix: str,
        structure_profile: Dict[str, Any],
        unary_views: List[Dict[str, Any]],
        unary_recon: Dict[str, np.ndarray],
        unary_tokens: List[Dict[str, Any]],
        pair_views: List[Dict[str, Any]],
        pair_recon: Dict[str, np.ndarray],
        pair_tokens: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.enable_surrogate_overview or len(feature_names) < 3:
            return {"tokens": {}, "trace": {"overview_status": "skipped"}, "artifacts": []}

        x, y = self._prepare_xy_matrix(df, feature_names, target_name)
        if x is None or y is None:
            return {
                "tokens": {},
                "trace": {"overview_status": "skipped", "overview_reason": "insufficient_finite_rows"},
                "artifacts": [],
            }

        surrogate_bundle = self._fit_surrogate_bundle(x, y, feature_names)
        projection_bundle = self._build_projection_bundle(x, surrogate_bundle["predictions"], feature_names)
        surface_bundle = self._build_smoothed_surface(projection_bundle["scores"], surrogate_bundle["predictions"])
        summary_bundle = self._build_overview_summary(
            feature_records=surrogate_bundle["importance_records"],
            unary_tokens=unary_tokens,
            pair_tokens=pair_tokens,
            structure_profile=structure_profile,
            surrogate_bundle=surrogate_bundle,
        )

        unary_map = {}
        for item in unary_views:
            var = str(item.get("variable", "")).strip()
            if not var:
                continue
            unary_map[var] = {
                "variable": var,
                "centers": np.asarray(item.get("centers", []), dtype=float),
                "curve": np.asarray(unary_recon.get(var, np.asarray(item.get("mean_curve", []), dtype=float)), dtype=float),
            }
        pair_map = {}
        for item in pair_views:
            variables = list(item.get("variables", []) or [])
            if len(variables) < 2:
                continue
            key = "__".join(variables[:2])
            pair_map[key] = {
                "variables": variables[:2],
                "heatmap": np.asarray(pair_recon.get(key, np.asarray(item.get("heatmap", []), dtype=float)), dtype=float),
            }

        selected_unary = []
        seen_vars = set()
        for item in surrogate_bundle["importance_records"]:
            var = str(item.get("variable", "")).strip()
            if var in unary_map and var not in seen_vars:
                seen_vars.add(var)
                selected_unary.append(unary_map[var])
            if len(selected_unary) >= self.overview_top_variables:
                break
        for item in unary_tokens:
            var = str(item.get("variable", "")).strip()
            if var in unary_map and var not in seen_vars:
                seen_vars.add(var)
                selected_unary.append(unary_map[var])
            if len(selected_unary) >= self.overview_top_variables:
                break

        selected_pairs = []
        seen_pairs = set()
        for item in pair_tokens:
            key = str(item.get("pair_name", "")).strip()
            if key in pair_map and key not in seen_pairs:
                seen_pairs.add(key)
                selected_pairs.append(pair_map[key])
            if len(selected_pairs) >= self.overview_top_pairs:
                break

        artifacts = []
        if surface_bundle is not None:
            artifacts.append(
                self._export_global_surface_artifact(
                    surface_bundle=surface_bundle,
                    surrogate_bundle=surrogate_bundle,
                    output_dir=output_dir,
                    prefix=prefix,
                )
            )
            artifacts.append(
                self._export_overview_panel_artifact(
                    surface_bundle=surface_bundle,
                    surrogate_bundle=surrogate_bundle,
                    projection_bundle=projection_bundle,
                    feature_records=surrogate_bundle["importance_records"],
                    unary_panels=selected_unary,
                    pair_panels=selected_pairs,
                    summary_lines=summary_bundle["summary_lines"],
                    output_dir=output_dir,
                    prefix=prefix,
                )
            )

        panel_paths = [item.image_path for item in artifacts if item.success and "overview panel" in item.description]
        surface_paths = [item.image_path for item in artifacts if item.success and "global surrogate response surface" in item.description]

        return {
            "tokens": {
                "surrogate_model": surrogate_bundle["model_name"],
                "projection_model": projection_bundle["projection_name"],
                "surrogate_train_r2": round(float(surrogate_bundle.get("train_r2", 0.0) or 0.0), 6),
                "surrogate_train_mae": round(float(surrogate_bundle.get("train_mae", 0.0) or 0.0), 6),
                "top_variable_importance": surrogate_bundle["importance_records"][:6],
                "projection_weights": projection_bundle["projection_records"][:6],
                "overview_summary_lines": summary_bundle["summary_lines"],
                "strong_interaction_pairs": summary_bundle["top_pairs"],
                "overview_panel_paths": panel_paths,
                "global_surface_paths": surface_paths,
            },
            "trace": {
                "overview_status": "ok" if surface_bundle is not None else "skipped",
                "surrogate_model": surrogate_bundle["model_name"],
                "projection_model": projection_bundle["projection_name"],
                "surrogate_train_r2": round(float(surrogate_bundle.get("train_r2", 0.0) or 0.0), 6),
                "overview_num_variables": len(selected_unary),
                "overview_num_pairs": len(selected_pairs),
                "overview_panel_count": len(panel_paths),
                "global_surface_count": len(surface_paths),
            },
            "artifacts": [item for item in artifacts if item is not None],
        }

    def _export_global_surface_artifact(
        self,
        surface_bundle: Dict[str, Any],
        surrogate_bundle: Dict[str, Any],
        output_dir: str,
        prefix: str,
    ) -> ReconstructionArtifact:
        save_path = os.path.join(output_dir, f"{prefix}_global_surrogate_surface.png")
        desc = "global surrogate response surface in supervised z1-z2 space"
        try:
            zz1 = np.asarray(surface_bundle.get("z1_grid"), dtype=float)
            zz2 = np.asarray(surface_bundle.get("z2_grid"), dtype=float)
            surface = np.asarray(surface_bundle.get("surface"), dtype=float)
            samples = np.asarray(surface_bundle.get("sample_scores"), dtype=float)
            signal = np.asarray(surface_bundle.get("sample_signal"), dtype=float).reshape(-1)

            fig = plt.figure(figsize=(9.2, 7.0))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(zz1, zz2, np.nan_to_num(surface, nan=np.nanmean(signal) if signal.size else 0.0), cmap="viridis", linewidth=0.0, antialiased=True, alpha=0.92)
            if samples.size > 0 and signal.size == samples.shape[0]:
                keep = min(180, samples.shape[0])
                idx = np.linspace(0, samples.shape[0] - 1, num=keep, dtype=int)
                ax.scatter(samples[idx, 0], samples[idx, 1], signal[idx], c=signal[idx], cmap="viridis", s=9, alpha=0.35)
            ax.set_xlabel("z1")
            ax.set_ylabel("z2")
            ax.set_zlabel("proxy y")
            ax.set_title(
                "global surrogate response surface\n"
                f"proxy={surrogate_bundle.get('model_name', 'unknown')}  R2={float(surrogate_bundle.get('train_r2', 0.0) or 0.0):.3f}"
            )
            fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.08, label="proxy response")
            fig.tight_layout()
            fig.savefig(save_path, dpi=170)
            plt.close(fig)
            return ReconstructionArtifact(save_path, desc, True, None)
        except Exception as e:
            plt.close("all")
            return ReconstructionArtifact(save_path, desc, False, str(e))

    def _export_overview_panel_artifact(
        self,
        surface_bundle: Dict[str, Any],
        surrogate_bundle: Dict[str, Any],
        projection_bundle: Dict[str, Any],
        feature_records: List[Dict[str, Any]],
        unary_panels: List[Dict[str, Any]],
        pair_panels: List[Dict[str, Any]],
        summary_lines: List[str],
        output_dir: str,
        prefix: str,
    ) -> ReconstructionArtifact:
        save_path = os.path.join(output_dir, f"{prefix}_overview_panel.png")
        desc = "overview panel with global surface, feature importance, local curves, pair heatmaps, and summary"
        try:
            zz1 = np.asarray(surface_bundle.get("z1_grid"), dtype=float)
            zz2 = np.asarray(surface_bundle.get("z2_grid"), dtype=float)
            surface = np.asarray(surface_bundle.get("surface"), dtype=float)
            samples = np.asarray(surface_bundle.get("sample_scores"), dtype=float)

            fig = plt.figure(figsize=(18.0, 11.5))
            grid = fig.add_gridspec(3, 4, height_ratios=[1.25, 1.0, 1.0])

            ax_import = fig.add_subplot(grid[0, 0])
            top_importance = list(feature_records[:6])
            names = [str(item.get("variable", "")) for item in top_importance][::-1]
            values = [float(item.get("importance", 0.0) or 0.0) for item in top_importance][::-1]
            ax_import.barh(names, values, color="#4C72B0")
            ax_import.set_title("proxy feature importance")
            ax_import.set_xlabel("importance")
            ax_import.grid(alpha=0.15, axis="x")

            ax_surface = fig.add_subplot(grid[0, 1:4])
            contour = ax_surface.contourf(zz1, zz2, np.nan_to_num(surface, nan=np.nanmean(surface[np.isfinite(surface)]) if np.any(np.isfinite(surface)) else 0.0), levels=18, cmap="viridis")
            if samples.size > 0:
                keep = min(220, samples.shape[0])
                idx = np.linspace(0, samples.shape[0] - 1, num=keep, dtype=int)
                ax_surface.scatter(samples[idx, 0], samples[idx, 1], s=8, c="white", alpha=0.18, edgecolors="none")
            ax_surface.set_xlabel("z1")
            ax_surface.set_ylabel("z2")
            ax_surface.set_title(
                "supervised global view of all x\n"
                f"proxy={surrogate_bundle.get('model_name', 'unknown')}  projection={projection_bundle.get('projection_name', 'unknown')}  "
                f"R2={float(surrogate_bundle.get('train_r2', 0.0) or 0.0):.3f}"
            )
            fig.colorbar(contour, ax=ax_surface, shrink=0.86, pad=0.02, label="proxy response")

            unary_axes = [fig.add_subplot(grid[1, idx]) for idx in range(4)]
            for ax, panel in zip(unary_axes, unary_panels[:4]):
                centers = np.asarray(panel.get("centers", []), dtype=float)
                curve = np.asarray(panel.get("curve", []), dtype=float)
                var = str(panel.get("variable", "")).strip()
                ax.plot(centers, curve, color="#1f77b4", linewidth=2.0)
                ax.set_title(f"y vs {var}")
                ax.set_xlabel(var)
                ax.set_ylabel("response")
                ax.grid(alpha=0.18)
            for ax in unary_axes[len(unary_panels[:4]):]:
                ax.axis("off")

            pair_axes = [fig.add_subplot(grid[2, idx]) for idx in range(3)]
            for ax, panel in zip(pair_axes, pair_panels[:3]):
                variables = list(panel.get("variables", []) or [])
                heatmap = np.asarray(panel.get("heatmap", []), dtype=float)
                if heatmap.ndim != 2 or len(variables) < 2:
                    ax.axis("off")
                    continue
                im = ax.imshow(heatmap.T, origin="lower", aspect="auto", cmap="magma")
                ax.set_title(f"{variables[0]} x {variables[1]}")
                ax.set_xlabel(variables[0])
                ax.set_ylabel(variables[1])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for ax in pair_axes[len(pair_panels[:3]):]:
                ax.axis("off")

            ax_text = fig.add_subplot(grid[2, 3])
            ax_text.axis("off")
            text = "\n".join(f"- {line}" for line in list(summary_lines or [])[:6])
            ax_text.text(
                0.0,
                1.0,
                "overview summary\n\n" + text,
                va="top",
                ha="left",
                fontsize=11,
                linespacing=1.5,
                family="monospace",
            )

            fig.tight_layout()
            fig.savefig(save_path, dpi=170)
            plt.close(fig)
            return ReconstructionArtifact(save_path, desc, True, None)
        except Exception as e:
            plt.close("all")
            return ReconstructionArtifact(save_path, desc, False, str(e))

    def _export_unary_artifacts(
        self,
        unary_views: List[Dict[str, Any]],
        unary_recon: Dict[str, np.ndarray],
        output_dir: str,
        prefix: str,
    ) -> List[ReconstructionArtifact]:
        artifacts = []
        for item in unary_views[: self.max_unary_views]:
            var = item["variable"]
            centers = np.asarray(item["centers"], dtype=float)
            curve = np.asarray(unary_recon.get(var, np.asarray(item["mean_curve"], dtype=float)), dtype=float)
            save_path = os.path.join(output_dir, f"{prefix}_recon_unary_{var}.png")
            desc = f"reconstructed unary response field of y versus {var}"
            try:
                plt.figure(figsize=(6, 4))
                plt.plot(centers, curve, linewidth=2.0)
                plt.xlabel(var)
                plt.ylabel("reconstructed response")
                plt.title(f"reconstructed response profile: {var}")
                plt.tight_layout()
                plt.savefig(save_path, dpi=160)
                plt.close()
                artifacts.append(ReconstructionArtifact(save_path, desc, True, None))
            except Exception as e:
                plt.close()
                artifacts.append(ReconstructionArtifact(save_path, desc, False, str(e)))
        return artifacts

    def _export_pair_artifacts(
        self,
        pair_views: List[Dict[str, Any]],
        pair_recon: Dict[str, np.ndarray],
        output_dir: str,
        prefix: str,
    ) -> List[ReconstructionArtifact]:
        artifacts = []
        for item in pair_views[: self.max_pair_views]:
            variables = list(item["variables"])
            key = "__".join(variables)
            heatmap = np.asarray(pair_recon.get(key, np.asarray(item["heatmap"], dtype=float)), dtype=float)
            save_path = os.path.join(output_dir, f"{prefix}_recon_pair_{variables[0]}_{variables[1]}.png")
            desc = f"reconstructed pair response field on {variables[0]}-{variables[1]} plane"
            try:
                plt.figure(figsize=(5.4, 4.8))
                plt.imshow(heatmap.T, origin="lower", aspect="auto", cmap="viridis")
                plt.xlabel(variables[0])
                plt.ylabel(variables[1])
                plt.title(f"reconstructed pair field: {variables[0]}-{variables[1]}")
                plt.colorbar(label="reconstructed response")
                plt.tight_layout()
                plt.savefig(save_path, dpi=160)
                plt.close()
                artifacts.append(ReconstructionArtifact(save_path, desc, True, None))
            except Exception as e:
                plt.close()
                artifacts.append(ReconstructionArtifact(save_path, desc, False, str(e)))
        return artifacts
