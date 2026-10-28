"""Export utilities for saving analysis results to JSON, CSV, and Markdown."""

import csv
import json
from typing import Any, Dict, List, Union


class ResultExporter:
    """Exports audio analysis results into various file formats."""

    @staticmethod
    def to_json(results: Dict[str, Any], output_path: str, indent: int = 2) -> None:
        """Save analysis results to a JSON file.

        Args:
            results: Dictionary containing analysis results
            output_path: Destination file path
            indent: JSON indentation level
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=indent, default=str)

    @staticmethod
    def to_csv(
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        output_path: str,
    ) -> None:
        """Save basic analysis summary to a CSV file.

        Args:
            results: Single result dict or list/dict of multiple file results
            output_path: Destination file path
        """
        rows = []

        if isinstance(results, list):
            items = [(f"track_{i}", r) for i, r in enumerate(results)]
        elif "basic_info" in results:
            items = [("single_track", results)]
        else:
            items = list(results.items())

        for file_name, data in items:
            if not isinstance(data, dict) or "error" in data:
                rows.append({
                    "file": file_name,
                    "bpm": 0.0,
                    "key": "Error",
                    "duration": 0.0,
                    "time_signature": "N/A",
                })
                continue

            basic = data.get("basic_info", {})
            rhythm = data.get("rhythm", {})
            rows.append({
                "file": file_name,
                "bpm": round(float(basic.get("bpm", 0.0)), 2),
                "key": basic.get("key", "Unknown"),
                "duration": round(float(basic.get("duration", 0.0)), 2),
                "time_signature": rhythm.get("time_signature", "4/4"),
            })

        fieldnames = ["file", "bpm", "key", "duration", "time_signature"]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def to_markdown(results: Dict[str, Any], output_path: str) -> None:
        """Save analysis results as a formatted Markdown report.

        Args:
            results: Dictionary containing analysis results
            output_path: Destination file path
        """
        basic = results.get("basic_info", {})
        rhythm = results.get("rhythm", {})
        structure = results.get("structure", {})

        lines = [
            "# Audio Analysis Report",
            "",
            "## Basic Information",
            f"- **BPM:** {basic.get('bpm', 0.0):.2f} (Confidence: {basic.get('bpm_confidence', 0.0):.1f}%)",
            f"- **Key:** {basic.get('key', 'Unknown')} (Confidence: {basic.get('key_confidence', 0.0):.1f}%)",
            f"- **Duration:** {basic.get('duration', 0.0):.2f} seconds",
        ]

        if rhythm:
            lines.extend([
                "",
                "## Rhythm Analysis",
                f"- **Time Signature:** {rhythm.get('time_signature', '4/4')}",
                f"- **Groove Type:** {rhythm.get('groove_type', 'straight')}",
                f"- **Syncopation Level:** {rhythm.get('syncopation_level', 0.0):.2f}",
                f"- **Rhythmic Complexity:** {rhythm.get('rhythmic_complexity', 0.0):.2f}",
            ])

        if structure:
            lines.extend([
                "",
                "## Structural Analysis",
                f"- **Form:** {structure.get('form', 'Unknown')}",
                f"- **Section Count:** {structure.get('section_count', 0)}",
                f"- **Structural Complexity:** {structure.get('structural_complexity', 0.0):.2f}",
            ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")