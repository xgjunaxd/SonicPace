"""Parallel processing audio analyzer module."""

import concurrent.futures
import multiprocessing
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

from .music_analyzer import AudioAnalyzer


class SmartParallelAudioAnalyzer:
    """Analyzer that intelligently utilizes parallel processing for faster execution."""

    def __init__(self, auto_parallel: bool = True, max_workers: Optional[int] = None):
        """Initialize the parallel analyzer.
        
        Args:
            auto_parallel: Whether to automatically determine the best parallel strategy
            max_workers: Maximum number of worker processes (defaults to CPU count - 1)
        """
        self.auto_parallel = auto_parallel
        
        if max_workers is None:
            # Leave one core free for system stability if possible
            cpu_count = multiprocessing.cpu_count()
            self.max_workers = max(1, cpu_count - 1)
        else:
            self.max_workers = max(1, max_workers)
            
        self.performance_stats: Dict[str, Dict[str, float]] = {}
        self._analyzer_cache: Optional[AudioAnalyzer] = None

    def _get_analyzer(self) -> AudioAnalyzer:
        """Get or create an AudioAnalyzer instance."""
        if self._analyzer_cache is None:
            self._analyzer_cache = AudioAnalyzer()
        return self._analyzer_cache

    def analyze_file(
        self, 
        file_path: Union[str, List[str]], 
        detect_key: bool = True, 
        comprehensive: bool = False,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Analyze one or multiple audio files using parallel processing.
        
        Args:
            file_path: Path to a single audio file or a list of paths
            detect_key: Whether to detect musical key
            comprehensive: Whether to run all available analyzers
            progress_callback: Optional callback for progress updates
            
        Returns:
            Analysis results dictionary (or dict of dicts for multiple files)
        """
        start_time = time.time()
        
        if isinstance(file_path, list):
            result = self._analyze_multiple_files(
                file_path, detect_key, comprehensive, progress_callback, **kwargs
            )
        else:
            # For a single file, route to the standard analyzer execution
            result = self._analyze_single_file(
                file_path, detect_key, comprehensive, progress_callback, **kwargs
            )
            
        execution_time = time.time() - start_time
        self._update_stats("total_analysis", execution_time)
        
        return result
        
    def _analyze_single_file(
        self, 
        file_path: str, 
        detect_key: bool, 
        comprehensive: bool,
        progress_callback: Optional[Callable[[int, str], None]],
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze a single file."""
        if progress_callback:
            progress_callback(10, f"Loading {os.path.basename(file_path)}...")
            
        analyzer = self._get_analyzer()
        result = analyzer.analyze_file(file_path, detect_key, comprehensive, **kwargs)
        
        if progress_callback:
            progress_callback(100, "Analysis complete")
            
        return result

    def _analyze_multiple_files(
        self, 
        file_paths: List[str], 
        detect_key: bool, 
        comprehensive: bool,
        progress_callback: Optional[Callable[[int, str], None]],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple files in parallel using ProcessPoolExecutor."""
        results = {}
        total_files = len(file_paths)
        
        if total_files == 0:
            return results

        if progress_callback:
            progress_callback(0, f"Starting analysis of {total_files} files...")

        # Define a top-level friendly worker function mapping
        def worker(path: str) -> tuple[str, Dict[str, Any]]:
            try:
                # Fresh instance per process to avoid memory leaks/serialization issues
                local_analyzer = AudioAnalyzer()
                res = local_analyzer.analyze_file(path, detect_key, comprehensive, **kwargs)
                return path, res
            except Exception as e:
                return path, {"error": str(e)}

        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(worker, path): path for path in file_paths}
            
            for future in concurrent.futures.as_completed(future_to_path):
                path, res = future.result()
                results[path] = res
                completed += 1
                
                if progress_callback:
                    percent = int((completed / total_files) * 100)
                    progress_callback(percent, f"Processed {completed}/{total_files} files")

        return results

    def _update_stats(self, task_name: str, execution_time: float) -> None:
        """Update performance statistics."""
        if task_name not in self.performance_stats:
            self.performance_stats[task_name] = {
                "total_time": 0.0,
                "execution_count": 0,
                "avg_execution_time": 0.0
            }
            
        stats = self.performance_stats[task_name]
        stats["total_time"] += execution_time
        stats["execution_count"] += 1
        stats["avg_execution_time"] = stats["total_time"] / stats["execution_count"]

    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        return self.performance_stats

    def cleanup(self) -> None:
        """Free up resources."""
        self._analyzer_cache = None
