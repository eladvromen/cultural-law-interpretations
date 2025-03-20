import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import yaml
from datetime import datetime
import sys

# Add pipeline directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Now use imports relative to pipeline directory
from extractors.ngram.ngram_determination_extraction import NgramDeterminationExtractor
from extractors.basic.basic_determination_extraction import BasicDeterminationExtractor
from extractors.sparse.sparse_explicit_extraction import SparseExplicitExtractor
from evaluation.evaluation import run_evaluation as eval_func

@dataclass
class ExperimentConfig:
    """Configuration for a determination extraction experiment."""
    model_name: str
    model_params: Dict[str, Any]
    experiment_name: Optional[str] = None
    description: Optional[str] = None

class DeterminationExperimentRunner:
    """Runner for determination extraction experiments."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the runner with base directory."""
        self.base_dir = base_dir or Path(__file__).parent.parent  # pipeline/
        self.models = self._get_available_models()
        self.experiment_history = []
        
        # Add paths for configs and experiments
        self.config_dir = self.base_dir / "configs"
        self.experiments_dir = self.base_dir / "experiments"
        
    def _get_available_models(self) -> Dict[str, str]:
        """Returns the list of available determination extraction models."""
        return {
            'ngram': 'NgramDeterminationExtractor',
            'basic': 'BasicDeterminationExtractor',
            'sparse': 'SparseExplicitExtractor'
        }
    
    def _load_model(self, model_name: str, **model_params) -> Any:
        """Load a model by name with given parameters."""
        if model_name == 'ngram':
            model = NgramDeterminationExtractor(**model_params)
        elif model_name == 'basic':
            model = BasicDeterminationExtractor(**model_params)
        elif model_name == 'sparse':
            model = SparseExplicitExtractor(**model_params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return model
    
    def _setup_paths(self, data_paths: Optional[Dict[str, Path]], 
                    output_dir: Optional[Path]) -> Dict[str, Path]:
        """Setup and validate all required paths."""
        # Get project root directory (2 levels up from pipeline)
        project_root = self.base_dir.parent.parent  # immigration_fine_tuning/
        data_dir = project_root / "data"
        
        paths = {
            'train': Path(data_paths['train']) if data_paths and 'train' in data_paths 
                    else data_dir / "merged" / "train_enriched.csv",
            'test': Path(data_paths['test']) if data_paths and 'test' in data_paths 
                    else data_dir / "merged" / "test_enriched.csv",
            'validation': Path(data_paths['validation']) if data_paths and 'validation' in data_paths 
                         else self.base_dir / "data" / "validation_set.csv",
            'output': Path(output_dir) if output_dir else self.base_dir / "results" / "experiments"
        }
        
        # Validate paths exist
        if not paths['train'].exists():
            raise FileNotFoundError(
                f"Training data not found at {paths['train']}\n"
                "Please ensure data files are in the correct location:\n"
                f"  {data_dir}/merged/train_enriched.csv\n"
                f"  {data_dir}/merged/test_enriched.csv"
            )
        if not paths['test'].exists():
            raise FileNotFoundError(f"Test data not found at {paths['test']}")
        
        # Create output directory
        paths['output'].mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def run_experiment(self, config: Union[ExperimentConfig, List[ExperimentConfig]], 
                      data_paths: Optional[Dict[str, Path]] = None,
                      output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run one or more experiments.
        
        Args:
            config: Single experiment config or list of configs
            data_paths: Optional dictionary with custom data paths
            output_dir: Optional custom output directory
        
        Returns:
            Dictionary with experiment results
        """
        # Convert single config to list
        configs = [config] if isinstance(config, ExperimentConfig) else config
        
        # Setup paths
        paths = self._setup_paths(data_paths, output_dir)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.experiments_dir / f"experiment_{timestamp}" if not output_dir else paths['output'] / f"experiment_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        # Run each experiment
        for exp_config in configs:
            exp_name = exp_config.experiment_name or f"{exp_config.model_name}_{timestamp}"
            print(f"\nRunning experiment: {exp_name}")
            if exp_config.description:
                print(f"Description: {exp_config.description}")
            
            # Run the model
            results = self._run_single_model(
                exp_config.model_name,
                paths['train'],
                paths['test'],
                paths['validation'],
                experiment_dir,
                exp_config.model_params
            )
            
            # Add experiment metadata
            results['experiment'] = {
                'name': exp_name,
                'description': exp_config.description,
                'timestamp': timestamp,
                'config': exp_config.__dict__
            }
            
            # Save individual results
            result_path = experiment_dir / f"{exp_name}_results.json"
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            all_results[exp_name] = results
            self.experiment_history.append(results)
        
        # Save combined results if multiple experiments
        if len(configs) > 1:
            self._save_combined_results(all_results, experiment_dir)
        
        return all_results
    
    def _run_single_model(self, model_name: str, train_path: Path, test_path: Path, 
                         validation_path: Path, output_dir: Path, 
                         model_params: Dict = None) -> Dict[str, Any]:
        """Run a single model experiment."""
        print(f"Running {model_name} model...")
        start_time = time.time()
        
        # Load the model
        model_params = model_params or {}
        model = self._load_model(model_name, **model_params)
        
        # Load training examples
        model.load_training_examples(str(train_path), str(test_path), use_chunking=True)
        
        # Run evaluation
        results, metrics, analysis_paths = self._run_evaluation(model, validation_path, output_dir)
        
        # Get performance stats
        perf_stats = model.get_performance_stats()
        
        # Prepare results
        output = {
            'model_name': model_name,
            'model_params': model_params,
            'metrics': metrics,
            'performance': perf_stats,
            'total_runtime': f"{time.time() - start_time:.2f} seconds",
            'case_results': [
                {
                    'case_id': r['case_id'],
                    'metrics': r['metrics'],
                    'extracted_count': r['metrics']['extracted_count'],
                    'expected_count': r['metrics']['expected_count']
                } for r in results
            ],
            'analysis_files': analysis_paths
        }
        
        # Print results
        print(f"\nPerformance Stats:")
        for k, v in perf_stats.items():
            print(f"  {k}: {v}")
        
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        return output
    
    def _run_evaluation(self, model, validation_file: Path, output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[str]]:
        """Run evaluation on the validation set."""
        try:
            print("\nRunning evaluation on validation set...")
            return eval_func(
                model, 
                validation_file=validation_file,
                output_dir=output_dir / "evaluation",  # Store evaluation results in a subdirectory
                analyze_failures=True,
                analyze_overextraction=True
            )
            
        except Exception as e:
            print(f"Error running evaluation: {e}")
            import traceback
            traceback.print_exc()
            return [], {}, []
    
    def _save_combined_results(self, all_results: Dict[str, Any], output_dir: Path):
        """Save combined results from multiple experiments."""
        combined_results = {
            'experiments': list(all_results.keys()),
            'metrics_by_experiment': {name: results['metrics'] 
                                    for name, results in all_results.items()},
            'performance_by_experiment': {name: results['performance'] 
                                        for name, results in all_results.items()},
            'total_runtime': sum(float(results['total_runtime'].split()[0]) 
                               for results in all_results.values())
        }
        
        # Calculate average metrics across experiments
        avg_metrics = {}
        for metric in ['precision', 'recall', 'f1', 'exact_match']:
            values = [results['metrics'].get(metric, 0) for results in all_results.values()]
            avg_metrics[f'avg_{metric}'] = sum(values) / len(values) if values else 0
        
        combined_results['average_metrics'] = avg_metrics
        
        # Save combined results
        combined_path = output_dir / "combined_results.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\nCombined results saved to {combined_path}")
        print("Average metrics across experiments:")
        for k, v in avg_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    def load_experiment_config(self, config_path: Union[str, Path]) -> List[ExperimentConfig]:
        """Load experiment configuration from YAML file."""
        # If config_path is just a filename, look in the configs directory
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
            
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        
        configs = []
        for exp in config_data['experiments']:
            configs.append(ExperimentConfig(
                model_name=exp['model'],
                model_params=exp.get('params', {}),
                experiment_name=exp.get('name'),
                description=exp.get('description')
            ))
        
        return configs

def main():
    """Main entry point for running determination extraction experiments."""
    # Initialize runner
    runner = DeterminationExperimentRunner()
    
    # Create config for sparse model experiment
    config = ExperimentConfig(
        model_name='sparse',
        model_params={},
        experiment_name='sparse_baseline',
        description='Testing sparse model extraction'
    )
    
    # Run the experiment
    results = runner.run_experiment(config)
    
if __name__ == "__main__":
    main()