import os
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_project_structure(training_project_name: str, app_project_name: str):
    """
    Create a complete project structure with both training pipeline and application.
    Args:
        training_project_name (str): Name of the training pipeline project
        app_project_name (str): Name of the application project
    """
    # Training pipeline structure
    training_files = [
        # Core ML components
        f'{training_project_name}/__init__.py',
        f'{training_project_name}/components/__init__.py',
        f'{training_project_name}/components/data_ingestion.py',
        f'{training_project_name}/components/data_validation.py',
        f'{training_project_name}/components/model_trainer.py',
        f'{training_project_name}/components/model_evaluation.py',
        
        # Constants and configs
        f'{training_project_name}/constants/__init__.py',
        f'{training_project_name}/constants/training_pipeline/__init__.py',
        f'{training_project_name}/constants/application.py',
        
        # Entity definitions
        f'{training_project_name}/entity/__init__.py',
        f'{training_project_name}/entity/config_entity.py',
        f'{training_project_name}/entity/artifacts_entity.py',
        
        # Exception handling and logging
        f'{training_project_name}/exception/__init__.py',
        f'{training_project_name}/logger/__init__.py',
        
        # Pipeline
        f'{training_project_name}/pipeline/__init__.py',
        f'{training_project_name}/pipeline/training_pipeline.py',
        
        # Utils
        f'{training_project_name}/utils/__init__.py',
        f'{training_project_name}/utils/main_utils.py',
    ]

    # Application structure
    application_files = [
        # Main application
        f'{app_project_name}/__init__.py',
        f'{app_project_name}/__main__.py',
        
        # Core functionality
        f'{app_project_name}/core/__init__.py',
        f'{app_project_name}/core/detector.py',
        f'{app_project_name}/core/tracker.py',
        f'{app_project_name}/core/counter.py',
        f'{app_project_name}/core/demographics.py',
        
        # Visualization
        f'{app_project_name}/visualization/__init__.py',
        f'{app_project_name}/visualization/annotator.py',
        
        # Configuration
        f'{app_project_name}/config/__init__.py',
        f'{app_project_name}/config/settings.py',
        
        # Utils
        f'{app_project_name}/utils/__init__.py',
        f'{app_project_name}/utils/cli.py',
        f'{app_project_name}/utils/video.py',
        
        # Models
        f'{app_project_name}/models/__init__.py',
    ]

    # Common project files
    common_files = [
        # Application entry points
        'app.py',
        'main.py',
        'run.py',
        'train.py',
        
        # Project configuration
        'setup.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        'LICENSE',
        
        # Docker files
        'Dockerfile',
        
        # CI/CD
        '.github/workflows/ci-cd.yml',
        
        # Data and model directories
        'data/.gitkeep',
        'weights/.gitkeep',
        'artifacts/.gitkeep',
        'output/.gitkeep',
        'notes/.gitkeep',
        
        # Additional files
        'zone.py',
    ]

    # Combine all files
    all_files = training_files + application_files + common_files

    for file_path in all_files:
        filepath = Path(file_path)
        filedir, filename = os.path.split(filepath)

        if filedir:
            os.makedirs(filedir, exist_ok=True)
            logging.info(f'Created directory: {filedir}')

        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                pass
            logging.info(f'Created file: {filepath}')
        else:
            logging.info(f'File already exists: {filepath}')

def main():
    parser = argparse.ArgumentParser(description='Create complete project structure')
    parser.add_argument('--training', type=str, required=True, help='Name of the training pipeline project')
    parser.add_argument('--app', type=str, required=True, help='Name of the application project')
    args = parser.parse_args()

    print(f"\nCreating project structure:")
    print(f"Training Pipeline: {args.training}")
    print(f"Application: {args.app}")
    
    create_project_structure(args.training, args.app)
    
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Update setup.py with project dependencies")
    print("2. Configure training pipeline in training_pipeline.py")
    print("3. Implement core application components")
    print("4. Set up Docker and CI/CD if needed")

if __name__ == "__main__":
    main()