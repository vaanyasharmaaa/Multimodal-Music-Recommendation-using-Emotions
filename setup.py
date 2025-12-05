"""
Setup script for multi-modal emotion detection system.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Successfully installed all requirements")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def setup_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/splits",
        "data/cache",
        "models/image",
        "models/text",
        "models/video", 
        "models/multimodal",
        "models/checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Created all necessary directories")

def main():
    """Main setup function."""
    print("Setting up Multi-Modal Emotion Detection System...")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Install requirements
    if install_requirements():
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Open any notebook in the 'notebooks/' directory")
        print("2. Run the cells to start training models")
        print("3. Check the 'config/' directory to modify parameters")
    else:
        print("\n✗ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()