#!/usr/bin/env python3
"""
Simple launcher script for the Document Chat Assistant Streamlit app.
This script ensures all dependencies are available and launches the app.
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'langchain',
        'chromadb',
        'langchain-ollama',
        'langchain-text-splitters',
        'pypdf'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    else:
        print("✅ All required packages are available!")
    
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        import ollama
        # Try to list models to check if Ollama is running
        ollama.list()
        print("✅ Ollama is running and accessible!")
        return True
    except Exception as e:
        print("⚠️  Warning: Ollama connection failed.")
        print("   Make sure Ollama is installed and running.")
        print("   You can still run the app, but chat functionality may not work.")
        return False

def check_data_directory():
    """Check if data directory exists and has PDF files"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"⚠️  Warning: '{data_dir}' directory not found.")
        print("   Create a 'data' directory and add PDF files to use the app.")
        return False
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"⚠️  Warning: No PDF files found in '{data_dir}' directory.")
        print("   Add PDF files to the data directory to use the chat functionality.")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF file(s) in data directory:")
    for pdf in pdf_files:
        print(f"   - {pdf}")
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Document Chat Assistant Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print()
    
    # Check Ollama
    check_ollama()
    
    print()
    
    # Check data directory
    check_data_directory()
    
    print()
    print("🎯 Starting Streamlit application...")
    print("📱 The app will open in your default web browser.")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application.")
    print("=" * 40)
    
    try:
        # Launch Streamlit app
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching application: {e}")
        print("Try running: streamlit run app.py")

if __name__ == "__main__":
    main()
