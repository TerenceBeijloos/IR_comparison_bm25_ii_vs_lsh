# IR Group 18

## Setup Instructions

### Step 1: Create a Virtual Environment
Create a virtual environment using your preferred method. For example:
```bash
python -m venv .venv
```

### Step 2: Install Requirements
Activate your virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code

### Option 1: Inverted Index and BM25
To run the inverted index and BM25 implementation, execute:
```bash
python ./collect_data.py
```

### Option 2: PyTerrier Version
To run the PyTerrier-based implementation, execute:
```bash
python ./pyterrier_collect_data.py
```

### Problems
It is possible that the existing ii.pickle does not work for your system, in that case delete it and run the code again.

## Memory Profiling

### Enabling Profiling
Uncomment all `@profile` decorators in the Python file you want to profile.

### Running the Profiler
Run memory profiling with the following commands:

For `collect_data.py`:
```bash
mprof run --interval 1 --include-children collect_data.py
```

For `pyterrier_collect_data.py`:
```bash
mprof run --interval 1 --include-children pyterrier_collect_data.py
```

### Generating the Plot
After running the profiler, generate the memory usage plot with:
```bash
mprof plot
```
