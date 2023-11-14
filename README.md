This code provides an interface for FDBinary code developed by S. Ilijić and K. Pavlovski (http://sail.zpf.fer.hr/fdbinary/) for spectral disentangling. 
The interface simplifies data preparation for FDBinary (and currently is set for HERMES spectra), as well as adds extra features such as error estimation using jackknife. 

Python 3.8 is recommended.

# Step 1: Installing Poetry

Poetry is a dependency management and packaging tool in Python, which makes it easy to manage project dependencies (https://python-poetry.org).
  
On Linux:

Run the following command in the terminal to get the Poetry installer script and execute it:

    curl -sSL https://install.python-poetry.org | python3 -
  
On macOS:

Either do the same command as for Linux or, alternatively, if you have Homebrew installed, you can run:

    brew install poetry

# Step 2: Configuring the Shell

After installation, you'll need to configure your shell to use Poetry.
  
On Linux:

Add Poetry to your PATH by adding the following line to your profile (e.g., ~/.bashrc, ~/.zshrc, etc.):

    export PATH="$HOME/.poetry/bin:$PATH"
    source ~/.bashrc  # or the appropriate file for your shell
    
On macOS:

If you've used the curl method, follow the same steps as Linux.

If you've used Homebrew, the path is already configured.

# Step 3: Verifying Installation

    poetry --version

# Step 4: Setting Up the Project and Installing Dependencies

Download all files from github page, and navigate to the project directory where the pyproject.toml file is located. Poetry manages dependencies in an isolated environment.

    cd path/to/your/project
    poetry install

# Step 5: Verify version of executable execfd3/fd3

Since this code is just a wrapper and runs an external executable, this executable needs to be compiled for your system. By default there is macOS executable included, but there is also Linux executable available. Go to directory execfd3, copy the one for your system (either fd3_macOS or fd3_Linux) and rename it to "fd3".

If the executable doesn't work, you may need to recompile it from c source code. Contact nadya.serebriakova@kuleuven.be 

# Step 6: Running the Application

    poetry run python fd3_gui/fd3bin_gui.py
