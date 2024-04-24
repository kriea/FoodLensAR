Set Up Virtual Environment (Optional but Recommended): It's a good practice to use virtual environments to manage project dependencies. Open a terminal in VS Code by clicking on "Terminal" in the menu bar and selecting "New Terminal". Then create a virtual environment by running:

>python3 -m venv venv

This command creates a virtual environment named "venv" in your project folder.

Activate Virtual Environment: After creating the virtual environment, you need to activate it. On Windows, run:

>.\venv\Scripts\activate

On macOS and Linux, run:
>source venv/bin/activate

Install Dependencies: With the virtual environment activated, you can now install Python packages using pip. For example:

pip install package_name
pip install PyQt5  //QT
