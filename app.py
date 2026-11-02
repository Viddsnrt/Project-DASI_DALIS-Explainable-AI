import os
import runpy

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "streamlit_app.py")
    runpy.run_path(app_path, run_name="__main__")
