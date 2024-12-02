import os
import subprocess 
import logging

app_file_path = 'C:\\Projects\\app.py'

def start_app():
    try:
        subprocess.run(["streamlit", "run", app_file_path], cwd=os.path.dirname(__file__)) 

    except Exception as e:
        error_message = "Помилка", f"Помилка запуску додатку: {e}"   
        logging.error(error_message)
        logging.info(error_message)

start_app()
