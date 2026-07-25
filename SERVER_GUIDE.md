# Server Guide: Smart Agile Task Manager AI

This guide provides step-by-step instructions on how to run the FastAPI server on your local machine to interact with the machine learning models.

> [!NOTE]
> Because the trained machine learning models (`.pkl` files) are already saved in the `models/` directory, you **do not** need to train the AI again to run the API. 

## 1. Setup the Virtual Environment
Open your terminal inside the root project directory and create a new Python virtual environment:
```bash
python -m venv venv
```

## 2. Activate the Virtual Environment
**On Windows:**
```powershell
.\venv\Scripts\Activate.ps1
```
*(If you encounter execution policy errors, you may need to run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` as Administrator).*

**On Mac/Linux:**
```bash
source venv/bin/activate
```

## 3. Install Dependencies
Install all the required Python packages exactly as they were used during training:
```bash
pip install -r requirements.txt
```

## 4. Start the FastAPI Server
Navigate into the `src/` directory and start the Uvicorn server. 
*(Note: If `uvicorn` fails to launch on Windows due to a launcher path error, explicitly use the python executable as shown below).*

```powershell
cd src

# Standard launch
uvicorn main:app --reload

# Fallback launch (if the standard launch fails)
..\venv\Scripts\python.exe -m uvicorn main:app --reload
```

## 5. Test the API
Once the terminal says `Application startup complete`, open your web browser and navigate to the interactive Swagger UI:

👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

From this dashboard, you can:
1. Click the **POST `/api/predict`** endpoint.
2. Click **"Try it out"**.
3. Modify the JSON request body with a fake task title and description.
4. Click **"Execute"** to instantly receive the AI's estimation!

## How to Retrain the Models (Optional)
If you update the dataset with new tasks and want the AI to learn from them, you can retrain the models by running the training pipeline from the root directory:
```bash
python src/train.py
```
This will automatically parse the data, train new XGBoost models, evaluate them, and overwrite the `.pkl` files in the `models/` folder.
