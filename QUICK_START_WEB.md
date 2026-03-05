# 🚀 Quick Start Guide - Explainable Medical AI Web Application

## ✅ You're All Set!

The FastAPI web application has been successfully built and configured. Follow these steps to use it:

## 📦 Step 1: Install Dependencies (If Not Already Done)

```bash
# Install backend dependencies
pip install -r backend/requirements.txt

# Or install individually
pip install fastapi uvicorn pandas numpy scikit-learn xgboost lightgbm joblib shap
```

## 🎯 Step 2: Ensure Models are Trained

Check if you have trained models in the `trained_models/` folder:

```bash
# List trained models
dir trained_models\*.pkl
```

If no models exist, train them:

```bash
python train_advanced_models.py
```

## 🌐 Step 3: Start the Web Application

### Option A: Using Python Script (Recommended)
```bash
python start_web_application.py
```

### Option B: Using Batch File (Windows)
```bash
start_web_app.bat
```

### Option C: Direct uvicorn Command
```bash
uvicorn backend.main:app --reload --port 8000
```

## 🎉 Step 4: Access the Application

Once the server starts, you'll see:
```
🚀 Starting Explainable Medical AI Web Application
======================================================================
📦 Initializing FastAPI server...
🔧 Loading ML models...

Once started, access the application at:
   🌐 Web App:  http://localhost:8000/app
   📚 API Docs: http://localhost:8000/docs
   🔍 ReDoc:    http://localhost:8000/redoc
======================================================================
```

Open your browser and navigate to: **http://localhost:8000/app**

## 🎨 Using the Web Interface

### 1. Disease Risk Prediction

1. **Enter Patient Data:**
   - Fill in demographics (age, gender)
   - Enter vital signs (heart rate, blood pressure, temperature, etc.)
   - Input lab values (WBC, hemoglobin, creatinine, glucose, etc.)

2. **Or Load Sample Patient:**
   - Click "Load Sample" button
   - Choose from 4 pre-configured patient profiles
   - Sample data will auto-fill the form

3. **Analyze:**
   - Click "Analyze Patient Risk"
   - View predictions for all 8 diseases
   - See explainable AI insights (feature importance)
   - Review risk scores and categories

### 2. What-If Analysis

1. **Set Baseline:**
   - Enter baseline patient parameters
   - Or copy from previous prediction

2. **Create Scenario:**
   - Select target disease
   - Choose parameter to modify
   - Enter new value

3. **Analyze Impact:**
   - See baseline vs. new risk comparison
   - View risk delta percentage
   - Get recommendations

### 3. API Access

Access the interactive API documentation at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 📊 Sample API Usage

### Predict Disease Risks

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 68,
      "gender": 1,
      "heart_rate": 115,
      "systolic_bp": 95,
      "diastolic_bp": 65,
      "temperature": 101.5,
      "respiratory_rate": 24,
      "wbc_count": 16.5,
      "hemoglobin": 10.5,
      "platelet_count": 150,
      "creatinine": 2.1,
      "bun": 40,
      "glucose": 180,
      "lactate": 3.2
    }
  }'
```

### What-If Analysis

```bash
curl -X POST "http://localhost:8000/api/whatif" \
  -H "Content-Type: application/json" \
  -d '{
    "baseline_features": { ... },
    "modified_features": {"glucose": 120},
    "disease": "diabetes"
  }'
```

## 🔍 Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info & available endpoints |
| `/health` | GET | Health check |
| `/app` | GET | Web application interface |
| `/docs` | GET | Interactive API docs (Swagger) |
| `/redoc` | GET | Alternative API docs |
| `/api/predict` | POST | Predict disease risks |
| `/api/whatif` | POST | What-if scenario analysis |
| `/api/models` | GET | List loaded models |
| `/api/feature-info` | GET | Feature descriptions |
| `/api/sample-patients` | GET | Sample patient data |

## 🎯 Disease Categories

The system predicts risks for:

1. **Sepsis** - Life-threatening infection response
2. **Kidney Failure** - Renal dysfunction
3. **Heart Disease** - Cardiovascular conditions
4. **Diabetes** - Blood glucose regulation
5. **Anemia** - Low hemoglobin
6. **Thalassemia** - Genetic blood disorder
7. **Thrombocytopenia** - Low platelet count
8. **Mortality** - Overall survival risk

## 🎨 Risk Categories

- 🟢 **LOW** (0-30%) - Minimal risk
- 🟡 **MODERATE** (30-50%) - Some risk present
- 🟠 **HIGH** (50-70%) - Significant risk
- 🔴 **CRITICAL** (70-100%) - Severe risk

## 🛑 Stopping the Application

- Press `Ctrl + C` in the terminal
- Or close the terminal window

## 🐛 Troubleshooting

### Models Not Loading
**Problem:** "Model not found" errors

**Solution:** 
```bash
python train_advanced_models.py
```

### Port Already in Use
**Problem:** "Address already in use"

**Solution:** Change port in `start_web_application.py`:
```python
uvicorn.run(..., port=8080)  # Use different port
```

### Frontend Not Loading
**Problem:** Blank page or 404

**Solution:** Ensure `backend/static/` contains:
- index.html
- styles.css
- app.js

### Import Errors
**Problem:** "ModuleNotFoundError"

**Solution:** Install missing packages:
```bash
pip install -r backend/requirements.txt
```

## 📚 Additional Help

- **Full Documentation:** See `README_WEB_APP.md`
- **Project Docs:** Check `Readme/` folder
- **API Schema:** http://localhost:8000/openapi.json

## ⚠️ Important Disclaimer

**This system is for RESEARCH and EDUCATIONAL purposes only.**

- ❌ NOT for clinical use
- ❌ NOT for medical diagnosis
- ❌ NOT HIPAA compliant as-is
- ✅ For learning and demonstration only

**Always consult qualified healthcare professionals for medical decisions.**

## 🎉 You're Ready!

Start the application and explore the features:

1. ✅ Load sample patients
2. ✅ Analyze disease risks
3. ✅ Explore AI explanations
4. ✅ Try what-if scenarios
5. ✅ Use the API programmatically

Happy exploring! 🚀

---

**Questions or Issues?**
- Check `README_WEB_APP.md` for detailed documentation
- Review API docs at http://localhost:8000/docs
- Check troubleshooting section above
