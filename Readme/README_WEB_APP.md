# Explainable Medical AI - Web Application Guide

## 🌐 Web Application Overview

A modern, user-friendly web interface for the Explainable Medical AI system, built with FastAPI (backend) and vanilla HTML/CSS/JavaScript (frontend).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install backend-specific dependencies
pip install -r backend/requirements.txt
```

### 2. Ensure Models are Trained

The web application requires trained models in the `trained_models/` directory:

```bash
# If models don't exist, train them first
python train_advanced_models.py
```

### 3. Start the Web Application

```bash
python start_web_application.py
```

### 4. Access the Application

Open your web browser and navigate to:
- **Web App:** http://localhost:8000/app
- **API Documentation:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

## 📋 Features

### 1. Disease Risk Prediction
- Enter patient vitals and lab values
- Get AI-powered risk predictions for multiple diseases
- View explainable AI insights (SHAP-based feature importance)
- Load sample patient profiles for testing

**Supported Diseases:**
- Sepsis
- Kidney Failure
- Heart Disease
- Diabetes
- Anemia
- Thalassemia
- Thrombocytopenia
- Mortality Risk

### 2. What-If Scenario Analysis
- Explore how changes in patient parameters affect disease risk
- Compare baseline vs. modified scenarios
- Get recommendations based on risk changes
- Interactive parameter modification

### 3. API Access
Full RESTful API with:
- Interactive Swagger documentation
- JSON request/response format
- CORS support for external integrations

## 🏗️ Architecture

```
Explainable-AI-system/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── static/              # Frontend files
│       ├── index.html       # Main HTML
│       ├── styles.css       # Styling
│       └── app.js           # JavaScript logic
├── trained_models/          # ML models (pkl files)
├── start_web_application.py # Startup script
└── README_WEB_APP.md       # This file
```

## 📡 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/app` | GET | Serve frontend application |
| `/docs` | GET | Interactive API documentation |

### Prediction Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | Predict disease risks |
| `/api/whatif` | POST | What-if scenario analysis |
| `/api/models` | GET | Get loaded models info |
| `/api/feature-info` | GET | Get feature descriptions |
| `/api/sample-patients` | GET | Get sample patient data |

### Example API Request

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

## 🎨 Frontend Features

### Modern UI/UX
- Clean, professional medical interface
- Responsive design (works on mobile, tablet, desktop)
- Real-time form validation
- Loading states and error handling
- Animated risk visualizations

### Interactive Elements
- Sample patient loader
- Risk score progress bars
- Color-coded risk categories:
  - 🟢 **LOW** (0-30%)
  - 🟡 **MODERATE** (30-50%)
  - 🟠 **HIGH** (50-70%)
  - 🔴 **CRITICAL** (70-100%)

## 🔧 Configuration

### Port Configuration
Default port is 8000. To change:

```python
# In start_web_application.py
uvicorn.run(
    "backend.main:app",
    host="0.0.0.0",
    port=8080,  # Change port here
    reload=True
)
```

### CORS Configuration
Add allowed origins in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🧪 Testing

### Manual Testing
1. Start the application
2. Navigate to http://localhost:8000/app
3. Load a sample patient
4. Click "Analyze Patient Risk"
5. Review predictions and explanations

### API Testing
Use the interactive docs at http://localhost:8000/docs to test API endpoints directly.

### Sample Patient Profiles

The application includes 4 pre-configured sample patients:
1. **Healthy Patient** - Normal vitals and labs
2. **Sepsis Risk** - Elevated WBC, fever, high lactate
3. **Kidney Failure Risk** - Elevated creatinine and BUN
4. **Diabetes Risk** - Very high glucose

## 📊 Input Parameters

### Demographics
- **Age:** 18-100 years
- **Gender:** 0 (Female) or 1 (Male)

### Vital Signs
- **Heart Rate:** 40-200 bpm
- **Systolic BP:** 70-250 mmHg
- **Diastolic BP:** 40-150 mmHg
- **Temperature:** 95-106 °F
- **Respiratory Rate:** 8-50 breaths/min

### Lab Values
- **WBC Count:** 1-50 K/µL
- **Hemoglobin:** 5-20 g/dL
- **Platelet Count:** 20-700 K/µL
- **Creatinine:** 0.3-15 mg/dL
- **BUN:** 5-200 mg/dL
- **Glucose:** 50-700 mg/dL
- **Lactate:** 0.5-25 mmol/L

## 🐛 Troubleshooting

### Models Not Loading
```
Error: Model not found: sepsis
```
**Solution:** Train models first using `python train_advanced_models.py`

### Port Already in Use
```
Error: Address already in use
```
**Solution:** Change port in `start_web_application.py` or kill process using port 8000

### Frontend Not Displaying
**Solution:** Ensure `backend/static/` directory exists with all files:
- index.html
- styles.css
- app.js

### CORS Errors
**Solution:** Add your domain to allowed origins in `backend/main.py`

## 🔐 Security Notes

⚠️ **Important:** This is a research/educational system:

1. **Not for clinical use** - Do not use for actual medical decisions
2. **Development mode** - Uses `reload=True` (disable in production)
3. **CORS** - Configure properly for production deployment
4. **Input validation** - Always validate patient data
5. **HIPAA compliance** - Not HIPAA compliant as-is

## 🚀 Deployment

### Production Deployment

1. **Disable reload mode:**
```python
uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
```

2. **Use production ASGI server:**
```bash
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

3. **Set up reverse proxy (nginx):**
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

4. **Enable HTTPS:**
Use Let's Encrypt or other SSL certificate

## 📝 Development

### Adding New Features

#### New API Endpoint
```python
@app.get("/api/custom-endpoint")
async def custom_endpoint():
    return {"data": "your data"}
```

#### New Frontend View
1. Add HTML section in `index.html`
2. Add navigation button
3. Implement JavaScript in `app.js`
4. Style in `styles.css`

### Code Structure

**Backend (backend/main.py):**
- Pydantic models for validation
- ModelManager for ML model handling
- Feature engineering functions
- API route handlers

**Frontend:**
- `index.html` - Structure
- `styles.css` - Styling with CSS variables
- `app.js` - Event handlers and API calls

## 📚 Additional Resources

- **Main Documentation:** See `Readme/` folder
- **API Schema:** http://localhost:8000/openapi.json
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **SHAP Documentation:** https://shap.readthedocs.io

## 🤝 Contributing

To enhance the web application:

1. Follow existing code structure
2. Maintain responsive design
3. Add error handling
4. Update this documentation
5. Test thoroughly

## 📄 License

See LICENSE.md in the main project directory.

---

**Need Help?**
- Check API docs at http://localhost:8000/docs
- Review main project README.md
- Check troubleshooting section above
