
#### 🌐 Python Api For React Frontend (Hosted On Hugging Face Spaces)
This repository contains a minimal Python backend (using FastAPI) that exposes API endpoints. It is deployed on Hugging Face Spaces and is designed to communicate with a React frontend hosted in the development server.


##### 🔧 Features
- Receives payloads from a client-side react application and returns Texture predictions and Fertilizer recommendations
  
- Setup in a Docker container for fast deployment and scaling

- Public REST API endpoints (e.g., /predict, /status)

- Cross-Origin Resource Sharing (CORS) enabled for frontend access

- Can receive data from and send responses to React app

- Hosted on Hugging Face Spaces (Free, serverless deployment)


##### API Usage
```javascript
try {
      const response = await fetch('api_endpoint', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          soil_type: formData.soilTexture,
          nitrogen: parseFloat(formData.nitrogen),
          phosphorus: parseFloat(formData.phosphorus),
          potassium: parseFloat(formData.potassium),
          temperature: parseFloat(formData.temperature),
          humidity: parseFloat(formData.humidity),
          rainfall: parseFloat(formData.rainfall),
          ph: parseFloat(formData.ph)
        }),
      });

      const data = await response.json();
      setRecommendations(data.predictions || []);
      
    } catch (err) {
      setError(err.message || 'Failed to get recommendations');
    } finally {
      setIsLoading(false);
    }
```

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
