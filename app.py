import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Pipeline Corrosion Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained corrosion detection model"""
    try:
        # Try loading .h5 model first
        model = keras.models.load_model('corrosion_model.h5')
        return model
    except FileNotFoundError:
        st.error("Model file 'corrosion_model.h5' not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def normalize_input_data(input_data):
    """Normalize input data - customize based on your training data preprocessing"""
    # Example normalization - ADJUST THESE VALUES based on your training data
    normalization_params = {
        'pipeline_age': {'mean': 15, 'std': 8},
        'wall_thickness': {'mean': 10, 'std': 3},
        'pressure': {'mean': 550, 'std': 250},
        'temperature': {'mean': 45, 'std': 20},
        'ph_level': {'mean': 7.5, 'std': 1.2},
        'humidity': {'mean': 60, 'std': 20},
        'coating_condition': {'mean': 3, 'std': 1.5},
        'soil_resistivity': {'mean': 200, 'std': 150}
    }
    
    if isinstance(input_data, dict):
        normalized_data = {}
        for key, value in input_data.items():
            if key in normalization_params:
                mean = normalization_params[key]['mean']
                std = normalization_params[key]['std']
                normalized_data[key] = (value - mean) / std
            else:
                normalized_data[key] = value
        return normalized_data
    else:
        # For DataFrame
        normalized_df = input_data.copy()
        for col in normalized_df.columns:
            if col in normalization_params:
                mean = normalization_params[col]['mean']
                std = normalization_params[col]['std']
                normalized_df[col] = (normalized_df[col] - mean) / std
        return normalized_df
    """Create sample data for demonstration"""
    np.random.seed(42)
    return pd.DataFrame({
        'pipeline_age': np.random.randint(1, 30, 100),
        'wall_thickness': np.random.uniform(5.0, 15.0, 100),
        'pressure': np.random.uniform(100, 1000, 100),
        'temperature': np.random.uniform(10, 80, 100),
        'ph_level': np.random.uniform(6.0, 9.0, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'coating_condition': np.random.choice([1, 2, 3, 4, 5], 100),
        'soil_resistivity': np.random.uniform(10, 1000, 100)
    })

def predict_corrosion(model, input_data):
    """Make prediction using the loaded Keras model"""
    try:
        # Convert input to DataFrame if it's not already
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
        
        # Convert to numpy array for Keras model
        input_array = input_df.values.astype(np.float32)
        
        # Make prediction
        prediction = model.predict(input_array, verbose=0)
        
        # Handle different output formats
        if prediction.shape[1] == 1:
            # Binary classification or regression
            pred_values = prediction.flatten()
            probabilities = None
        else:
            # Multi-class classification
            pred_values = np.argmax(prediction, axis=1)
            probabilities = prediction
        
        return pred_values, probabilities
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def main():
    # Header
    st.title("🔧 Pipeline Corrosion Detection System")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("Please upload your corrosion_model.h5 file to proceed.")
        uploaded_file = st.file_uploader("Upload Model File", type=['h5'])
        if uploaded_file:
            try:
                # Save uploaded file temporarily
                with open("temp_model.h5", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                model = keras.models.load_model("temp_model.h5")
                st.success("Model loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading uploaded model: {str(e)}")
        return
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        mode = st.radio(
            "Select Mode:",
            ["Single Prediction", "Batch Prediction", "Model Analysis", "About"]
        )
    
    if mode == "Single Prediction":
        single_prediction_interface(model)
    elif mode == "Batch Prediction":
        batch_prediction_interface(model)
    elif mode == "Model Analysis":
        model_analysis_interface(model)
    else:
        about_section()

def single_prediction_interface(model):
    st.header("Single Pipeline Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pipeline Parameters")
        pipeline_age = st.number_input("Pipeline Age (years)", min_value=0, max_value=100, value=10)
        wall_thickness = st.number_input("Wall Thickness (mm)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
        pressure = st.number_input("Operating Pressure (psi)", min_value=0, max_value=2000, value=500)
        temperature = st.number_input("Operating Temperature (°C)", min_value=-50, max_value=200, value=25)
    
    with col2:
        st.subheader("Environmental Conditions")
        ph_level = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=50)
        coating_condition = st.selectbox(
            "Coating Condition", 
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Excellent", 2: "Good", 3: "Fair", 4: "Poor", 5: "Very Poor"}[x]
        )
        soil_resistivity = st.number_input("Soil Resistivity (Ω·m)", min_value=1, max_value=10000, value=100)
    
    # Prediction button
    if st.button("Assess Corrosion Risk", type="primary"):
        input_data = {
            'pipeline_age': pipeline_age,
            'wall_thickness': wall_thickness,
            'pressure': pressure,
            'temperature': temperature,
            'ph_level': ph_level,
            'humidity': humidity,
            'coating_condition': coating_condition,
            'soil_resistivity': soil_resistivity
        }
        
        # Normalize input data (if your model was trained on normalized data)
        # Uncomment the next line if your model expects normalized inputs
        # input_data = normalize_input_data(input_data)
        
        prediction, probabilities = predict_corrosion(model, input_data)
        
        if prediction is not None:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_level = prediction[0]
                if isinstance(risk_level, (int, float)):
                    if risk_level >= 0.7:
                        st.error(f"🚨 HIGH RISK: {risk_level:.2%}")
                    elif risk_level >= 0.4:
                        st.warning(f"⚠️ MEDIUM RISK: {risk_level:.2%}")
                    else:
                        st.success(f"✅ LOW RISK: {risk_level:.2%}")
                else:
                    st.info(f"Prediction: {risk_level}")
            
            with col2:
                if probabilities is not None:
                    st.metric("Confidence", f"{np.max(probabilities[0]):.2%}")
            
            with col3:
                st.metric("Assessment Date", pd.Timestamp.now().strftime("%Y-%m-%d"))
            
            # Risk factors visualization
            if probabilities is not None:
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Low Risk', 'Medium Risk', 'High Risk'],
                        y=probabilities[0] if len(probabilities[0]) == 3 else [probabilities[0][0], 0.5, probabilities[0][1]],
                        marker_color=['green', 'orange', 'red']
                    )
                ])
                fig.update_layout(title="Risk Assessment Probabilities", yaxis_title="Probability")
                st.plotly_chart(fig, use_container_width=True)

def batch_prediction_interface(model):
    st.header("Batch Pipeline Assessment")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with pipeline data",
        type=['csv'],
        help="CSV should contain columns matching your model's features"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} records")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Make predictions
            if st.button("Run Batch Assessment"):
                # Normalize data if needed
                # Uncomment the next line if your model expects normalized inputs
                # df_normalized = normalize_input_data(df)
                
                predictions, probabilities = predict_corrosion(model, df)
                
                if predictions is not None:
                    # Add predictions to dataframe
                    df['corrosion_risk'] = predictions
                    if probabilities is not None:
                        df['confidence'] = np.max(probabilities, axis=1)
                    
                    # Display results
                    st.subheader("Assessment Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    if isinstance(predictions[0], (int, float)):
                        high_risk = sum(p >= 0.7 for p in predictions)
                        medium_risk = sum(0.4 <= p < 0.7 for p in predictions)
                        low_risk = sum(p < 0.4 for p in predictions)
                        
                        with col1:
                            st.metric("High Risk Pipelines", high_risk)
                        with col2:
                            st.metric("Medium Risk Pipelines", medium_risk)
                        with col3:
                            st.metric("Low Risk Pipelines", low_risk)
                        
                        # Risk distribution chart
                        fig = px.histogram(
                            x=predictions,
                            nbins=20,
                            title="Risk Distribution",
                            labels={'x': 'Corrosion Risk Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="corrosion_assessment_results.csv">Download Results</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Provide sample data option
        if st.button("Use Sample Data for Demo"):
            sample_df = create_sample_data()
            st.session_state['sample_data'] = sample_df
            st.success("Sample data generated!")
            st.dataframe(sample_df.head())

def model_analysis_interface(model):
    st.header("Model Analysis & Insights")
    
    # Model information
    try:
        st.subheader("Model Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"Model Type: {type(model).__name__}")
            st.info(f"Input Shape: {model.input_shape}")
            st.info(f"Output Shape: {model.output_shape}")
        
        with col2:
            st.info(f"Total Parameters: {model.count_params():,}")
            st.info(f"Layers: {len(model.layers)}")
            if len(model.layers) > 0:
                st.info(f"First Layer: {model.layers[0].__class__.__name__}")
        
        # Model architecture summary
        st.subheader("Model Architecture")
        
        # Create a string buffer to capture model summary
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            model.summary()
            summary_string = buffer.getvalue()
            sys.stdout = old_stdout
            st.text(summary_string)
        except:
            sys.stdout = old_stdout
            st.info("Model summary not available")
        
        # Training history visualization (if available)
        st.subheader("Model Information")
        st.markdown("""
        **Deep Learning Model Features:**
        - Neural network architecture for corrosion prediction
        - Handles complex non-linear relationships
        - Trained on historical pipeline data
        - Supports both regression and classification outputs
        """)
        
        # Model performance metrics (if available)
        st.subheader("Performance Guidelines")
        st.markdown("""
        **Risk Assessment Interpretation:**
        - **Low Risk (0.0 - 0.4)**: Regular maintenance schedule sufficient
        - **Medium Risk (0.4 - 0.7)**: Increased monitoring recommended
        - **High Risk (0.7 - 1.0)**: Immediate inspection and potential replacement needed
        
        **Key Factors for Corrosion:**
        - Pipeline age and wall thickness
        - Environmental conditions (pH, humidity)
        - Operating conditions (pressure, temperature)
        - Protective coating condition
        """)
        
    except Exception as e:
        st.error(f"Error analyzing model: {str(e)}")

def about_section():
    st.header("About Pipeline Corrosion Detection")
    
    st.markdown("""
    ## Overview
    This application uses machine learning to assess corrosion risk in pipelines based on various operational and environmental factors.
    
    ## Key Features
    - **Single Assessment**: Evaluate individual pipeline segments
    - **Batch Processing**: Analyze multiple pipelines from CSV data
    - **Risk Visualization**: Interactive charts and risk indicators
    - **Model Insights**: Understanding feature importance and model behavior
    
    ## Input Parameters
    - **Pipeline Age**: Years since installation
    - **Wall Thickness**: Current thickness measurement
    - **Operating Pressure**: Internal pressure during operation
    - **Operating Temperature**: Temperature during operation
    - **pH Level**: Environmental acidity/alkalinity
    - **Humidity**: Environmental moisture content
    - **Coating Condition**: Protective coating state (1=Excellent, 5=Very Poor)
    - **Soil Resistivity**: Electrical resistance of surrounding soil
    
    ## Usage Instructions
    1. Load your trained model file (.pkl format)
    2. Choose between single or batch prediction mode
    3. Input pipeline parameters or upload CSV data
    4. Review risk assessments and recommendations
    
    ## Model Requirements
    Your .h5 model should:
    - Accept 8 input features (or adjust based on your model)
    - Be trained using TensorFlow/Keras
    - Handle the standard pipeline parameters
    - Return risk predictions (0-1 for binary, or class probabilities)
    
    ## Installation Requirements
    ```bash
    pip install streamlit pandas numpy tensorflow plotly
    ```
    """)

if __name__ == "__main__":
    main()