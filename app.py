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
from PIL import Image
import json
import os

# Page configuration
st.set_page_config(
    page_title="Pipeline Corrosion Detection",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom DepthwiseConv2D class to handle the groups parameter
class FixedDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove the 'groups' parameter if it exists
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Register the custom layer
keras.utils.get_custom_objects()['DepthwiseConv2D'] = FixedDepthwiseConv2D

@st.cache_resource
def load_model():
    """Load the trained corrosion detection model with multiple fallback methods"""
    model = None
    error_messages = []
    
    # Method 1: Try loading with custom objects
    try:
        custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}
        model = keras.models.load_model('corrosion_model.h5', custom_objects=custom_objects)
        st.success("✅ Model loaded successfully with custom objects!")
        return model
    except FileNotFoundError:
        error_messages.append("Model file 'corrosion_model.h5' not found.")
    except Exception as e:
        error_messages.append(f"Custom objects method failed: {str(e)}")
    
    # Method 2: Try loading with compile=False
    try:
        model = keras.models.load_model('corrosion_model.h5', compile=False)
        st.success("✅ Model loaded successfully without compilation!")
        return model
    except FileNotFoundError:
        pass  # Already recorded above
    except Exception as e:
        error_messages.append(f"No-compile method failed: {str(e)}")
    
    # Method 3: Try loading weights only if architecture file exists
    try:
        if os.path.exists('model_architecture.json'):
            with open('model_architecture.json', 'r') as f:
                model_json = f.read()
            model = keras.models.model_from_json(model_json)
            model.load_weights('model_weights.h5')
            st.success("✅ Model loaded from architecture and weights!")
            return model
    except Exception as e:
        error_messages.append(f"Architecture+weights method failed: {str(e)}")
    
    # Method 4: Try with TensorFlow compatibility mode
    try:
        # Disable eager execution temporarily
        tf.compat.v1.disable_eager_execution()
        model = keras.models.load_model('corrosion_model.h5')
        tf.compat.v1.enable_eager_execution()
        st.success("✅ Model loaded with TF compatibility mode!")
        return model
    except Exception as e:
        error_messages.append(f"TF compatibility method failed: {str(e)}")
        tf.compat.v1.enable_eager_execution()  # Re-enable if it failed
    
    # All methods failed
    st.error("❌ Failed to load model with all methods:")
    for i, msg in enumerate(error_messages, 1):
        st.error(f"{i}. {msg}")
    
    return None

def save_model_components(model_path):
    """Helper function to save model architecture and weights separately"""
    try:
        model = keras.models.load_model(model_path, compile=False)
        
        # Save architecture
        model_json = model.to_json()
        with open('model_architecture.json', 'w') as f:
            f.write(model_json)
        
        # Save weights
        model.save_weights('model_weights.h5')
        
        st.success("Model components saved successfully!")
        return True
    except Exception as e:
        st.error(f"Error saving model components: {str(e)}")
        return False

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction using PIL only"""
    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        img_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_array
    
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

def predict_corrosion_from_image(model, image):
    """Make prediction using the loaded Keras model on image"""
    try:
        # Preprocess the image
        processed_image, display_image = preprocess_image(image)
        
        if processed_image is None:
            return None, None, None
        
        # Make prediction with error handling
        try:
            prediction = model.predict(processed_image, verbose=0)
        except Exception as pred_error:
            st.error(f"Prediction error: {str(pred_error)}")
            # Try alternative prediction method
            try:
                prediction = model(processed_image, training=False)
                if hasattr(prediction, 'numpy'):
                    prediction = prediction.numpy()
            except Exception as alt_error:
                st.error(f"Alternative prediction also failed: {str(alt_error)}")
                return None, None, None
        
        # Handle different output formats
        if prediction.shape[1] == 1:
            # Binary classification or regression
            pred_values = prediction.flatten()[0]
            probabilities = None
        else:
            # Multi-class classification
            pred_values = np.argmax(prediction, axis=1)[0]
            probabilities = prediction[0]
        
        return pred_values, probabilities, display_image
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def create_sample_images():
    """Create placeholder for sample images"""
    return {
        "No Corrosion": "sample_no_corrosion.jpg",
        "Light Corrosion": "sample_light_corrosion.jpg", 
        "Heavy Corrosion": "sample_heavy_corrosion.jpg"
    }

def main():
    # Header
    st.title("🔧 Pipeline Corrosion Detection System")
    st.markdown("**AI-Powered Visual Inspection for Pipeline Integrity**")
    st.markdown("---")
    
    # Model loading section with troubleshooting
    st.subheader("Model Loading Status")
    
    # Add model troubleshooting options
    with st.expander("🔧 Model Troubleshooting Options"):
        st.markdown("""
        **If you're getting model loading errors, try these solutions:**
        
        1. **TensorFlow Version Mismatch**: Update TensorFlow to match your model version
        ```bash
        pip install tensorflow==2.15.0  # or your specific version
        ```
        
        2. **Model Conversion**: Convert your model to a compatible format
        3. **Architecture + Weights**: Save model architecture and weights separately
        4. **Custom Objects**: Handle custom layers with compatibility fixes
        """)
        
        # Model conversion tool
        st.markdown("**Model Conversion Tool:**")
        uploaded_model = st.file_uploader(
            "Upload your .h5 model file for conversion", 
            type=['h5'],
            help="This will attempt to convert your model to a compatible format"
        )
        
        if uploaded_model:
            if st.button("Convert Model"):
                try:
                    # Save the uploaded model temporarily
                    with open("temp_model.h5", "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    
                    # Try to convert it
                    with st.spinner("Converting model..."):
                        success = save_model_components("temp_model.h5")
                        if success:
                            st.success("✅ Model converted! Architecture and weights saved separately.")
                            st.info("Files created: model_architecture.json, model_weights.h5")
                except Exception as e:
                    st.error(f"Conversion failed: {str(e)}")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please check the troubleshooting options above.")
        
        # Alternative model upload
        st.subheader("Upload Model File")
        uploaded_file = st.file_uploader("Upload Model File (.h5)", type=['h5'])
        if uploaded_file:
            try:
                # Save uploaded file temporarily
                with open("temp_model.h5", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Try loading with custom objects
                try:
                    custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}
                    model = keras.models.load_model("temp_model.h5", custom_objects=custom_objects)
                    st.success("✅ Uploaded model loaded successfully!")
                    st.rerun()
                except:
                    # Try without compilation
                    model = keras.models.load_model("temp_model.h5", compile=False)
                    st.success("✅ Uploaded model loaded without compilation!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading uploaded model: {str(e)}")
                st.info("💡 Try the model conversion tool above")
        return
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        mode = st.radio(
            "Select Mode:",
            ["Single Image Analysis", "Batch Image Analysis", "Model Analysis", "About"]
        )
        
        st.markdown("---")
        st.subheader("Image Requirements")
        st.markdown("""
        - **Format**: JPG, PNG, JPEG
        - **Size**: Any size (auto-resized)
        - **Quality**: Clear, well-lit images
        - **Focus**: Pipeline surface area
        """)
        
        # Model status indicator
        st.markdown("---")
        st.subheader("System Status")
        st.success("🟢 Model: Loaded")
        st.info(f"📊 TensorFlow: {tf.__version__}")
    
    if mode == "Single Image Analysis":
        single_image_interface(model)
    elif mode == "Batch Image Analysis":
        batch_image_interface(model)
    elif mode == "Model Analysis":
        model_analysis_interface(model)
    else:
        about_section()

def single_image_interface(model):
    st.header("Single Pipeline Image Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Pipeline Image")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Choose a pipeline image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the pipeline surface"
        )
        
        # Camera input option
        st.markdown("**Or take a photo:**")
        camera_image = st.camera_input("Take a picture")
        
        # Use either uploaded or camera image
        image_source = uploaded_image if uploaded_image else camera_image
        
        if image_source:
            # Display the image
            image = Image.open(image_source)
            st.image(image, caption="Pipeline Image", use_column_width=True)
            
            # Image info
            st.info(f"Image size: {image.size[0]} x {image.size[1]} pixels")
            
            # Additional metadata input
            st.subheader("Optional Metadata")
            pipeline_id = st.text_input("Pipeline ID", placeholder="e.g., PIPE-001")
            location = st.text_input("Location", placeholder="e.g., Section A, Mile 5")
            inspection_date = st.date_input("Inspection Date")
    
    with col2:
        st.subheader("Analysis Results")
        
        if image_source:
            # Prediction button
            if st.button("Analyze Corrosion", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    prediction, probabilities, processed_img = predict_corrosion_from_image(model, image)
                
                if prediction is not None:
                    # Display processed image
                    if processed_img is not None:
                        st.subheader("Processed Image")
                        st.image(processed_img, caption="Model Input (Resized)", use_column_width=True)
                    
                    st.markdown("---")
                    
                    # Results display
                    if isinstance(prediction, float):
                        # Regression output (0-1 scale)
                        risk_score = prediction
                        
                        if risk_score >= 0.7:
                            st.error(f"🚨 **HIGH CORROSION RISK**")
                            st.error(f"Risk Score: {risk_score:.3f}")
                            recommendation = "Immediate inspection and potential replacement required"
                            color = "red"
                        elif risk_score >= 0.4:
                            st.warning(f"⚠️ **MEDIUM CORROSION RISK**")
                            st.warning(f"Risk Score: {risk_score:.3f}")
                            recommendation = "Increased monitoring and maintenance recommended"
                            color = "orange"
                        else:
                            st.success(f"✅ **LOW CORROSION RISK**")
                            st.success(f"Risk Score: {risk_score:.3f}")
                            recommendation = "Regular maintenance schedule sufficient"
                            color = "green"
                        
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = risk_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Corrosion Risk Score"},
                            delta = {'reference': 0.5},
                            gauge = {
                                'axis': {'range': [None, 1]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 0.4], 'color': "lightgreen"},
                                    {'range': [0.4, 0.7], 'color': "yellow"},
                                    {'range': [0.7, 1], 'color': "lightcoral"}],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.9}}))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        # Classification output
                        class_names = ["No Corrosion", "Light Corrosion", "Moderate Corrosion", "Severe Corrosion"]
                        if len(class_names) > prediction:
                            predicted_class = class_names[prediction]
                        else:
                            predicted_class = f"Class {prediction}"
                        
                        st.info(f"**Predicted Class:** {predicted_class}")
                        
                        if probabilities is not None:
                            # Probability chart
                            prob_df = pd.DataFrame({
                                'Class': class_names[:len(probabilities)],
                                'Probability': probabilities
                            })
                            
                            fig = px.bar(
                                prob_df, 
                                x='Class', 
                                y='Probability',
                                title="Class Probabilities",
                                color='Probability',
                                color_continuous_scale='RdYlGn_r'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Set recommendation based on class
                        if prediction == 0:
                            recommendation = "No immediate action required"
                            color = "green"
                        elif prediction == 1:
                            recommendation = "Monitor condition, schedule routine maintenance"
                            color = "yellow"
                        elif prediction == 2:
                            recommendation = "Increased inspection frequency recommended"
                            color = "orange"
                        else:
                            recommendation = "Immediate inspection and repair required"
                            color = "red"
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    st.markdown(f"**Action Required:** {recommendation}")
                    
                    # Report generation
                    if st.button("Generate Report"):
                        report_data = {
                            "Pipeline ID": pipeline_id if pipeline_id else "N/A",
                            "Location": location if location else "N/A",
                            "Inspection Date": str(inspection_date),
                            "Risk Score/Class": prediction,
                            "Recommendation": recommendation
                        }
                        
                        st.subheader("Inspection Report")
                        for key, value in report_data.items():
                            st.write(f"**{key}:** {value}")
        else:
            st.info("👆 Upload an image or take a photo to start analysis")

def batch_image_interface(model):
    st.header("Batch Pipeline Image Analysis")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload multiple pipeline images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Select multiple images for batch processing"
    )
    
    if uploaded_files:
        st.success(f"Loaded {len(uploaded_files)} images")
        
        # Display thumbnails
        st.subheader("Image Preview")
        cols = st.columns(min(4, len(uploaded_files)))
        
        for idx, uploaded_file in enumerate(uploaded_files[:4]):
            with cols[idx % 4]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
        
        if len(uploaded_files) > 4:
            st.info(f"... and {len(uploaded_files) - 4} more images")
        
        # Batch processing
        if st.button("Analyze All Images", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                image = Image.open(uploaded_file)
                prediction, probabilities, _ = predict_corrosion_from_image(model, image)
                
                if prediction is not None:
                    if isinstance(prediction, float):
                        risk_level = "High" if prediction >= 0.7 else "Medium" if prediction >= 0.4 else "Low"
                        confidence = prediction
                    else:
                        class_names = ["No Corrosion", "Light", "Moderate", "Severe"]
                        risk_level = class_names[prediction] if prediction < len(class_names) else f"Class {prediction}"
                        confidence = np.max(probabilities) if probabilities is not None else 0.0
                    
                    results.append({
                        "Image": uploaded_file.name,
                        "Risk Level": risk_level,
                        "Confidence": f"{confidence:.3f}",
                        "Action Required": get_recommendation(prediction)
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Analysis complete!")
            
            # Display results
            if results:
                st.subheader("Batch Analysis Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                high_risk_count = sum(1 for r in results if "High" in r["Risk Level"] or "Severe" in r["Risk Level"])
                medium_risk_count = sum(1 for r in results if "Medium" in r["Risk Level"] or "Moderate" in r["Risk Level"] or "Light" in r["Risk Level"])
                low_risk_count = len(results) - high_risk_count - medium_risk_count
                
                with col1:
                    st.metric("High Risk", high_risk_count)
                with col2:
                    st.metric("Medium Risk", medium_risk_count)
                with col3:
                    st.metric("Low Risk", low_risk_count)
                
                # Risk distribution chart
                risk_data = {"High": high_risk_count, "Medium": medium_risk_count, "Low": low_risk_count}
                fig = px.pie(
                    values=list(risk_data.values()),
                    names=list(risk_data.keys()),
                    title="Risk Distribution",
                    color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="batch_corrosion_analysis.csv">📥 Download Results CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    else:
        st.info("👆 Upload multiple images to start batch analysis")

def get_recommendation(prediction):
    """Get recommendation based on prediction"""
    if isinstance(prediction, float):
        if prediction >= 0.7:
            return "Immediate inspection required"
        elif prediction >= 0.4:
            return "Increased monitoring"
        else:
            return "Regular maintenance"
    else:
        recommendations = [
            "No action required",
            "Monitor condition", 
            "Increased inspection",
            "Immediate repair"
        ]
        return recommendations[prediction] if prediction < len(recommendations) else "Review required"

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
        
        # Model capabilities
        st.subheader("Model Capabilities")
        st.markdown("""
        **Computer Vision Model Features:**
        - Convolutional Neural Network for image analysis
        - Trained on pipeline corrosion imagery
        - Handles various lighting and angle conditions
        - Supports real-time inference
        - Robust to image quality variations
        """)
        
        # Performance guidelines
        st.subheader("Usage Guidelines")
        st.markdown("""
        **For Best Results:**
        - Use clear, well-lit images
        - Focus on pipeline surface area
        - Avoid extreme angles or distances
        - Ensure good contrast between corrosion and pipe surface
        - Remove obstructions from view
        
        **Image Quality Tips:**
        - Resolution: Minimum 224x224 pixels
        - Format: JPG, PNG preferred
        - Lighting: Natural or bright artificial light
        - Distance: 1-3 feet from surface
        """)
        
    except Exception as e:
        st.error(f"Error analyzing model: {str(e)}")

def about_section():
    st.header("About Pipeline Corrosion Detection")
    
    st.markdown("""
    ## Overview
    This application uses deep learning computer vision to detect and assess corrosion in pipeline imagery.
    
    ## Key Features
    - **Single Image Analysis**: Upload or capture pipeline images for instant analysis
    - **Batch Processing**: Analyze multiple images simultaneously
    - **Risk Assessment**: Automated classification and scoring
    - **Visual Reports**: Generate inspection reports with recommendations
    - **Real-time Processing**: Fast inference for field use
    - **Model Compatibility**: Advanced error handling for different TensorFlow versions
    
    ## Troubleshooting Model Loading Issues
    
    ### Common Solutions:
    1. **Update TensorFlow**: Ensure version compatibility
    ```bash
    pip install --upgrade tensorflow
    ```
    
    2. **Use Model Conversion**: Convert incompatible models using the built-in tool
    
    3. **Check Model Format**: Ensure your model is saved in .h5 format
    
    4. **Custom Layers**: The app handles custom layer compatibility automatically
    
    ### Supported Model Types:
    - Standard Keras/TensorFlow models (.h5)
    - Models with custom layers (DepthwiseConv2D, etc.)
    - Both classification and regression outputs
    - Various CNN architectures (ResNet, MobileNet, EfficientNet, etc.)
    
    ## Technical Details
    - **Framework**: TensorFlow/Keras
    - **Input**: 224x224x3 RGB images (auto-resized)
    - **Output**: Risk scores (0-1) or class probabilities
    - **Processing**: Real-time inference with PIL image handling
    - **Compatibility**: Handles TensorFlow version mismatches
    
    ## Installation Requirements
    ```bash
    pip install streamlit pandas numpy tensorflow plotly pillow
    ```
    
    ## Use Cases
    - Pipeline inspection and maintenance
    - Asset integrity management
    - Preventive maintenance planning
    - Regulatory compliance documentation
    - Field inspection support

    ## Created by
    - **Website Developer**: Shubham Dave
    - **Model Trainer**: Aditya Singh 
    """)

if __name__ == "__main__":
    main()