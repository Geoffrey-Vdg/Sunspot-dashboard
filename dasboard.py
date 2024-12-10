import os
import json
import dash
import base64
import shutil
import threading
import glob
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from PIL import Image
import io
import sys
import warnings
import time


warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Define the app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Sunspot Detection Dashboard'
server = app.server

# Global variables
project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(project_dir, "dataset")
images_dir = os.path.join(dataset_dir, "images")
annotations_dir = os.path.join(dataset_dir, "annotations")
masks_dir = os.path.join(dataset_dir, "masks")  # For completeness
augmented_dataset_dir = os.path.join(project_dir, "augmented_dataset")  # May not be needed anymore
output_dir = os.path.join(project_dir, "output")
trained_models_dir = os.path.join(output_dir, "trained_models")
images_to_run_dir = os.path.join(project_dir, "images_to_run")
segmented_dir = os.path.join(output_dir, "segmented_dataset")
csv_path = os.path.join(output_dir, "sunspot_table.csv")

# Ensure necessary directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(segmented_dir, exist_ok=True)
os.makedirs(images_to_run_dir, exist_ok=True)
os.makedirs(trained_models_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Layout components for each tab
dataset_layout = html.Div([
    html.H5("Upload Labeled Dataset:"),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-images',
                children=html.Div(['Upload Images']),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True
            ),
            html.Div(id='images-upload-output'),
        ], width=6),
        dbc.Col([
            dcc.Upload(
                id='upload-annotations',
                children=html.Div(['Upload Annotations']),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed',
                    'borderRadius': '5px', 'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False  # Assuming one annotation file
            ),
            html.Div(id='annotations-upload-output'),
        ], width=6),
    ]),
    html.Hr(),
    html.H5("Upload Dataset for Inference:"),
    dcc.Upload(
        id='upload-inference-dataset',
        children=html.Div(['Drag and Drop or ', html.A('Select Images')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='inference-dataset-upload-output'),
    html.Hr(),
    # Remove Data Augmentation section if not needed
])

training_layout = html.Div([
    html.H5("Model Naming:"),
    dbc.Input(id='model-name', placeholder='Enter model name...', type='text'),
    html.Br(),
    dbc.Row([
    # Column for "Number of Epochs"
        dbc.Col([
            html.Div([
                html.H5("Number of Epochs", style={'textAlign': 'center', 'marginBottom': '10px'}),  # Label centered
                dbc.Input(id='num-epochs', placeholder='Enter number of epochs...',
                        type='number', min=1, value=10),  # Input field
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginRight': '20px'})
        ], width=6),  # Half of the row width

                # Column for "Batch Size"
        dbc.Col([
            html.Div([
                html.H5("Batch Size", style={'textAlign': 'center', 'marginBottom': '10px'}),  # Label centered
                dbc.Input(id='batch-size', placeholder='Enter batch size...',
                        type='number', min=1, value=20),  # Input field
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginRight': '20px'})
        ], width=6),
    
        # Column for "Learning Rate"
        dbc.Col([
            html.Div([
                html.H5("Learning Rate", style={'textAlign': 'center', 'marginBottom': '10px'}),  # Label centered
                dbc.Input(id='learning-rate', placeholder='Enter learning rate...',
                        type='number', min=0.0001, step = 0.01, value=0.001),  # Input field
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginRight': '20px'})
        ], width=6), 

        # Column for "Tiled Images size"
        dbc.Col([
            html.Div([
                html.H5("Tiled Images size", style={'textAlign': 'center', 'marginBottom': '10px'}),  # Label centered
                dbc.Input(id='img-size', placeholder='Enter Tiled image size...',
                        type='number', min=8, step =8, value=64),  # Input field
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginRight': '20px'})
        ], width=6),

        # Column for "Penalty Weight"
        dbc.Col([
            html.Div([
                html.H5("Penalty Weight", style={'textAlign': 'center', 'marginBottom': '10px'}),  # Label centered
                dbc.Input(id='penalty-weight', placeholder='Enter penalty weight...',
                        type='number', min=1, step=1, value=10.0),  # Input field
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'marginRight': '20px'})
        ], width=6),


    ], className='g-4', style={'display': 'flex', 'justifyContent': 'start', 'alignItems': 'center'}),
    html.Br(),
    dbc.Button("Train CNN", id='train-button', style={'backgroundColor': '#28a745', 'color': 'white'}),
    html.Div(id='training-output'),
    html.Hr(),
    html.H5("Training Progress:"),
    dcc.Loading(
        id="loading-training",
        type="default",
        children=html.Div(id="training-log", style={'whiteSpace': 'pre-line', 'height': '150px', 'overflowY': 'auto'})
    ),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    html.Hr(),
    html.H5("Training Metrics:"),
    html.Div([
        html.Img(id='training-plot', style={'width': '500px', 'height': 'auto'}),
        html.Div(id='evaluation-metrics')
    ])

])





inference_layout = html.Div([
    html.H5("Select Trained CNN:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[],
        placeholder="Select a trained model...",
        style={'backgroundColor': '#010626', 'color': 'white'}
    ),
    html.Br(),
    dbc.Button("Run Inference", id='inference-button', style={
        'backgroundColor': '#28a745',
        'color': 'white',
        'fontSize': '16px',
        'width': '200px',
        'height': '50px'
    }),
    html.Div(id='inference-output'),
    html.Hr(),
    html.H5("Inference Progress:"),
    dcc.Loading(
        id="loading-inference",
        type="default",
        children=html.Div(
            id="inference-log",
            style={
                'whiteSpace': 'pre-line',
                'height': '150px',
                'overflowY': 'auto'
            }
        )
    ),
    dcc.Interval(id='inference-interval', interval=1000, n_intervals=0),
    html.Hr(),
    html.H5("Processed Images:"),
    html.Div([
        html.Div([
            html.H6("Select Image:"),
            dcc.Dropdown(
                id='processed-images-dropdown',
                options=[],
                placeholder="Select an image...",
                style={
                    'backgroundColor': '#010626',
                    'color': 'white',
                    'width': '300px'  # Constrain the dropdown width
                }
            )
        ], style={
            'flex': '1',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'flex-start'
        }),
        html.Div(
            id='image-comparison',
            style={
                'flex': '2',
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'flex-start',  # Align images to the left of their container
                'alignItems': 'center',  # Align images vertically
                'marginLeft': '20px',  # Add spacing between dropdown and images
                'overflow': 'auto'
            }
        )
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'justifyContent': 'space-between'  # Space out the dropdown and images
    })
], style={
    'maxHeight': 'none',
    'overflow': 'auto'
})



# Main layout
app.layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col(
                dcc.Tabs(
                    id='tabs', value='tab-dataset', children=[
                        dcc.Tab(label='Dataset', value='tab-dataset', className='custom-tab'),
                        dcc.Tab(label='Training', value='tab-training', className='custom-tab'),
                        dcc.Tab(label='Inference', value='tab-inference', className='custom-tab'),
                    ],
                    className='custom-tabs',
                ),
                width='auto',
                style={'margin-left': 'auto'}
            ),
        ], align='center'),
    ], className='navbar-custom'),
    html.Div(id='tabs-content', style={'height': 'calc(100vh - 60px)', 'overflowY': 'auto'})
])

# Helper functions
def get_trained_models():
    models = []
    for file in os.listdir(trained_models_dir):
        if file.endswith('.pth'): 
            model_name = os.path.splitext(file)[0]  # Remove the .pth extension
            models.append({'label': model_name, 'value': model_name})
    return models


def get_processed_images():
    images = []
    for file in os.listdir(segmented_dir):
        if file.lower().endswith(('_overlay.png','_overlay.jpg','_overlay.jpeg','_overlay.bmp')):
            images.append({'label': file, 'value': file})
    return images

# Callbacks
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-dataset':
        return dataset_layout
    elif tab == 'tab-training':
        return training_layout
    elif tab == 'tab-inference':
        return inference_layout
    


@app.callback(
    Output('images-upload-output', 'children'),
    Input('upload-images', 'contents'),
    State('upload-images', 'filename'),
)
def handle_images_upload(contents_list, filenames):
    if contents_list is not None:
        for contents, filename in zip(contents_list, filenames):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            # Save image file
            save_path = os.path.join(images_dir, filename)
            with open(save_path, 'wb') as f:
                f.write(decoded)
        return dbc.Alert("Images uploaded successfully.", color="success")
    return ""

@app.callback(
    Output('annotations-upload-output', 'children'),
    Input('upload-annotations', 'contents'),
    State('upload-annotations', 'filename'),
)
def handle_annotations_upload(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Save annotations file
        save_path = os.path.join(annotations_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(decoded)
        # Optionally, check if annotations file is valid
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            required_keys = {'images', 'annotations', 'categories'}
            if not required_keys.issubset(data.keys()):
                return dbc.Alert("Invalid annotations file. Please upload a valid COCO format annotations file.", color="danger")
        except Exception as e:
            return dbc.Alert("Invalid annotations file. Please upload a valid COCO format annotations file.", color="danger")
        return dbc.Alert("Annotations uploaded successfully.", color="success")
    return ""

@app.callback(
    Output('inference-dataset-upload-output', 'children'),
    Input('upload-inference-dataset', 'contents'),
    State('upload-inference-dataset', 'filename'),
)
def handle_inference_dataset_upload(contents_list, filenames):
    if contents_list is not None:
        for contents, filename in zip(contents_list, filenames):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            save_path = os.path.join(images_to_run_dir, filename)
            with open(save_path, 'wb') as f:
                f.write(decoded)
        return dbc.Alert("Inference dataset uploaded successfully.", color="success")
    return ""



@app.callback(
    Output('training-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('model-name', 'value'),
    State('num-epochs', 'value'),
    State('penalty-weight', 'value'),
    State('img-size', 'value'),
    State('learning-rate', 'value'),
    State('batch-size', 'value')
)
def start_training(n_clicks, model_name, num_epochs, penalty_weight, img_size, learning_rate, batch_size):

    if n_clicks and model_name:
                # Ensure default values for missing inputs
        if learning_rate is None:
            learning_rate = 0.001  # Default learning rate
        if num_epochs is None:
            num_epochs = 10  # Default number of epochs
        if penalty_weight is None:
            penalty_weight = 10.0  # Default penalty weight
        if img_size is None or not isinstance(img_size, (int, float)) or img_size <= 0:
            img_size = 64  # Default value
        else:
            img_size = int(img_size)
            # Ensure img-size is divisible by 8
            if img_size % 8 != 0:
                img_size -= img_size % 8  # Round down to the nearest multiple of 8
                if img_size <= 0:
                    img_size = 64  # Fallback to default
        if batch_size is None:
            batch_size = 8  # Default batch size
            

         # Ensure float conversion for float inputs
        try:
            img_size = int(img_size)
            learning_rate = float(learning_rate)
        except ValueError:
            return dbc.Alert("Invalid input: 'Learning Rate' and 'Image Size' must be numbers.", color="danger")


        def train():
            train_script = os.path.join(project_dir, 'train_cnn.py')
            command = (
                f'python "{train_script}" '
                f'--model-name "{model_name}" '
                f'--epochs {num_epochs} '
                f'--penalty-weight {penalty_weight} '
                f'--img-size {img_size} '
                f'--learning-rate {learning_rate} '
                f'--batch-size {batch_size}'
            )
            os.system(command)

        # Start and block until training is complete
        training_thread = threading.Thread(target=train)
        training_thread.start()
        training_thread.join()

        return dbc.Alert(f"Training completed for model '{model_name}'. Checking for the plot...", color="success")

    return dbc.Alert("Please provide a valid model name and parameters.", color="danger")


@app.callback(
    Output('training-plot', 'src'),
    Output('evaluation-metrics', 'children'),
    Input('training-output', 'children'),
    State('model-name', 'value')
)
def display_training_plot(training_output, model_name):
    if training_output and model_name:
        # Correct file naming convention for the plot
        plot_save_path = os.path.join(output_dir, f"{model_name}_learning_curve.png")

        # Poll for the plot file
        for _ in range(20):  # Check every 0.1 seconds for up to 2 seconds
            time.sleep(0.1)
            if os.path.exists(plot_save_path):
                with open(plot_save_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode()
                plot_src = f"data:image/png;base64,{encoded_image}"
                return plot_src, f"Metrics Updated for {model_name}."

        # If plot is not found after polling
        return "", "Plot could not be loaded. Please check manually."

    return "", ""





@app.callback(
    Output('training-log', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_training_log(n):
    log_path = os.path.join(output_dir, 'logs', 'training.log')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                training_logs = f.read()
            return training_logs
        except UnicodeDecodeError as e:
            return f"[ERROR] Unable to decode training log: {e}"
    return "[INFO] No training logs available."


@app.callback(
    Output('model-dropdown', 'options'),
    Output('model-dropdown', 'value'),
    Input('tabs', 'value')
)
def update_model_dropdown(tab):
    if tab == 'tab-inference':
        models = get_trained_models()
        if models:
            return models, models[-1]['value']  # Select the last trained model by default
        else:
            return [], None
    return dash.no_update, dash.no_update

@app.callback(
    Output('inference-output', 'children'),
    Input('inference-button', 'n_clicks'),
    State('model-dropdown', 'value')
)
def run_inference(n_clicks, model_name):
    if n_clicks and model_name:
        # Run inference script
        def inference():
            inference_script = os.path.join(project_dir, 'run_inference.py')
            command = f'python "{inference_script}" --model-name "{model_name}"'
            os.system(command)
        threading.Thread(target=inference).start()
        return dbc.Alert("Inference started. Please wait...", color="info")
    return ""

@app.callback(
    Output('inference-log', 'children'),
    Input('inference-interval', 'n_intervals')
)
def update_inference_log(n):
    log_path = os.path.join(output_dir, 'inference.log')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            inference_logs = f.read()
        return inference_logs
    return ""




@app.callback(
    Output('processed-images-dropdown', 'options'),
    Output('processed-images-dropdown', 'value'),
    Input('inference-button', 'n_clicks'),
    State('model-dropdown', 'value')
)
def update_processed_images_list(_, __):
    images = get_processed_images()
    if images:
        return images, images[0]['value']  # Default to the first image in the list
    return [], None



@app.callback(
    Output('image-comparison', 'children'),
    Input('processed-images-dropdown', 'value')
)
def display_image_comparison(selected_image):
    if not selected_image:
        return html.Div("No image selected.", style={'textAlign': 'center', 'color': 'red'})

    # Base name of the selected image (e.g., "usd202412010953")
    base_name = selected_image.rsplit('_', 1)[0] if '_mask' in selected_image or '_overlay' in selected_image else selected_image.split('.')[0]

    # Paths for the images
    original_path = None
    for ext in ['.jpg', '.png', '.bmp']:
        potential_path = os.path.join("C:\\Users\\geoff\\OneDrive\\Bureau\\Sunspot detection cnn\\Sunspot detection cnn\\images_to_run", f"{base_name}{ext}")
        if os.path.exists(potential_path):
            original_path = potential_path
            break

    overlay_path = os.path.join("C:\\Users\\geoff\\OneDrive\\Bureau\\Sunspot detection cnn\\Sunspot detection cnn\\output\\segmented_dataset", f"{base_name}_overlay.png")
    mask_path = os.path.join("C:\\Users\\geoff\\OneDrive\\Bureau\\Sunspot detection cnn\\Sunspot detection cnn\\output\\segmented_dataset", f"{base_name}_mask.png")

    # Validate that all required files exist
    if not original_path or not os.path.exists(overlay_path) or not os.path.exists(mask_path):
        return html.Div("One or more required images are missing.", style={'textAlign': 'center', 'color': 'red'})

    # Display the images side by side
    return html.Div([
        html.Div([
            html.H6("Original"),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(original_path, 'rb').read()).decode()}", style={'maxWidth': '400px', 'maxHeight': '400px', 'objectFit': 'contain'})
        ], style={'padding': '10px'}),
        html.Div([
            html.H6("Overlay"),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(overlay_path, 'rb').read()).decode()}", style={'maxWidth': '400px', 'maxHeight': '400px', 'objectFit': 'contain'})
        ], style={'padding': '10px'}),
        html.Div([
            html.H6("Mask"),
            html.Img(src=f"data:image/png;base64,{base64.b64encode(open(mask_path, 'rb').read()).decode()}", style={'maxWidth': '400px', 'maxHeight': '400px', 'objectFit': 'contain'})
        ], style={'padding': '10px'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center'})



if __name__ == '__main__':
    app.run_server(debug=True)
