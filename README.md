# Stable Diffusers FastAPI Application

This project is a FastAPI application that serves a Stable Diffusers model over HTTP/REST APIs. It allows users to generate images using the Stable Diffusion technique.

## Features

- Generate images using the Stable Diffusion technique.
- RESTful API endpoints to interact with the model.
- Integration with Pydantic for input data validation.
- Streaming response to efficiently serve generated imag

## Setup

- Move to the backend folder.
- Run the following:
    ```bash
    cd backend 
    touch .env 
    ```
- Open the .env file, and paste your HF token:
    ```bash
    HF_TOKEN=YOUR_TOKEN
    ```
- Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
- Start your server:
    ```bash
    uvicorn main:app --port 8000
    ```
- Run the following CURL command:
    ```bash
    curl -X GET "localhost:8000/api/generate" -H "Content-Type: image/png"
    ```
## Crux: ML Engineer

### Fine tune a Stable Diffusion model and serve it

- create a github repo
- create a `trainer.py` file that takes in arguments like file path `--fp` and some other training arguments and fintunes a stable diffusion model (protip: check it out on `huggingface`)
- the result of the training should be a model weight file
- create a file called `server.py` that serves the Model over a HTTP/REST over some APIs (protip: use `pydantic` for models)
- A `curl` command to call the model and get response
- an ipython notebook that contains the steps to run this