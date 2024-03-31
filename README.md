# Stable Diffusion with FastAPI and React

This project shows how to pair together a high performant fastapi backend with a react frontend.

## Project Overview

This project is designed to show how a python based fastapi backend can be connected to a react frontend for stable diffusion based image generation. It is divided into two main sections -
1. Backend - logic containing the fastapi endpoints with stable diffusion inference
2. Frontend - the react frontend with a little bit of css

## Installation

1. ### Install the requirements

```bash
pip install -r requirements.txt
```

2. ### Setup ruff for linting and styling using the toml file

```bash
pip install ruff
```

3. ### Navigate to the backend directory and start the server
```bash
cd src/
cd backend/
uvicorn main:app --reload
```

4. ### Navigate to the frontend directory and start the node.js server
```bash
cd src/
cd frontend/
npm start
```

5. ### Launch the UI on http://localhost:3000/ and have fun :)
