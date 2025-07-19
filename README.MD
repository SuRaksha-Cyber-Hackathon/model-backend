# SuRaksha Cyber Hackathon - Model Backend

This repository implements the backend server for behavioral biometric authentication based on sensor and keypress data, designed for the SuRaksha Cyber Hackathon project.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Models](#models)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

---

## Overview

This backend provides RESTful APIs for:
- Enrolling users with biometric behavioral data (sensor and keypress events).
- Verifying/authenticating users based on new input.
- Storing and managing feature embeddings for each user.
- Powered by FastAPI and PyTorch, using GRU-based neural network models for biometric analysis.

## Features

- **Sensor Data Authentication:** Supports embedding and matching of mobile device sensor data (accelerometer, gyroscope, etc.).
- **Keypress Dynamics Authentication:** Models typing patterns for behavioral biometric verification.
- **Enrollment & Verification:** REST endpoints for user enrollment and verification.
- **Embeddings Management:** Efficient storage, pruning, and loading of biometric feature vectors.
- **CORS Support:** Ready for integration with web and mobile clients.

## Directory Structure

```
├── main.py                     # Main FastAPI server (sensor + keypress API)
├── extra.py                    # Alternate FastAPI server focusing on keypress APIs
├── models.py                   # Model definitions (KeypressGRU, SensorNetwork)
├── utils.py                    # Helper utilities and schemas
├── sensor_model.pt             # Pretrained sensor model weights
├── siamese_gru_model.pth       # Pretrained keypress model weights
├── KEY_EMBEDDINGS/             # Directory for keypress user embeddings
├── sensor_embeddings/          # Directory for sensor user embeddings
└── __pycache__/                # Python cache files
```

## Setup

1. **Install requirements**

   ```bash
   pip install fastapi torch numpy pydantic starlette scipy
   ```

2. **Model Weights**

   Ensure `sensor_model.pt` and `siamese_gru_model.pth` are present in the root directory.

3. **Run the Server**

   - For the main (sensor + keypress) API:
     ```bash
     uvicorn main:app --reload
     ```
   - For keypress-only API:
     ```bash
     uvicorn extra:app --reload
     ```

## Usage

- Register (`enroll`) users with their behavioral data.
- Verify user identity using new sensor/keypress input.
- Embeddings are managed per user and pruned to maintain history.

## API Endpoints

### Sensor APIs (main.py)

- `GET /check_user/{user_id}`  
  Check if sensor embeddings exist for a user.

- `POST /receive`  
  Upload and embed new sensor data for a user.

- `POST /authenticate`  
  Authenticate user based on new sensor data.

- `GET /health`  
  Health check for the sensor model server.

### Keypress APIs (main.py & extra.py)

- `POST /enroll/{user_id}`  
  Enroll a user with their keypress event stream.

- `POST /verify/{user_id}`  
  Verify a user's identity via new keypress data.

- `GET /health`  
  Health check for the keypress model server.

## Models

- **KeypressGRU:** GRU-based Siamese neural network for modeling temporal patterns in keypress events.
- **SensorNetwork:** Feedforward neural network for aggregating sensor windows into embeddings.

Model weights are loaded at server startup. The device (CPU or CUDA) is auto-detected.


## License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Contact**: For questions or issues, please use the repository's issue tracker.