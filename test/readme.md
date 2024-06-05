# XTTS Streaming Server

## Introduction
This repository contains the XTTS Streaming Server, which allows you to use cloned voices for text-to-speech (TTS) applications.

## Getting Started
To use a cloned voice, you need to follow these steps:

1. Run the `clone_speaker.py` script to create a JSON file for the cloned voice. You must first modify to give  This script takes input from the user to generate the necessary voice data.

    ```bash
    python clone_speaker.py
    ```

    This will generate a `cloned_voice.json` file.

2. In the `test_streaming.py` file, update the `reference_voice` variable to point to the `cloned_voice.json` file.

    ```python
    reference_voice = "cloned_voice.json"
    ```

    This will ensure that the XTTS Streaming Server uses the cloned voice for TTS.

## Usage
Once you have completed the setup steps, you can start the XTTS Streaming Server by running the following command:
