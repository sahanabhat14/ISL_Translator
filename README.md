# ISL Translator

Windows webcam prototype for five isolated signs: HELLO, THANK_YOU, YES, NO and BYE, plus a silent NO_SIGN background class. Uses MediaPipe hand landmarks, a TensorFlow LSTM, Streamlit and offline Windows speech.

**Status:** v1.0.0 is the initial tagged prototype. The current main branch adds a local app and targeted data collection as work in progress. NO can be confused with NO_SIGN, causing silence. This is not continuous sign-language or sentence translation, and the planned 30 signs are not implemented.

## Start on a new Windows computer

Install Python 3.12 and Git, then run in PowerShell:

```powershell
git clone https://github.com/sahanabhat14/sign_launguage_thing.git
cd sign_launguage_thing
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install -r requirements-app.txt
```

Download the MediaPipe Hand Landmarker asset from Google's model storage. It is not included in Git:

```powershell
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile 'models/hand_landmarker.task'
Get-FileHash models/hand_landmarker.task -Algorithm SHA256
```

Expected SHA256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.
Do not run a mismatching asset; retry the download. The project uses MediaPipe 1.1.0, TensorFlow 2.21.0, opencv-contrib-python 5.0.0.93, Streamlit 1.66.1 and pywin32 312, as pinned in the requirement files. Speech uses Windows SAPI, so this setup targets Windows.

Launch by double-clicking `start_app.bat`, or run:

```powershell
.\.venv-train\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none
```

Open the address shown, normally http://127.0.0.1:8501/. Keep the server terminal open. After changing source code, stop the server with Ctrl+C, restart it and refresh the browser. The launcher reuses an existing server; it does not reload edited code automatically.

Close other camera apps, press Start camera, and wait for model loading. Use Clear/reset before a fresh attempt. The speech toggle controls sign announcements; Test audio explicitly plays a phrase even while announcements are muted. Stop releases capture. Removing hands or a confirmed NO_SIGN can rearm repeated-sign speech. NO_SIGN intentionally stays silent; the meaningful NO sign should speak "No".

The app uses the webcam and speakers of the computer running Python. Preview images and word history stay in memory; the app does not save webcam images. It is a local interface, not a deployed multi-user website.

## Model and evaluation

The default model and report are bundled under `models/v1.0.0/`. There are six output classes, in the order saved in the report.

- Input: 30 frames x 126 features (two hands x 21 landmarks x xyz).
- Mirrored frames; MediaPipe Left features first, Right second; missing hands are zeros.
- Training-only feature normalization embedded in the model, LSTM(64), Dropout(0.3), Dense(32, ReLU), Dense(6, softmax).
- Adam, sparse categorical cross-entropy, batch size 16, early stopping on validation loss.
- Initial 180 valid clips: 132 training, 24 validation, 24 test, split by whole clip with seed 42.
- Saved test result: 22/24 correct (91.7%). Errors: THANK_YOU to HELLO and NO to NO_SIGN.

This small random clip split is not signer/session-independent evaluation. It does not establish live accuracy or performance on new people. The repository does not contain the private recording dataset, so a fresh clone cannot reproduce that evaluation without separately obtaining the data. Tests use synthetic input when original test clips are absent.

## Current development and known issues

Camera capture runs independently of recognition. MediaPipe and compiled TensorFlow inference share a persistent local worker process to avoid Streamlit rerun cleanup interfering with model state. Prediction confirmation requires three agreeing windows above a 0.8 threshold; predictions are attempted every five valid frames once the 30-frame buffer is full. This confidence score is not calibrated reliability.

The app now retains native camera resolution and waits 50 ms after recognition processing to approximate the collector's cadence; the preview runs independently. Very short tracking gaps under 0.2 seconds are tolerated. Longer gaps clear the sequence. Timing depends on hardware and detection quality.

Numeric historical measurements are in `results/`. Model-only median prediction dropped from about 127 ms eager to 2.4 ms compiled on the development laptop, with matching saved-test outputs. This excludes gesture collection and tracking. Historical tracking rates predate the latest timing correction; they are not a current sign-to-word latency guarantee.

Remaining work: improve NO/NO_SIGN separation, verify all five signs live, test resting hands/transitions, collect independent sessions/users, and then expand vocabulary. The background class cannot reject every unfamiliar gesture. Hands-only features do not capture body pose or facial grammar.

## Collect and train

Raw landmark recordings under `data/` are ignored by Git. Collection resumes toward a total number of valid clips without overwriting earlier files:

```powershell
.\.venv-train\Scripts\python.exe collect_data.py
```

For the current NO/NO_SIGN issue, target 60 total valid clips for each selected class:

```powershell
.\.venv-train\Scripts\python.exe collect_data.py --signs NO NO_SIGN --target 60
```

With 30 existing clips, this requests 30 more per selected class. Keep hands visible, use verified ISL gestures, and perform one complete sign during each recording. For NO_SIGN, rest or move naturally without performing a trained sign. A clip is 30 frames, with a 50 ms wait plus processing per frame; it is not a fixed-duration video. Validation requires finite values, shape (30,126), and hands visible in at least 80% of frames. Q stops collection; rerun to resume.

```powershell
.\.venv-train\Scripts\python.exe train_model.py --check
.\.venv-train\Scripts\python.exe train_model.py
```

Training requires at least 30 valid clips for every class and saves a separate timestamped `models/run_*` directory. It does not replace the bundled default automatically. Compare per-class errors and independent live trials before changing the default model. Keep v1 intact. Extra collection and retraining have not been confirmed complete in this update.

Original CLI alternative:

```powershell
.\.venv-train\Scripts\python.exe recognize.py
```

Q quits; R resets; M mutes; T tests speech. Use `--model-dir models/run_YOUR_RUN` to select another complete model/report pair. Do not open the CLI camera and app camera together.

## Checks

```powershell
.\.venv-train\Scripts\python.exe -m unittest discover -s tests -v
```

Checks cover UI controls, resource cleanup, silent background behavior, responsiveness during blocked loading/prediction, and model/worker execution. They do not replace real gesture accuracy testing. Model-dependent tests need the downloaded MediaPipe asset.

## Source map

| File | Responsibility |
| --- | --- |
| collect_data.py / dataset.py | Collection, vocabulary, validation and loading |
| hand_features.py | Shared feature ordering |
| train_model.py | Training and evaluation reports |
| recognize.py / speech.py | CLI recognition, confirmation and offline speech |
| app.py / app_runtime.py | Local UI, camera and recognition orchestration |
| inference_worker.py | Persistent model/tracker process |
| start_app.bat | Windows local app launcher |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the team workflow and [CHANGELOG.md](CHANGELOG.md) for changes. The mentor report is in `output/pdf/ISL_Translator_Mentor_Review.pdf`; it describes project status as of 26 October 2026, rather than subsequent changes.
