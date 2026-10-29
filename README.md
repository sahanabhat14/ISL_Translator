# ISL Translator

Windows webcam prototype for **15 isolated signs**: HELLO, THANK_YOU, YES, NO, BYE, DRINK, EAT, COME, GO, PLEASE, SLEEP, WRITE, HELP, SORRY and STOP, plus a silent NO_SIGN class.

The latest candidate is bundled in `models/15_signs/`. The original v1.0.0 model/tag is preserved. The goal is 20 meaningful signs; this is not continuous sentence translation.

Run the latest candidate after the setup below:

```powershell
.\.venv-train\Scripts\python.exe recognize.py --model-dir .\models\15_signs
```

Camera text displays the best candidate immediately, even below the confidence threshold. Spoken words require confirmation. Use `--debug` to see scores. The latest internal result is **61/64 = 95.3%** on a small random clip split, not independent-user accuracy.

**Streamlit still defaults to the older v1.0.0 model**. Use the explicit CLI command above for all 15 signs; model selection in the browser is pending.

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

Two complete model/report pairs are included: `models/v1.0.0/` (5 signs + background) and `models/15_signs/` (15 signs + background). Labels are always read from the selected report.

- Input: 30 frames x 126 values (two hands x 21 landmarks x xyz).
- Mirrored frames; Left hand first, Right second; absent hands zero-filled.
- Training-only normalization, LSTM(64), Dropout(0.3), Dense(32, ReLU), softmax output matching the report labels.
- Adam, sparse categorical cross-entropy, batch size 16, validation-loss early stopping.
- Latest dataset: 480 valid clips, 30/class; 352 training, 64 validation, 64 test.
- Latest saved test result: 61/64 (95.3125%). Errors: NO to NO_SIGN, GO to EAT, SLEEP to THANK_YOU.
- Earlier v1 result: 22/24 (91.7%) on its different six-class test set. These scores are not a controlled comparison.

This small random clip split is not signer/session-independent evaluation. It does not establish live accuracy or performance on new people. The repository does not contain the private recording dataset, so a fresh clone cannot reproduce that evaluation without separately obtaining the data. Tests use synthetic input when original test clips are absent.

## Current development and known issues

Camera capture runs independently of recognition. MediaPipe and compiled TensorFlow inference share a persistent local worker process to avoid Streamlit rerun cleanup interfering with model state. Prediction confirmation requires three agreeing windows above a 0.8 threshold; predictions are attempted every five valid frames once the 30-frame buffer is full. This confidence score is not calibrated reliability.

The browser preview uses a separate loopback-only MJPEG stream with a per-session URL. Display copies are resized to at most 640 pixels and JPEG-encoded once per captured frame, while the Streamlit status panel refreshes every 0.3 seconds. Recognition retains the original image; preview frames are not queued or recorded. The local stream closes on Stop. This preview works on the same Windows computer as the server.

The app now retains native camera resolution and waits 50 ms after recognition processing to approximate the collector's cadence; the preview runs independently. Very short tracking gaps under 0.2 seconds are tolerated. Longer gaps clear the sequence. Timing depends on hardware and detection quality.

Numeric historical measurements are in `results/`. Model-only median prediction dropped from about 127 ms eager to 2.4 ms compiled on the development laptop, with matching saved-test outputs. This excludes gesture collection and tracking. Historical tracking rates predate the latest timing correction; they are not a current sign-to-word latency guarantee.

Remaining work: investigate confused pairs, test all 15 signs live and during transitions, collect independent sessions/users, integrate explicit browser model selection, and add five more meaningful signs. The background class cannot reject every unfamiliar gesture. Hands-only features do not capture body pose or facial grammar.

## Collect and train

Raw landmark recordings under `data/` are ignored by Git. Collection resumes toward a total number of valid clips without overwriting earlier files:

```powershell
.\.venv-train\Scripts\python.exe collect_data.py --signs SLEEP WRITE HELP SORRY STOP --target 30 --wait-ready
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

Training requires at least 30 valid clips for every class and saves a separate timestamped `models/run_*` directory. It does not replace the bundled default automatically. Compare per-class errors and independent live trials before changing the default model. Keep v1 intact. The 15-sign candidate is trained and bundled; future collection/training does not replace that model automatically. --wait-ready pauses for Space before each new sign.

Original CLI alternative:

```powershell
.\.venv-train\Scripts\python.exe recognize.py --model-dir .\models\15_signs
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
| inference_worker.py / preview_stream.py | Persistent model/tracker process and local video preview |
| start_app.bat | Windows local app launcher |

See [CONTRIBUTING.md](CONTRIBUTING.md) for the team workflow and [CHANGELOG.md](CHANGELOG.md) for changes. The mentor report is in `output/pdf/ISL_Translator_Mentor_Review.pdf`; it describes project status as of 26 October 2026, rather than subsequent changes.
