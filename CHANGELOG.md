# Changelog

## Unreleased — Streamlit app

- Added selected-label collection and resumable targets for NO/NO_SIGN data improvement.
- Added fresh-clone setup, collaborator workflow and mentor review PDF.
- Known issue: NO/NO_SIGN confusion remains; no retrained model is included.
- Compiled model inference: measured median 127.07 ms to 2.39 ms, with matching saved-test predictions.
- Faster JPEG preview updates, independent of recognition sampling.
- Restored the recorder's 50 ms pause after hand processing and native camera resolution to reduce differences between training and app inputs.
- Live performance readings for capture, hand tracking and prediction.
- Isolated TensorFlow from Streamlit's Keras cleanup to prevent model stalls during reruns.
- Preserved sign windows across tracking gaps shorter than 0.2 seconds.

- Fixed preview freezing during model startup or prediction by separating camera capture.
- Stop can release capture while model loading continues in the background.
- Added regression checks for deliberately blocked loading and slow predictions.

- Double-click Windows launcher with clear instructions to keep the local server running.

- Simplified layout with fixed camera area, compact recognition display and settings expander.
- Controls stay outside periodic updates; idle pages no longer rerun continuously.
- Opens the camera before model loading, reports startup stages and tries a Windows camera fallback.
- Checked actual live frame updates, hand detection, audio test, reset and Stop in a browser.
- Local browser interface with camera Start/Stop, recognized words and recent word history.
- Speech toggle, audio test, camera selection and reset controls.
- Background capture with shared landmark format, LSTM and smoothing from v1.
- Camera resources are released on Stop, errors or browser inactivity.
- Uses the existing v1.0.0 model; its recorded evaluation remains unchanged.

## v1.0.0 — 2026-10-23

First GitHub release of the working Windows webcam prototype.

- Five signs: HELLO, THANK_YOU, YES, NO and BYE.
- Silent NO_SIGN class for visible resting hands and ordinary movement.
- Shared two-hand features, 30-frame sequences, LSTM classification and prediction smoothing.
- Offline Windows speech, repetition suppression and audio test controls.
- Resumable collection with quality checks and a reproducible training script.
- Included trained model and report with repository-relative dataset references.

Model evaluation: 22/24 held-out clips correct (91.7%), including 4/4 NO_SIGN
clips. One THANK_YOU became HELLO; one NO became NO_SIGN. Random clip splitting
does not establish generalization across signers or recording sessions.
Live text and speech were reported working by the user before this release;
live NO_SIGN accuracy has not been formally measured.
