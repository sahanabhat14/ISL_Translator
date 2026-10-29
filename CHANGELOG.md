# Changelog

## 2026-10-29 - 15-sign candidate update

- Added DRINK, EAT, COME, GO, PLEASE, SLEEP, WRITE, HELP, SORRY and STOP to the original five signs; NO_SIGN remains separate.
- Bundled the 16-class candidate in models/15_signs with a portable evaluation report; preserved v1.0.0.
- Latest internal test: 61/64 (95.3%); no independent signer/session evaluation is claimed.
- Added Space-to-start collection, compiled CLI inference and optional confidence diagnostics.
- Restored the original camera overlay; it displays the candidate immediately while speech still requires confirmation.
- Streamlit continues to default to v1.0.0; the latest model is selected explicitly through the CLI.


## Unreleased — Streamlit app

- Moved live preview out of Streamlit image reruns into a loopback-only latest-frame MJPEG stream; status updates remain separate.
- Display copies are resized without changing model input resolution or sampling cadence.
- Added preview delivery, input-preservation and shutdown checks.

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
