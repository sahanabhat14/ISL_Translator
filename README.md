# ISL Translator training starter

## Current model with a silent background class

The default live model is now `models/run_20261022_232911_931283/`.
It was trained on 180 valid clips (132 training, 24 validation, 24 test),
stopped after 46 epochs and restored epoch 34. Reload verification reproduced
22/24 correct test predictions (91.7%). All four NO_SIGN test clips were correct.
One THANK_YOU became HELLO and one NO became NO_SIGN. The latter would suppress
speech, so explicitly test NO when trying the new model. These small random
clip results do not establish generalization to new sessions/signers.
The earlier five-sign model remains available via
`--model-dir models/run_20261022_232038_450902`.

Collection and training now include NO_SIGN alongside the five signs. Run
`.\.venv\Scripts\python.exe collect_data.py` to collect 30 valid NO_SIGN clips;
completed sign folders are skipped. Keep at least one hand detected throughout
each clip. Vary relaxed hand positions, small natural movements and ordinary
repositioning without performing HELLO, THANK_YOU, YES, NO or BYE. Avoid making
every background clip the same pose. The 80% hand-visibility check still applies:
empty frames are already handled by the live detector and are not useful examples
of visible non-sign activity.

The six-class model has been trained, verified and selected for live use.
The recognition code displays a confirmed NO_SIGN as
"No sign", suppresses speech for that class, clears pending speech when it is
the leading prediction, and rearms repeated-sign speech after confirmation.
An utterance already playing may finish. This class can reduce false positives
but cannot guarantee rejection of every unfamiliar gesture. Test resting hands
and transitions live; reduced false positives are not established by the clip test.

## Previous five-sign model

The previous webcam model recognizes HELLO, THANK_YOU, YES, NO and BYE.
Run `.\.venv-train\Scripts\python.exe recognize.py` to test all five with speech.
Training used 150 valid clips, split into 110 training, 20 validation and 20 test
clips. The run stopped after 26 epochs and restored epoch 14. The saved model
correctly classified 19/20 test clips (95%); one THANK_YOU was classified as NO.
Each class had four test clips. Reload verification reproduced the confusion matrix.
These random clip results do not establish accuracy on new signers or sessions.

Model and report: `models/run_20261022_232038_450902/`.
The original three-sign model remains in `models/run_20261019_232551_504610/`.

## Collection target

Collection and the next training run now use five signs: HELLO, THANK_YOU,
YES, NO and BYE. The existing three signs already have 30 valid clips each.
Run `.\.venv\Scripts\python.exe collect_data.py` to collect 30 valid NO clips
and 30 valid BYE clips. Completed signs are skipped, rejected clips are
retried, and existing files are preserved. Verify the ISL gestures before recording.

Then run `.\.venv-train\Scripts\python.exe train_model.py --check`.
Training requires 30 valid clips for all five signs and creates a separate model
with all five output classes. The current live default remains the tested
five-sign model after training and reload verification. To use a different run,
pass `--model-dir models/run_...` to recognize.py. Live label order always comes
from that model's report, not the collection vocabulary.

The sections below document the original three-sign baseline and its results.

The first experiment recognizes HELLO, THANK_YOU and YES from 30 frames of
two-hand landmarks (126 numbers per frame). This is a hands-only isolated-word
baseline for the larger MediaPipe + LSTM/GRU project in the synopsis and slides.
It does not yet implement pose/face landmarks, continuous translation, speech,
or the planned 20–40 sign vocabulary.

## 1. Collect usable examples

Run these PowerShell commands from this project directory:

```powershell
.\.venv\Scripts\python.exe collect_data.py
```

Use verified ISL gestures consistently. Watch the **Hands detected** counter:
it should show the expected number of hands. Position the camera so both hands
stay visible, with good lighting. Perform one complete sign during each
30-frame recording, then return to the starting position during the countdown.
Capture waits 50 ms per frame in addition to processing time, so a clip lasts
at least 1.5 seconds. Future inference must use the same capture cadence.
Press Q in the camera window to stop.

The collector resumes toward 30 valid clips per sign and saves with unique names,
so rerunning preserves earlier examples. Rejected clips are retried. It rejects clips with hands detected
in fewer than 80% of frames. This is a basic quality check, not proof that the
gesture or both required hands were captured correctly. Repeat collection if
there are rejections. Vary repetitions naturally and avoid duplicate clips.

The two original HELLO recordings were inspected on 19 October 2026 and both
contained only zeros. They remain on disk for inspection but training rejects
them. Do not count them as useful examples.

```powershell
.\.venv\Scripts\python.exe train_model.py --check
```

Collect at least 30 valid examples **for each sign** before the first experiment.
This is a pipeline milestone, not evidence of sufficient data for deployment.

## 2. Prepare the training environment

The existing .venv uses Python 3.14 and handles collection. A separate
.venv-train with Python 3.12 and TensorFlow 2.21.0 has now been installed and
verified locally with a synthetic LSTM training step. To train once data is ready:

```powershell
.\.venv-train\Scripts\python.exe train_model.py
```

For setup on another computer, install Python 3.12 if needed and create a
separate environment without replacing .venv:

```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv-train\Scripts\python.exe train_model.py
```

TensorFlow Windows build compatibility:
https://www.tensorflow.org/install/source_windows

## 3. Understand the result

The script splits whole recordings by class into training, validation and test
sets (approximately 70/15/15). It fits feature standardization on training data
only and embeds it in the saved model. This standardization does not provide
the shoulder-relative spatial normalization described in the proposal.
An LSTM learns temporal patterns, and early stopping restores the weights with
the best validation loss. The test set is evaluated after training.

Each run creates a new models/run_* directory containing isl_lstm.keras and
report.json with label order, input format, split membership, learning curves,
test accuracy, per-class precision/recall and a confusion matrix. Matrix rows
are true classes and columns are predicted classes, in the saved label order.

This initial random clip split may share signer and session conditions across
sets. Before reporting generalization, record signer/session identifiers and
hold out entire signers or sessions. Do not repeatedly tune on test results.
The current classifier always chooses among three signs; confidence alone does
not establish that an unseen gesture is a known sign.

## 4. Try live recognition

```powershell
.\.venv-train\Scripts\python.exe recognize.py
```

Use the training environment for recognition. For a fresh installation, install
`requirements-live.txt` there first. The script defaults to the verified first
training run; use `--model-dir models/run_...` to select another complete run.
`--check` loads the model, runs saved test clips and initializes the hand detector
without opening a webcam. Use `--camera 1` if your camera has a different index.

Sign HELLO, THANK_YOU or YES at the same pace and position as collection. The
first result takes about two seconds or more: 30 frames plus agreement over
three overlapping prediction windows. Q quits; R clears the buffer. Removing
your hands clears the prediction and starts a fresh sequence when they return.
Brief hand-detection gaps also reset the buffer, so keep hands visible.

Collection and inference share the same feature extraction (mirrored camera,
Left then Right, missing hands zero-filled). Normalization is inside the model.
Both use a 50 ms display wait plus processing; live inference adds overhead,
so this is approximate pacing, not a fixed frame rate. Three consecutive
windows must exceed the default 0.8 score threshold and agree before display.
This threshold is a starting heuristic, not a calibrated reliability measure.
Visible resting hands and unknown gestures can still get confident labels.

Speech is enabled by default and runs on a background thread using the Windows
offline voice using direct Windows SAPI calls, avoiding repeated-utterance cancellation in
the previous pyttsx3 implementation. Press T to test audio without signing
(this explicit test also works while muted). The terminal logs `Speaking:`
when an utterance starts. Each stable sign is spoken once. Another sign may be spoken
after a two-second cooldown. To repeat the same sign, remove your hands for at
least one second or press R, then sign again. Brief tracking gaps do not rearm
speech. Press M to mute/unmute or launch with `--mute`. Muting clears pending
speech; an utterance already playing may finish. Voice initialization failures
leave text recognition running and show speech as unavailable.

Test each sign several times, then test resting hands and transitions. Record
mistakes and whether predictions take too long. No new webcam recordings are
saved by this script. Next milestones are background examples, evaluation on
new sessions/signers and expanding vocabulary/landmarks. Adding
pose/face features changes the input schema and requires new training data.

## Verification status

Dataset rejection and clip-split logic are checked locally. TensorFlow 2.21.0
loads and runs a synthetic LSTM training step in .venv-train. The latest dataset
check found 30 valid clips for each sign. The first training run completed:

- Model: `models/run_20261019_232551_504610/isl_lstm.keras`
- Report: `models/run_20261019_232551_504610/report.json`
- Split: 66 training, 12 validation, 12 test clips.
- Early stopping: 27 epochs, restored epoch 15.
- Test: 12/12 correct, four clips per sign. Reloaded model reproduced this result.

This small random clip test does not measure performance on new signers or
sessions. Live webcam recognition is implemented; interactive accuracy still
requires testing by a signer at the webcam.
The user subsequently confirmed live recognition worked and reported correct
predictions during their informal test, with occasional delay. No exact trial
counts or timed latency measurements were supplied; this is qualitative feedback.
Keras saving stalled in Windows sandbox temporary-file creation; rerunning
with approved access outside the sandbox completed successfully.
