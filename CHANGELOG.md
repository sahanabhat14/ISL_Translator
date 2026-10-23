# Changelog

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
