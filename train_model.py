"""Run --check first. Train an isolated-sign LSTM on valid recordings."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from dataset import SIGNS, FRAMES, FEATURES, load_dataset

ROOT = Path(__file__).resolve().parent


def split_indices(labels, seed=42):
    """Split whole clips per class, never adjacent frames."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        held = max(1, int(len(indices) * 0.15))
        test.extend(indices[:held])
        val.extend(indices[held:2 * held])
        train.extend(indices[2 * held:])
    return [np.asarray(part, dtype=int) for part in (train, val, test)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    x, y, paths, rejected = load_dataset(args.data)
    for reason in rejected:
        print(f"REJECTED: {reason}")
    counts = {sign: int((y == i).sum()) for i, sign in enumerate(SIGNS)}
    print("Valid recordings:", counts)
    if not all(count >= 30 for count in counts.values()):
        print("Not ready: collect at least 30 valid clips for EACH sign.")
        return 1
    if args.check:
        print("Ready for an initial training experiment.")
        return 0
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is missing. See README.md for the separate training environment.")
        return 1
    tf.keras.utils.set_random_seed(args.seed)
    train, val, test = split_indices(y, args.seed)
    normalization = tf.keras.layers.Normalization(axis=-1)
    normalization.adapt(x[train])
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(FRAMES, FEATURES)), normalization,
        tf.keras.layers.LSTM(64), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(SIGNS), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x[train], y[train], validation_data=(x[val], y[val]),
                        epochs=args.epochs, batch_size=16,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=12, restore_best_weights=True)])
    loss, accuracy = model.evaluate(x[test], y[test], verbose=0)
    predictions = model.predict(x[test], verbose=0).argmax(axis=1)
    confusion = np.zeros((len(SIGNS), len(SIGNS)), dtype=int)
    np.add.at(confusion, (y[test], predictions), 1)
    precision = np.diag(confusion) / np.maximum(confusion.sum(axis=0), 1)
    recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    output = ROOT / "models" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    model.save(output / "isl_lstm.keras")
    report = dict(labels=SIGNS, input_shape=[FRAMES, FEATURES], seed=args.seed,
                  feature_order="Left xyz (63), Right xyz (63); mirrored camera; missing hand = zeros",
                  normalization="training-set feature standardization embedded in model",
                  test_loss=float(loss), test_accuracy=float(accuracy),
                  precision=precision.tolist(), recall=recall.tolist(),
                  confusion_matrix=confusion.tolist(), history=history.history,
                  split={name: [paths[i] for i in indices] for name, indices in
                         zip(["train", "validation", "test"], [train, val, test])},
                  limitation="Random clip split, not signer-independent evaluation. Hands-only baseline.")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Test accuracy: {accuracy:.1%}. Saved model and report to {output}")
    print("Small same-session test sets do not establish accuracy on new signers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
