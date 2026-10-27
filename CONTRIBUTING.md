# Collaborating

Clone the repository and follow the Windows setup in README.md. The webcam asset must be downloaded separately; recordings, environments and experimental models are not in Git.

## Branch and pull request workflow

Start with a clean working directory. Update main, then create a descriptive branch:

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/your-change
```

Make one focused change, run the relevant checks, and inspect your diff. Stage specific files, commit and push:

```powershell
git diff
git add path/to/changed-file
git commit -m "Describe the change"
git push -u origin feature/your-change
```

Open a pull request to main on GitHub. Describe the problem, what changed, how you checked it, and known limitations. Ask another teammate to review before merging. Coordinate changes to shared files rather than overwriting each other's work. Never force-push main.

## Data and model work

- Keep raw recordings, virtual environments, credentials and temporary render files out of commits.
- Agree on sign definitions, feature order and recording cadence before collecting. Changes to the feature schema require compatible data and retraining.
- Share dataset access separately with the owner's agreement. A code clone does not include data/.
- Save candidate models in separate run directories. Document label order, split, per-class results, and live trials; do not claim generalization from same-session clips.
- Preserve the v1.0.0 tag and model. Agree on a new version only after testing. Never move an existing release tag.
- The current NO/NO_SIGN confusion is unresolved; do not make NO_SIGN speak as a shortcut.

Only the repository owner can invite collaborators through GitHub Settings. Accept the invitation before pushing. If branch protection is configured, follow its review/check requirements.
