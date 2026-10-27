"""Local app: python -m streamlit run app.py"""
import html
import json
import threading

import streamlit as st
from app_runtime import CameraSession
from recognize import DEFAULT_RUN, ROOT


def main():
    st.set_page_config(page_title="ISL Translator", layout="wide", initial_sidebar_state="collapsed")
    st.markdown("""<style>
    header[data-testid="stHeader"] { background:transparent; height:0; }
    [data-testid="stToolbar"] { display:none; }
    .stApp { background:#f6f5f1; color:#202725; }
    .block-container { max-width:1140px; padding-top:1.7rem; padding-bottom:1rem; }
    h1 { font-size:1.85rem !important; font-weight:650 !important; letter-spacing:-.04em; }
    h3 { font-size:1rem !important; font-weight:600 !important; }
    [data-testid="stButton"] button { border-radius:5px; min-height:2.6rem; }
    .camera-idle { aspect-ratio:4/3; background:#202829; color:#e4e8e6;
      display:flex; align-items:center; justify-content:center; flex-direction:column; border-radius:5px; }
    .camera-idle strong { font-size:1.1rem; font-weight:500; }
    .camera-idle span { color:#a8b4b0; font-size:.9rem; margin-top:.5rem; }
    .result-label { color:#65706b; font-size:.8rem; margin:1.1rem 0 .5rem; }
    .result-word { color:#202725; font-size:clamp(2rem,4vw,3.4rem); font-weight:600;
      letter-spacing:-.04em; line-height:1.15; margin:0 0 1rem; }
    [data-testid="stImage"] img { border-radius:5px; }
    </style>""", unsafe_allow_html=True)


    @st.cache_resource
    def camera_lock():
        return threading.Lock()


    if "camera_session" not in st.session_state:
        st.session_state.camera_session = None
    if "speech_enabled" not in st.session_state:
        st.session_state.speech_enabled = True


    def start_camera():
        current = st.session_state.camera_session
        if current is not None and current.thread.is_alive():
            return
        session = CameraSession(camera_lock(), DEFAULT_RUN,
                                camera=st.session_state.get("camera_number", 0),
                                speech=st.session_state.speech_enabled)
        st.session_state.camera_session = session
        session.start()


    def stop_camera():
        session = st.session_state.camera_session
        if session is not None:
            session.stop()


    def reset_camera():
        session = st.session_state.camera_session
        if session is not None:
            session.reset_event.set()


    def change_speech():
        session = st.session_state.camera_session
        if session is not None:
            session.set_speech(st.session_state.speech_enabled)


    def test_audio():
        session = st.session_state.camera_session
        if session is not None and session.thread.is_alive():
            session.test_audio.set()


    st.title("ISL Translator")
    st.caption("Webcam to text and speech")
    try:
        report = json.loads((DEFAULT_RUN / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        st.error(f"Cannot read the model: {error}")
        st.stop()
    missing = [str(p.relative_to(ROOT)) for p in
               [DEFAULT_RUN / "isl_lstm.keras", ROOT / "models/hand_landmarker.task"] if not p.exists()]
    if missing:
        st.error("Missing model files: " + ", ".join(missing))
        st.stop()
    signs = {label.replace("_", " ") for label in report["labels"] if label != "NO_SIGN"}

    # Keep controls outside the timed preview so updates do not interrupt clicks.
    controls = st.columns([1, .8, 1, 1.5, 1.3])
    controls[0].button("Start camera", type="primary", on_click=start_camera, width="stretch")
    controls[1].button("Stop", on_click=stop_camera, width="stretch")
    controls[2].button("Clear / reset", on_click=reset_camera, width="stretch")
    with controls[3]:
        st.toggle("Speech", key="speech_enabled", on_change=change_speech)
    controls[4].button("Test audio", on_click=test_audio, width="stretch")

    with st.expander("Settings and supported signs"):
        st.selectbox("Camera", [0, 1, 2], key="camera_number",
                     help="Stop the camera before switching. Camera 0 is usually the built-in webcam.")
        st.write("Supported: " + ", ".join(label.replace("_", " ") for label in report["labels"] if label != "NO_SIGN"))
        st.caption("Keep your hands visible. No sign stays silent. Use Stop when finished. "
                   "The app uses the camera and speakers on this Windows laptop.")

    session = st.session_state.camera_session
    refresh = 0.1 if session is not None and session.thread.is_alive() else None


    @st.fragment(run_every=refresh)
    def live_view():
        current = st.session_state.camera_session
        if current is None:
            snapshot = dict(frame=None, status="Camera off", error=None, hands=0,
                            speech="Off", history=[], frames=0)
        else:
            snapshot = current.snapshot()
        preview, output = st.columns([1.65, 1], gap="large")
        with preview:
            if snapshot["frame"] is not None:
                st.image(snapshot["frame"], channels="RGB", width="stretch", output_format="JPEG")
            else:
                message = "Camera is off" if current is None or not current.thread.is_alive() else snapshot["status"]
                st.markdown('<div class="camera-idle"><strong>' + html.escape(message) +
                            '</strong><span>Use Start camera above.</span></div>', unsafe_allow_html=True)
        with output:
            status = snapshot["status"]
            word = status.capitalize() if status in signs else "—"
            st.markdown('<p class="result-label">RECOGNIZED WORD</p><p class="result-word">' +
                        html.escape(word) + '</p>', unsafe_allow_html=True)
            st.write(status if status not in signs else "Confirmed")
            st.caption(f"Hands: {snapshot['hands']}  /  Speech: {snapshot['speech']}")
            if snapshot.get("ready"):
                with st.expander("Performance"):
                    st.write(f"Camera: {snapshot.get('camera_fps', 0):.1f} frames/s")
                    st.write(f"Hand tracking: {snapshot.get('tracking_fps', 0):.1f} frames/s")
                    st.write(f"Last prediction: {snapshot.get('prediction_ms', 0):.1f} ms")
            if snapshot["error"]:
                st.error(snapshot["error"])
            st.divider()
            st.subheader("History")
            st.write(" · ".join(snapshot["history"]) if snapshot["history"] else "No words yet.")
        if current is not None and refresh is not None and not current.thread.is_alive():
            st.rerun()


    live_view()


if __name__ == "__main__":
    main()
