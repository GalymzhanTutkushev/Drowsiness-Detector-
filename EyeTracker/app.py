from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
from eye_catcher import EyeCatcher

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
socketio = SocketIO(app, async_mode='threading')

auto_id = False
reset_btn_pressed = False
names = []
name_list_changed = True
new_name_list_emitted = False
extract_features_state = False
addDB_Name = ""


@app.route('/')
def main():
    return render_template('main.html')

@socketio.on('auto_id_check')
def auto_id_check(isChecked):
    global auto_id
    if isChecked == 1:
        auto_id = True
    else:
        auto_id = False

@socketio.on('reset_pressed')
def reset_pressed():
    global reset_btn_pressed
    reset_btn_pressed = True

@socketio.on('database_display')
def database_display():
    global new_name_list_emitted
    if not new_name_list_emitted:
        new_name_list_emitted = True
        socketio.emit('display_event', names)

@socketio.on('extract_features')
def extract_features(name):
    global extract_features_state, addDB_Name
    extract_features_state = True
    addDB_Name = name

def gen(tracker):
    while True:
        global reset_btn_pressed
        global names, name_list_changed, extract_features_state
        if name_list_changed:
            for name in tracker.personNames:
                names.append(name)
            name_list_changed = False
        if extract_features_state:
            tracker.person_state = 32
            extract_features_state = False
        if tracker.person_state == 16:  # identification state
            frame = tracker.identification_state()
        elif tracker.person_state == 32:
            frame = tracker.createNewFeatures(addDB_Name)
            if tracker.person_state == 0:
                names = []
                for name in tracker.personNames:
                    names.append(name)
                socketio.emit('display_event', names)
                tracker.person_state = 16
        else:
            frame = tracker.analyze_frame(reset_btn_pressed, auto_id)
            reset_btn_pressed = False
        
        socketio.emit('my_event', {'message': tracker.person_state,
                                    'name': tracker.current_person_name,
                                    'idframe': tracker.identification_frame_num,
                                    'exframe': tracker.feature_extraction_num,
                                    'opencnt': tracker.feature_extraction_counter_open,
                                    'maxcnt': tracker.feature_extraction_counter})
        ret, frame = cv2.imencode('.jpg', frame)
        frame = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(EyeCatcher()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=False)
