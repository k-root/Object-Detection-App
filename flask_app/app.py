
import os
import zipfile


from ObjDetectTrain import runner
from ObjDetectTest import detect
import ObjDetectTestVideo
from os.path import dirname
from flask import Flask, request, redirect, url_for, flash, render_template, send_from_directory
from werkzeug.utils import secure_filename
import shutil
from genericpath import exists

AUTO_DETECTION_DIR = dirname(os.getcwd())
CORE_DIR = os.path.join(AUTO_DETECTION_DIR,'core')
MODELS_DIR = os.path.join(AUTO_DETECTION_DIR,'models')

ALLOWED_EXTENSIONS = set(['zip'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/models')
def list_models():
    all_models = os.listdir(MODELS_DIR)
    return render_template('models.html',all_models = all_models)


@app.route('/train_model', methods = ['GET','POST'])
def train_model():
    if request.method == 'POST':
        model_name = request.form['modelName']
        if not model_name :
            print("model name is not given")
        print(model_name)
        file = request.files['file']
        if file.filename == '':
            print("file not given")
        print(file.filename)
        if file and allowed_file(file.filename):
            print(file.filename)
            filename = secure_filename(file.filename)
            if not os.path.exists(os.path.join(dirname(os.getcwd()),'models')):
                os.makedirs(os.path.join(dirname(os.getcwd()),'models'))
            if not os.path.exists(os.path.join(MODELS_DIR,model_name)):
                os.mkdir(os.path.join(MODELS_DIR,model_name))
            UPLOAD_FOLDER = os.path.join(MODELS_DIR,model_name)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
            zip_ref.extractall(UPLOAD_FOLDER)
            zip_ref.close()
            if os.path.exists(os.path.join(UPLOAD_FOLDER, 'dataset')):
                shutil.rmtree(os.path.join(UPLOAD_FOLDER, 'dataset'))
            os.rename(os.path.join(UPLOAD_FOLDER, filename.replace('zip','')),os.path.join(UPLOAD_FOLDER, 'dataset'))
            runner(['','--imdir',os.path.join(UPLOAD_FOLDER,'dataset/images'),'--dir',os.path.join(UPLOAD_FOLDER,'dataset/annotations'),'--model',model_name])            
    return render_template('train.html')


@app.route('/predect/<string:model_name>',methods=['GET', 'POST'])
def predect(model_name):
    if request.method == 'POST':
        f = request.files['file']
        model_loc = os.path.join(MODELS_DIR,model_name)
        img_name = secure_filename(f.filename)
        if not os.path.exists(os.path.join(model_loc, 'test_images')):
            os.mkdir(os.path.join(model_loc,'test_images'))
        img_save_loc = os.path.join(model_loc, 'test_images')
        f.save(os.path.join(img_save_loc,img_name))
        # try:
        #     os.remove(MODELS_DIR + '/' + model_name + '/' + 'input.jpg')
        # except OSError:
        #     pass
        # os.rename(name,MODELS_DIR + '/' + model_name + '/' + 'input.jpg')
        # p = subprocess.Popen(['python','object_detection_exec.py','input.jpg'],cwd=MODELS_DIR + '/' + model_name)
        # p.wait()
        # try:
        #     os.remove('static/output.jpg')
        # except OSError:
        #     pass
        # os.rename(MODELS_DIR + '/' + model_name + '/' + 'output.jpg','static/output.jpg')
        detect(os.path.join(img_save_loc,img_name), model_name)
        
        return  redirect(url_for('output',model_name = model_name, img_name = img_name))
    return render_template('predect.html',model_name = model_name)


@app.route('/predect_video/<string:model_name>', methods=['GET', 'POST'])
def predect_video(model_name):
    if request.method == 'POST':
        f = request.files['file']
        model_loc = os.path.join(MODELS_DIR, model_name)
        vid_name = secure_filename(f.filename)
        if not os.path.exists(os.path.join(model_loc, 'test_videos')):
            os.mkdir(os.path.join(model_loc, 'test_videos'))
        vid_save_loc = os.path.join(model_loc, 'test_videos')
        f.save(os.path.join(vid_save_loc, vid_name))
        # try:
        #     os.remove(MODELS_DIR + '/' + model_name + '/' + 'input.jpg')
        # except OSError:
        #     pass
        # os.rename(name,MODELS_DIR + '/' + model_name + '/' + 'input.jpg')
        # p = subprocess.Popen(['python','object_detection_exec.py','input.jpg'],cwd=MODELS_DIR + '/' + model_name)
        # p.wait()
        # try:
        #     os.remove('static/output.jpg')
        # except OSError:
        #     pass
        # os.rename(MODELS_DIR + '/' + model_name + '/' + 'output.jpg','static/output.jpg')
        ObjDetectTestVideo.detect(os.path.join(vid_save_loc, vid_name), model_name)

        return redirect(url_for('output_video', model_name=model_name, img_name='test_output.mp4'))
    return render_template('predect_video.html', model_name=model_name)


@app.route('/output_video/<string:model_name>/<string:img_name>')
def output_video(model_name,img_name):
    return render_template('output_video.html',img_name = img_name,model_name = model_name)

@app.route('/output/<string:model_name>/<string:img_name>')
def output(model_name,img_name):
    return render_template('output.html',img_name = img_name,model_name = model_name)

@app.route('/download/<string:model_name>')
def download(model_name):
    model_path = os.path.join(dirname(os.getcwd()), 'models')
    
    if not os.path.exists(os.path.join(dirname(os.getcwd()), 'Downloads')):
        os.makedirs(os.path.join(dirname(os.getcwd()), 'Downloads'))
    
    downloadsFolder = os.path.join(dirname(os.getcwd()), 'Downloads')
    with zipfile.ZipFile(dirname(os.getcwd())+'/Downloads/' + model_name + '.zip', 'w', zipfile.ZIP_DEFLATED) as myzip:
        #for f in [model_path+'/'+model_name+'/object_detection_graph/frozen_inference_graph.pb', model_path+'/'+model_name+'/pet_label_map.pbtxt']:   
        myzip.write(model_path+'/'+model_name+'/object_detection_graph/frozen_inference_graph.pb', arcname='frozen_inference_graph.pb' )
        myzip.write(model_path+'/'+model_name+'/pet_label_map.pbtxt', arcname='pet_label_map.pbtxt')
    return send_from_directory(dirname(os.getcwd())+'/Downloads', filename = model_name+'.zip')

@app.route('/runlabelImg/')
def runLabelImg():
    os.system(dirname(os.getcwd())+'\labelImg.exe')
    return redirect(url_for('train_model'))
if __name__ == "__main__":
    app.run(debug=True)

