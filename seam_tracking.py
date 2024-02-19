from flask import Flask, render_template, request, Response,url_for
import cv2
import time
#import pytesseract
import json,os
import numpy as np
from multiprocessing import Process, Queue
from flask import jsonify
import torch
from stacked_hourglass.utils.imutils import resize
from stacked_hourglass.utils.transforms import color_normalize, fliplr, flip_back
from stacked_hourglass.utils.evaluation import final_preds_untransformed


import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import cv2
import torch.nn.functional as F
from torchvision import  models, transforms
import torchvision

RGB_MEAN=torch.as_tensor([0.2787472093235083,0.2787472093235083,0.2787472093235083])
RGB_STDDEV=torch.as_tensor([0.24897527225582264,0.24897527225582264,0.24897527225582264])
sizee=256

h,w=224,224
mean = [0.15854794283031065, 0.15854794283031065, 0.15854794283031065]
std = [0.22935350279045194, 0.22935350279045194, 0.22935350279045194]

def _check_batched(images):
    if isinstance(images, (tuple, list)):
        return True
    if images.ndimension() == 4:
        return True
    return False


def check_list(words):
    # Check if the list contains any item other than 'normal'
    other_items = [word for word in words if word != 'normal']

    if other_items:
        # Return all items if there are other items than 'normal'
        return other_items
    else:
        # Return ['normal'] if there are no other items
        return ['normal']

class HumanPosePredictor:
    def __init__(self, model, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        model.to(device)
        self.model = model
        self.device = device

    def do_forward(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

    def prepare_image(self, image):
        was_fixed_point = not image.is_floating_point()
        image = torch.empty(image.shape, device='cpu', dtype=torch.float32).copy_(image)
        if was_fixed_point:
            image /= 255.0
        if image.shape[-2:] != (sizee,sizee):
            image = resize(image, sizee,sizee)
        image = color_normalize(image, RGB_MEAN, RGB_STDDEV)
        return image

    def estimate_heatmaps(self, images, flip=False):
        is_batched = _check_batched(images)
        raw_images = images if is_batched else images.unsqueeze(0)
        input_tensor = torch.empty((len(raw_images), 3, sizee,sizee),
                                   device=self.device, dtype=torch.float32)
        for i, raw_image in enumerate(raw_images):
            input_tensor[i] = self.prepare_image(raw_image)
        heatmaps = self.do_forward(input_tensor)[-1].cpu()
        if flip:
            flip_input = fliplr(input_tensor.cpu().clone().numpy())
            flip_input = torch.as_tensor(flip_input, device=self.device, dtype=torch.float32)
            flip_heatmaps = self.do_forward(flip_input)[-1].cpu()
            heatmaps += flip_back(flip_heatmaps)
            heatmaps /= 2
        if is_batched:
            return heatmaps
        else:
            return heatmaps[0]

    def estimate_joints(self, images, flip=False):
        """Estimate human joint locations from input images.

        Images are expected to be centred on a human subject and scaled reasonably.

        Args:
            images: The images to estimate joint locations for. Can be a single image or a list
                    of images.
            flip (bool): If set to true, evaluates on flipped versions of the images as well and
                         averages the results.

        Returns:
            The predicted human joint locations.
        """
        is_batched = _check_batched(images)
        raw_images = images if is_batched else images.unsqueeze(0)
        heatmaps = self.estimate_heatmaps(raw_images, flip=flip).cpu()
        coords = final_preds_untransformed(heatmaps, (64,64))
        # Rescale coords to pixel space of specified images.
        for i, image in enumerate(raw_images):
            coords[i, :, 0] *= image.shape[-1] / 64
            coords[i, :, 1] *= image.shape[-2] / 64
        if is_batched:
            return coords
        else:
            return coords[0]
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


app = Flask(__name__)
frame_queue = Queue(maxsize=10)  # Adjust maxsize as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@app.route('/')
def index():
    return render_template('seam_tracking.html')
ALLOWED_EXTENSIONS = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


PATH = "entire_model.pt"
model = torch.load(PATH, map_location=device)
model.eval()

def estimate_pose(frame):
    
    frame1 = np.moveaxis(frame, -1, 0)
    frame2 = torch.as_tensor(np.array(frame1, copy=True))
    predictor = HumanPosePredictor(model, device=device)
    joints = predictor.estimate_joints(frame2, flip=False)
    
    return joints             


def generate_frames_seam(video_path, speed_multiplier=1.0):

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = int((1.0 / frame_rate) * 1000 / speed_multiplier)
    i=0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if not frame_queue.full():
            frame1=frame[:88,:,:]
            frame_queue.put(frame1.copy())  # Put a copy of the frame into the queue for OCR
            
        if i % 1 == 0:
            frame=frame[88:,400:-128,:]
            frame=cv2.resize(frame,(sizee,sizee),interpolation=cv2.INTER_NEAREST)
            #frame=cv2.flip(frame,1)
            pose_info = estimate_pose(frame)
            # Draw pose on the frame
            for point in pose_info:
                frame = cv2.circle(frame, (int(point[0]), int(point[1])), radius=5, color=(0, 0, 255), thickness=-1)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(frame_delay / 1000.0)
        i+=1

defects=['lop','lof','por','u','c']
models=[]
for i,defect in enumerate(defects):
    PATH=defect+'.pth'
    checkpoint=torch.load(PATH)
    models.append(torchvision.models.resnet18(pretrained=True)) # Using pre-trained for demo purpose

    for parameter in models[i].parameters():
            parameter.requires_grad = False 

    #feat_out=resnet50.fc.out_features
    num_feat=models[i].fc.in_features
    #model.fc=nn.Sequential(
    #     nn.Dropout(dropout_rate),
    #     nn.Linear(model.fc.in_features, 2)

    models[i].fc=nn.Linear(num_feat,2)

    models[i].load_state_dict(checkpoint['model_state_dict'])

    models[i].to(device)
    models[i].eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(h),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

def generate_frames_defect(video_path, speed_multiplier=1.0):

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = int((1.0 / frame_rate) * 1000 / speed_multiplier)
    i=0

    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # org 
    org = (100,200) 
    
    # fontScale 
    fontScale = 5
    
    # Blue color in BGR 
    color = (255, 255, 255) 
    
    # Line thickness of 2 px 
    thickness = 3

    while True:
        success, frame1 = cap.read()
        if not success:
            break

        frame = preprocess(frame1)
        # Add batch dimension (model expects batches of images)
        frame = frame.unsqueeze(0)
        
        # Transfer to the device
        frame = frame.to(device)

        outputs=[]
        for i in range(len(models)):
            output = models[i](frame)
            _, pred = torch.max(output, 1)
            outputs.append(int(pred.item()==0)*'normal'+int(pred.item()==1)*defects[i])
            final_output=check_list(outputs)

            for item in final_output:
                final_outputs=int(item=='c')*'Cold Lap_' + int(item=='u')*'Undercut_' + int(item=='por')*'Porosity_' + int(item=='lop')*'LOP_' + int(item=='lof')*'LOF_' + int(item=='n' or item=='normal')*'normal_' 

        
        # Using cv2.putText() method 
        image = cv2.putText(frame1, final_outputs, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 



        _, buffer = cv2.imencode('.jpg', image)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(frame_delay / 1000.0)
        i+=1

# Declare a variable to store the last uploaded video path
last_uploaded_video_path_seam, last_uploaded_video_path_defect = None, None

@app.route('/upload_seam', methods=['POST'])
def upload_seam():
    global last_uploaded_video_path_seam
    if 'video' not in request.files:
        return "No file part"
    file = request.files['video']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = "uploaded_video.mp4"
        file_path = os.path.join('static/videos', filename)
        file.save(file_path)
        last_uploaded_video_path_seam = file_path
        return render_template('seam_tracking.html', video_name=filename)
    return "Invalid file type"


@app.route('/upload_defect', methods=['POST'])
def upload_defect():
    global last_uploaded_video_path_defect
    if 'video' not in request.files:
        return "No file part"
    file = request.files['video']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = "uploaded_video.mp4"
        file_path = os.path.join('static/videos', filename)
        file.save(file_path)
        last_uploaded_video_path_defect = file_path
        return render_template('seam_tracking.html', video_name=filename)
    return "Invalid file type"


@app.route('/video_feed_seam')
def video_feed_seam():
    global last_uploaded_video_path_seam
    return Response(generate_frames_seam(last_uploaded_video_path_seam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_defect')
def video_feed_defect():
    global last_uploaded_video_path_defect
    return Response(generate_frames_defect(last_uploaded_video_path_defect),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


speed_multiplier = 1.0
@app.route('/set_speed/<float:speed>', methods=['GET'])
def set_speed(speed):
    global speed_multiplier
    speed_multiplier = speed
    return "Speed set to " + str(speed_multiplier)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
