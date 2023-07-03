import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from struct import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.ops.knn import knn_points

from multiprocessing import Pool, Process, Queue
from multiprocessing.pool import ThreadPool
import subprocess

from functools import partial
import signal
import pymeshlab
from scipy import ndimage
import mediapipe as mp
import glob
import png
import shutil
cudnn.benchmark = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3,4'

device = torch.device("cuda:0")

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')



parser.add_argument('--testpath', default="./data/face_demo",help='testing data dir for some scenes')
#no "/" last

parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='mdi_melungl', help='select dataset')
parser.add_argument('--testlist', default="lists/our_list.txt", help='testing scene list')

parser.add_argument('--frontfaceidx', default=[1,2,3,4,5], help='Camera Index for making 3d landmark')
parser.add_argument('--landmark', default=True, help='extract landmark')
parser.add_argument('--faceseg', default=False, help='Face segmentation')

parser.add_argument('--facealign', default=False, help='Face coordinate alignment up -y forward +z')
parser.add_argument('--facecut', default=True, help='Face Mesh cutting bellow chin and only mediapipe face mesh')
parser.add_argument('--facempcut', default=True, help='Face Mesh cutting with mediapipe face mesh')
parser.add_argument('--flamefit', default=True, help='Flame fitting using Mp cut mesh')
parser.add_argument('--flameimgidx', default=[0,1,2,3,4,5,6,7,8,13], help='Camera Index for face optimization')


parser.add_argument('--face_opt', default=True, help='Face optimization -Lee Jeonghaeng')
parser.add_argument('--landmarks_projection', default=True, help='Project Landmarks to Mesh and get the Vertex indices for each Lanmarks')



parser.add_argument('--local_rank',help='testing data dir for some scenes')

parser.add_argument('--batch_size', type=int, default=4, help='testing batch size')#7 GPUs can use maximum 18 batches +- 2
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')#192 no significant performance improvement even if increased

parser.add_argument('--loadckpt', default="../casmvsnet.ckpt", help='load a specific checkpoint')


parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')#48 32 8 no significant performance improvement even if increased
parser.add_argument('--depth_inter_r', type=str, default="4,8,2", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')


parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')#2 is blur, 10 is too much
parser.add_argument('--max_h', type=int, default=512, help='testing max h')#864
parser.add_argument('--max_w', type=int, default=512, help='testing max w')#1152


parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=16, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=60, help='save freq of local pcd')
parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")

#filter

#filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='/hdd1/lhs/dev/code/Github_melungl/fusibile/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.9')
parser.add_argument('--disp_threshold', type=float, default='1.5')
parser.add_argument('--num_consistent', type=float, default='3')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

Interval_Scale = args.interval_scale

##############################################################################################
def template_opt_init3(n_pos1, n_pos2):
    n_p1 = n_pos1
    n_p2 = n_pos2
    n_p2_result = torch.zeros_like(n_pos2)

    # lmk_valid_center = [6, 197, 195, 5, 4, 1, 19, 94, 2, 20, 242, 141, 370, 462, 250, 245, 122, 351, 465]
    lmk_valid_center = [6, 197, 195, 5, 4, 1, 19, 94, 2, 20, 242]
    lmk_valid_up = [67, 109, 10, 338, 297]
    lmk_valid_right = [162, 127, 234, 93, 132]
    lmk_valid_left = [389, 356, 454, 323, 361]
    lmk_valid = lmk_valid_center + lmk_valid_up + lmk_valid_right + lmk_valid_left
    print(len(lmk_valid))

    R_init = torch.eye(3, device=device, requires_grad=True)
    T_init = (n_p1[4] - n_pos2[4]).detach().clone()
    T_init.requires_grad = True
    S_init = torch.tensor(1.0, device=device, requires_grad=True)

    optimizer_init = torch.optim.Adam([R_init, T_init, S_init], lr=0.01)

    epoch_init = 500
    print("Optimization Start!")
    for i in range(epoch_init):
        optimizer_init.zero_grad()
        # n_p1_deform = S_init * torch.matmul(R_init, (n_p1 + T_init).t()).t()
        n_p2_deform = S_init * torch.matmul(R_init, (n_p2 + T_init).t()).t()

        # loss = torch.sum((n_p2 - n_p1_deform) ** 2)
        loss = torch.sum((n_p1[lmk_valid] - n_p2_deform[lmk_valid]) ** 2)
        loss.backward()
        optimizer_init.step()
        print(i)
    print("R :",R_init)
    return R_init

def rotation_matrix_from_direction(direction):
    # Normalize the direction vector
    direction = direction / torch.norm(direction)
    
    # Define the reference vectors
    up = torch.tensor([0, 1, 0], dtype=torch.float32)
    forward = torch.tensor([0, 0, 1], dtype=torch.float32)
    
    # Calculate the rotation axis and angle
    rotation_axis = torch.cross(up, direction)
    rotation_angle = torch.acos(torch.dot(up, direction))
    
    # Create the rotation matrix
    rotation_matrix = torch.tensor([
        [torch.cos(rotation_angle) + rotation_axis[0]**2 * (1 - torch.cos(rotation_angle)),
         rotation_axis[0] * rotation_axis[1] * (1 - torch.cos(rotation_angle)) - rotation_axis[2] * torch.sin(rotation_angle),
         rotation_axis[0] * rotation_axis[2] * (1 - torch.cos(rotation_angle)) + rotation_axis[1] * torch.sin(rotation_angle)],
        [rotation_axis[0] * rotation_axis[1] * (1 - torch.cos(rotation_angle)) + rotation_axis[2] * torch.sin(rotation_angle),
         torch.cos(rotation_angle) + rotation_axis[1]**2 * (1 - torch.cos(rotation_angle)),
         rotation_axis[1] * rotation_axis[2] * (1 - torch.cos(rotation_angle)) - rotation_axis[0] * torch.sin(rotation_angle)],
        [rotation_axis[0] * rotation_axis[2] * (1 - torch.cos(rotation_angle)) - rotation_axis[1] * torch.sin(rotation_angle),
         rotation_axis[1] * rotation_axis[2] * (1 - torch.cos(rotation_angle)) + rotation_axis[0] * torch.sin(rotation_angle),
         torch.cos(rotation_angle) + rotation_axis[2]**2 * (1 - torch.cos(rotation_angle))]
    ], dtype=torch.float32)
    
    return rotation_matrix
def calculate_perpendicular_direction(points):
    # Convert the points to a PyTorch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32)

    # Calculate the centroid of the points
    centroid = torch.mean(points_tensor, dim=0)
    
    # Compute the covariance matrix
    centered_points = points_tensor - centroid
    covariance_matrix = torch.matmul(centered_points.t(), centered_points)
    
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
    
    # Sort the eigenvalues and eigenvectors in descending order
    eigenvalues = eigenvalues[:, 0]
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Get the eigenvector corresponding to the smallest eigenvalue
    perpendicular_direction = eigenvectors[:, -1]
    
    return perpendicular_direction
def write_2d_landmark(landmark_filename, lmk_array, landmark_image):
    np.savez(landmark_filename, lmk=lmk_array)
    cv2.imwrite(landmark_filename[:-3]+'.png', landmark_image)
    
    

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics

# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5

# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(model, test_dataset):
    # dataset, dataloader  
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)

            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            end_time = time.time()
            print('only inference: ' +str(end_time-start_time))
            outputs = tensor2numpy(outputs)

            del sample_cuda
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            # save depth maps and confidence maps
            if args.landmark:
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) 
            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, \
                                                            outputs["depth"], outputs["photometric_confidence"]):

                img = img[0]  #ref view
                cam = cam[0]  #ref cam
                depth_filename = os.path.join(args.testpath + '_output', (filename[:-11] + filename[-10:]).format('2333__', '/disp.dmb'))
                normal_filename = os.path.join(args.testpath + '_output', (filename[:-11] + filename[-10:]).format('2333__', '/normals.dmb'))
                landmark_filename = os.path.join(args.testpath + '_output', filename.format('2d_landmark', '.npz'))
                
                cam_filename = os.path.join(args.testpath + '_output', filename.format('cams', '.jpg.P'))
                img_filename = os.path.join(args.testpath + '_output', filename.format('images', '.jpg'))
                
                os.makedirs(depth_filename[:-8], exist_ok=True)
                os.makedirs(cam_filename[:-14], exist_ok=True)
                os.makedirs(img_filename[:-12], exist_ok=True)
                if args.landmark:
                    os.makedirs(landmark_filename[:-12], exist_ok=True)
                
                #save depth maps
                save_pfm(depth_filename+'.pfm', depth_est)
                #save confidence maps
                #save_pfm(confidence_filename, photometric_confidence)
                #save cams, img
                depth_est[photometric_confidence < args.prob_threshold] = 0
                write_gipuma_dmb(depth_filename, depth_est)
                
                fake_gipuma_normal(cam, normal_filename, depth_est)
                
                write_gipuma_cam(cam, cam_filename)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)
                
                if args.landmark:
                    image_height, image_width, _ = img.shape
                    #print(img.shape)
                    img_landmark = img
                    results = face_mesh.process(img_landmark)
                    img_landmark = cv2.cvtColor(img_landmark, cv2.COLOR_RGB2BGR)
                    if not results.multi_face_landmarks:
                        # print("no detected data - ", file)
                        continue
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    lmk_array = np.zeros((len(face_landmarks), 3))
                    black_img = np.zeros_like((img),dtype = np.uint8).copy()
                    #print(np.shape(img))
                    #print(np.shape(black_img))
                    for f_idx in range(len(face_landmarks)):
                        lmk_array[f_idx, 0] = face_landmarks[f_idx].x * image_width
                        lmk_array[f_idx, 1] = face_landmarks[f_idx].y * image_height
                        lmk_array[f_idx, 2] = face_landmarks[f_idx].z
                        
                    for f_idx in range(len(face_landmarks)):    
                        landmark_image = cv2.circle(img_landmark,(int(lmk_array[f_idx,0]),int(lmk_array[f_idx,1])),1,(255,255,255),-1)
                        
                        black_img = cv2.circle(black_img,(int(lmk_array[f_idx,0]),int(lmk_array[f_idx,1])),1,(255,255,255),-1)
                        for s_f_idx in range(len(face_landmarks)):
                            black_img = cv2.line(black_img,(int(lmk_array[f_idx,0]),int(lmk_array[f_idx,1])),(int(lmk_array[s_f_idx,0]),int(lmk_array[s_f_idx,1])),(255,255,255),1)
                            
                            
                    img_gray = cv2.cvtColor(black_img,cv2.COLOR_BGR2GRAY)
                    res, thr = cv2.threshold(img_gray, 50,255,cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thr,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = np.vstack(contours)
                    for idx in contours:
                        hull = cv2.convexHull(idx, clockwise = True)
                        cv2.drawContours(black_img,[hull],0,(255,255,255),-1)
                    
                        
                    write_2d_landmark(landmark_filename, lmk_array, landmark_image)
                    
                    
                    
                    img_filename = os.path.join(args.testpath, filename.format('images', '.png'))
                    img = cv2.imread(img_filename)
                    black_img = cv2.resize(black_img,(0,0),fx = 2,fy =2, interpolation=cv2.INTER_NEAREST)
                    res = cv2.bitwise_and(img,img,mask = black_img[:,:,0])

                    cv2.imwrite(landmark_filename[:-3]+'_mask.png', res)
                
                
                

    gc.collect()
    torch.cuda.empty_cache()
    
def write_gipuma_cam(cam, out_path):
    '''convert mvsnet camera to gipuma camera format'''
    #cam[0] extrinsic cam[1] intrinsic

    projection_matrix = np.matmul(cam[1], cam[0])
    projection_matrix = projection_matrix[0:3][:]

    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    f = open(out_path+'camera-intrinsics.txt', "w")
    for i in range(0, 3):
        for j in range(0, 3):
            if j ==2:
                f.write(str(cam[1][i][j]))
            else:
                f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()
    
    f = open(out_path+'frame-000000.pose.txt', "w")
    for i in range(0, 4):
        for j in range(0, 4):
            if j ==3:
                f.write(str(cam[0][i][j]/1000))
            else:
                f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()


    return


def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''

    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = image.squeeze()
        #print(image)

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    np.save(path+".npy", image)
    
    if np.shape(image)[-1] !=3:
        image = image.astype(np.uint16)
        #print(image)
        with open(path+'.png', 'wb') as f:
            writer = png.Writer(width=image.shape[1], height=image.shape[0], bitdepth=16,
                                greyscale=True)
            zgray2list = image.tolist()
            writer.write(f, zgray2list)

    return



def gradient(im_smooth):

    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.arange(-1,2).astype(float)
    kernel = - kernel / 2

    gradient_x = ndimage.convolve(gradient_x, kernel[np.newaxis])
    gradient_y = ndimage.convolve(gradient_y, kernel[np.newaxis].T)

    return gradient_x,gradient_y


def sobel(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    gradient_x = ndimage.convolve(gradient_x, kernel)
    gradient_y = ndimage.convolve(gradient_y, kernel.T)

    return gradient_x,gradient_y


def compute_normal_map(gradient_x, gradient_y, intensity=1):

    width = gradient_x.shape[1]
    height = gradient_x.shape[0]
    max_x = np.max(gradient_x)
    max_y = np.max(gradient_y)

    max_value = max_x

    if max_y > max_x:
        max_value = max_y

    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    intensity = 1 / intensity

    #print(max_value)
    strength = max_value / (max_value * intensity)

    normal_map[..., 0] = gradient_x / max_value
    normal_map[..., 1] = gradient_y / max_value

    normal_map[..., 2] = 1 / strength

    norm = np.sqrt(np.power(normal_map[..., 0], 2) + np.power(normal_map[..., 1], 2) + np.power(normal_map[..., 2], 2))

    normal_map[..., 0] /= norm
    normal_map[..., 1] /= norm
    normal_map[..., 2] /= norm


    return normal_map



def Convert(im,intensity):

    sobel_x, sobel_y = sobel(im)

    normal_map = compute_normal_map(sobel_x, sobel_y, intensity)

    return normal_map


def fake_gipuma_normal(cam, path, depth_image):
    image_shape = np.shape(depth_image)
    h, w = np.shape(depth_image)
    extrinsic = cam[0]
    
    a = True
    
    if a == True:
        #tt = time.time()
        normals = Convert(depth_image,1000)
        #ee = time.time()
        #print("noraml time : ", str(ee-tt))

        normal_image = normals
        
        # for x in range(h):
        #     for y in range(w):
        #         normal_image[x,y,:] = -np.matmul(extrinsic[:3,:3].T,normals[x,y,:])
        # eee = time.time()
        # print("noraml r time : ", str(eee-ee))        
    else:
        fake_normal = np.array(-extrinsic[2,:3])
        
        normal_image = np.ones_like(depth_image) #depth
        normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1)) #one channel depth
        normal_image = np.tile(normal_image, [1, 1, 3])

        normal_image[:, :, 0] = fake_normal[0]
        normal_image[:, :, 1] = fake_normal[1]
        normal_image[:, :, 2] = fake_normal[2]


    normal_image = np.float32(normal_image)
    write_gipuma_dmb(path, normal_image)
    
    #########################################
    # normal_image *= 0.5
    # normal_image += 0.5
    # normal_image *= 255
    # normals *= 0.5
    # normals += 0.5
    # normals *= 255
    
    # normal_image.astype('uint8')
    # normals.astype('uint8')
    # tt = time.time()
    # normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("/root/lhs/normal/normal_g"+ str(tt) +".png", normal_image)
    # normals = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("/root/lhs/normal/normal_o"+ str(tt) +".png", normals)
    ##########################################
    
    return

def obj_read_w_color(obj_path):
    vs = []
    fs = []
    cs = []
    with open(obj_path, 'r') as obj_f:
        lines = obj_f.readlines()
        for line in lines:
            line_ = line.split(' ')
            if len(line_) < 4:
                line_ = line.split('\t')
            if line_[0] == 'v':
                v1 = float(line_[1])
                v2 = float(line_[2])
                v3 = float(line_[3])
                vs.append([v1, v2, v3])
                c1 = float(line_[4])
                c2 = float(line_[5])
                c3 = float(line_[6])
                cs.append([c1, c2, c3])
            if line_[0] == 'f':
                f1 = int(line_[1].split('/')[0]) - 1
                f2 = int(line_[2].split('/')[0]) - 1
                f3 = int(line_[3].split('/')[0]) - 1
                fs.append([f1, f2, f3])
    vs = np.array(vs)
    fs = np.array(fs)
    cs = np.array(cs)

    return vs, fs, cs  


def Extract_3d_landmark(lmk_2d_folder, used_view_list):
    full_lmk_2d_list = sorted(glob.glob(lmk_2d_folder+'/*.npz'))
    lmk_2d_list = [full_lmk_2d_list[i] for i in used_view_list]
    lmks_2d = np.zeros((len(lmk_2d_list), 478, 2))
     
    for i, lmk_name in enumerate(lmk_2d_list):
        lmk2d = np.load(lmk_name)
        lmks_2d[i,:,:] = lmk2d['lmk'][...,:2]
    
    
    A = proj_matricies[None, :, 2:3].repeat(n_lmk, 1, 2, 1) * lmk_2d[None].transpose(0, 2, 1, 3).reshape(
            n_lmk, n_view, 2, 1)
    A -= proj_matricies[None, :, :2].tile(n_lmk, 1, 1, 1)

    # A = A.detach().cpu().numpy()
    u, s, vh = np.linalg.svd(A.reshape(n_lmk, n_view * 2, 4))
    point_3d_homo = -vh[:, 3, :]

    # lmk_3d : n_lmk x 3
    lmk_3d = homogeneous_to_euclidean(point_3d_homo)
    #print(np.shape(lmk_3d))
    # np.savez(os.path.join(args.lmk_dir, '3D_lmk/lmk_3D_%03d.npz' % i), lmk=lmk_3d)

    if i == 0:
        lmk_3d_tr = tr.tensor(lmk_3d).float().to(args.device)
        knn = knn_points(lmk_3d_tr[None], v.float())

        lmk_idx = knn[1].squeeze(0).squeeze(-1).detach().cpu().numpy()
        lmk_3d2 = v[0, lmk_idx, :].detach().cpu().numpy()

        np.savez(os.path.join(args.lmk_dir, 'lmk_ref_with_idx.npz'), lmk_idx=lmk_idx, lmk=lmk_3d)
        np.savez(os.path.join(args.lmk_dir, 'lmk_3D_%03d.npz' % i), lmk=lmk_3d)

        if args.visualize:
            lmk_3d = tr.tensor(lmk_3d).to(args.device)
            lmk_3d2 = tr.tensor(lmk_3d2).to(args.device)
            # print(lmk_3d.shape,lmk_3d2.shape)
            point_cloud1 = Pointclouds(points=[lmk_3d],
                                        features=[tr.ones_like(lmk_3d) * tr.tensor([[1, 0, 0]]).to(args.device)])
            point_cloud2 = Pointclouds(points=v, features=tr.ones_like(v) * tr.tensor([[0, 1, 0]]).to(args.device))
            point_cloud3 = Pointclouds(points=[lmk_3d2], features=[tr.ones_like(lmk_3d2) * tr.tensor([[0, 0, 1]]).to(args.device)])
            # render both in the same plot in different traces
            fig = plot_scene({
                "Pointcloud": {
                    "lmk_3d": point_cloud1,
                        "v": point_cloud2,
                    "lmk_3d_mapping": point_cloud3
                }
            })
            fig.show()
            #exit()
    return lmk_3d

def pc2obj_poisson(scan_folder, depth, threads, source, target, filtered_result):
    
    cmd = "PoissonRecon --in " + scan_folder + source +" --out " + scan_folder + target + " --depth " + str(depth)+ " --threads " + str(threads)
    os.system(cmd)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(scan_folder + target)
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag = pymeshlab.Percentage(20))
    ms.save_current_mesh(scan_folder + filtered_result)
    print("Saved " + filtered_result)
    return ms

def cut_face_under_neck(ms, point_folder, scan_folder, target):
    lmk_3d = np.load(os.path.join(point_folder,'lmk_3d.npz'))
    lmk_3d = lmk_3d['lmk']
    chin_pos = lmk_3d[152,1]

    ms.compute_selection_by_condition_per_vertex(condselect="y>{}".format(chin_pos+10)) 
    ms.meshing_remove_selected_vertices()
    ms.save_current_mesh(scan_folder + target)
    
    return ms


def compute_mvs_RT(R_world, new_origin, cam_list, source, target, tt):
    R = np.zeros((len(cam_list),3,3))
    T = np.zeros((len(cam_list),3,1))
    K = np.zeros((len(cam_list),3,3))
    RTT = np.zeros((len(cam_list),4,4))
    for i, filename in enumerate(cam_list):
        Ki, RT = read_camera_parameters(filename)
        R[i,:,:] = RT[:3,:3]
        T[i,:,0] = RT[:3,3]
        K[i,:,:] = Ki[:,:]
        RTT[i,:,:] = RT[:,:]
    
    R_world_inv = np.linalg.inv(R_world)
    T_world_inv = -np.matmul(R_world_inv, np.array([0,0,0])).astype(np.float64)
    T_world_inv2 = -np.matmul(np.identity(3), new_origin).astype(np.float64)
    for i, filename in enumerate(cam_list): 

        # Define the transformation matrix that maps points from the new world coordinate system to the camera coordinate system
        M_world_to_cam = np.array([
            [R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], T[i, 0, 0]],
            [R[i, 1, 0], R[i, 1, 1], R[i, 1, 2], T[i, 1, 0]],
            [R[i, 2, 0], R[i, 2, 1], R[i, 2, 2], T[i, 2, 0]],
            [0, 0, 0, 1]
        ])
        M_new_world_to_cam = np.array([
            [R_world[0, 0], R_world[0, 1], R_world[0, 2], T_world_inv[0]],
            [R_world[1, 0], R_world[1, 1], R_world[1, 2], T_world_inv[1]],
            [R_world[2, 0], R_world[2, 1], R_world[2, 2], T_world_inv[2]],
            [0, 0, 0, 1]
        ])
        
        M_new_cam_to_world = np.linalg.inv(M_new_world_to_cam.astype(np.float64))
        M_new_world_to_cam = np.matmul(M_world_to_cam, M_new_cam_to_world)
        # Extract the rotation matrix R_new, translation vector T_new, and intrinsic camera matrix K_new from the new transformation matrix
        R_new = M_new_world_to_cam[:3,:3]
        T_new = M_new_world_to_cam[:3, 3:]
        
        
        M_world_to_cam = np.array([
            [R_new[0, 0], R_new[0, 1], R_new[0, 2], T_new[0, 0]],
            [R_new[1, 0], R_new[1, 1], R_new[1, 2], T_new[1, 0]],
            [R_new[2, 0], R_new[2, 1], R_new[2, 2], T_new[2, 0]],
            [0, 0, 0, 1]
        ])
        M_new_world_to_cam = np.array([
            [1, 0, 0, T_world_inv2[0]],
            [0, 1, 0, T_world_inv2[1]],
            [0, 0, 1, T_world_inv2[2]],
            [0, 0, 0, 1]
        ])
        
        M_new_cam_to_world = np.linalg.inv(M_new_world_to_cam.astype(np.float64))
        M_new_world_to_cam = np.matmul(M_world_to_cam, M_new_cam_to_world)
        # Extract the rotation matrix R_new, translation vector T_new, and intrinsic camera matrix K_new from the new transformation matrix
        R_new = M_new_world_to_cam[:3, :3]
        T_new = M_new_world_to_cam[:3, 3:]
        
        
        
        K_new = K[i,:,:].copy()

        
        cam = np.zeros((2,4,4))
        cam[0,:3,:3] = R_new
        cam[1,:3,:3] = K_new
        cam[0,:3, 3] = T_new.T
        cam[0, 3, 3] = 1
        
        os.makedirs(filename[:-(17+tt)]+target, exist_ok= True)
        newfile = filename.replace(source, target)
        f = open(newfile, "w")
        f.write('extrinsic\n')
        for i in range(0, 4):
            for j in range(0, 4):
                f.write(str(np.round(cam[0][i][j],7)) + ' ')
            f.write('\n')
        f.write('\n')

        f.write('intrinsic\n')
        for i in range(0, 3):
            for j in range(0, 3):
                f.write(str(cam[1][i][j]) + ' ')
            f.write('\n')

        f.write('\n' + str(900) + ' ' + str(1.5))

        f.close()
    return cam

def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent, testpath, scene):
    cam_folder = os.path.join(point_folder, 'cams')
    image_folder = os.path.join(point_folder, 'images')
    
    scan_folder = os.path.join(point_folder, '3D_Scan')
    
    depth_min = 0.0001
    depth_max = 100000000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + point_folder + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    #print(cmd)
    os.system(cmd)
    target = "/filtered_mesh_8.obj"    
    ms = pc2obj_poisson(scan_folder, 9, 10, "/point_clouds.ply", "/mesh9.ply", "/filtered_mesh_9.obj")
    ms = pc2obj_poisson(scan_folder, 8, 10, "/point_clouds.ply", "/mesh8.ply", target)

    if args.facealign:
        print("Change the World coordinate to up -Y forward +Z")
        lmk_3d = np.load(os.path.join(point_folder,'lmk_3d.npz'))
        lmk_3d = lmk_3d['lmk']
        vs, fs, cs = obj_read_w_color(scan_folder + target)
        vertex_mean = np.mean(lmk_3d, 0)
        cam_list = sorted(glob.glob(point_folder[:-16]+'/cams/*_cam.txt'))

        R_world = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        new_origin = vertex_mean.T
        print("New Origin is ",new_origin)
        cam = compute_mvs_RT(R_world, new_origin, cam_list, "cams" ,"cams_align_trans", 4)
        
        
        cam_list = sorted(glob.glob(point_folder[:-16]+'/cams_align_trans/*_cam.txt'))


        transed_3d_lmk = lmk_3d - vertex_mean
        init_3d_lmk = np.load("./Init_3dlmk.npy")
        tmpz = init_3d_lmk[:,2].copy()
        init_3d_lmk[:,2] = -init_3d_lmk[:,1]
        init_3d_lmk[:,1] = -tmpz 
        # change landmarks to up -Y and forward +Z

        torch.from_numpy(init_3d_lmk).to(dtype = torch.float,device = device)
        rotation_matrix = template_opt_init3(torch.from_numpy(init_3d_lmk).to(dtype = torch.float,device = device),torch.from_numpy(transed_3d_lmk).to(dtype = torch.float,device = device))
        R_world = rotation_matrix.detach().cpu().numpy()
        print(R_world)
        new_origin = np.array([0,0,0])
        cam = compute_mvs_RT(R_world, new_origin, cam_list, "cams_align_trans" ,"cams_align_trans_rot", 16)


    if args.facecut:
        target = '/filtered_mesh_cut.obj'
        ms = cut_face_under_neck(ms, point_folder, scan_folder, target)


    if args.face_opt:
        os.makedirs(testpath + '/optimize',exist_ok=True)
        thf = Process(target=face_optim, args=(testpath, scene, target, 10))
        thf.start()    
        target = '/optimized_mesh_final.obj'
        # thf2 = Process(target=face_optim, args=(testpath, scene, "filtered_mesh_9.obj"))
        # thf2.start() 
        thf.join()
        
        
    if args.landmarks_projection:
        v, f, c = obj_read_w_color(os.path.join(scan_folder, 'optimized_mesh_final.obj'))

        v = torch.tensor(v)[None].to(device)
        f = torch.tensor(f)[None].to(device)
        c = torch.tensor(c)[None].to(device)

        lmk_3d = np.load(os.path.join(point_folder,'lmk_3d.npz'))
        lmk_3d = lmk_3d['lmk']
        lmk_3d_tr = torch.tensor(lmk_3d).float().to(device)
        knn = knn_points(lmk_3d_tr[None], v.float())

        lmk_idx = knn[1].squeeze(0).squeeze(-1).detach().cpu().numpy()
        lmk_3d2 = v[0, lmk_idx, :].detach().cpu().numpy()

        np.savez(os.path.join(scan_folder, 'lmk_ref_with_idx.npz'), lmk_idx=lmk_idx, lmk=lmk_3d)
        if True:
            lmk_3d = torch.tensor(lmk_3d).to(device)
            lmk_3d2 = torch.tensor(lmk_3d2).to(device)
            # print(lmk_3d.shape,lmk_3d2.shape)
            point_cloud1 = Pointclouds(points=[lmk_3d],
                                        features=[torch.ones_like(lmk_3d) * torch.tensor([[1, 0, 0]]).to(device)])

            point_cloud2 = Pointclouds(points=v, features=torch.ones_like(v) * torch.tensor([[0, 1, 0]]).to(device))
            point_cloud3 = Pointclouds(points=[lmk_3d2], features=[torch.ones_like(lmk_3d2) * torch.tensor([[0, 0, 1]]).to(device)])
            
            aaa = torch.zeros((2,3)).to(device)
            aaa[0] = lmk_3d[98,:]
            aaa[1] = lmk_3d[327,:]

            point_cloude = Pointclouds(points=[aaa],
                            features=[torch.ones_like(aaa) * torch.tensor([[1, 0, 1]]).to(device)])

            
            # render both in the same plot in different traces
            fig = plot_scene({
                "Pointcloud": {
                    "lmk_3d": point_cloud1,
                        "v": point_cloud2,
                    "lmk_3d_mapping": point_cloud3,
                    "98,327": point_cloude
                }
            })
            fig.show()
        
    if args.facempcut:
        ms.load_new_mesh(scan_folder + '/optimized_mesh_final.obj')
        lmk_3d = np.load(os.path.join(point_folder,'lmk_3d.npz'))
        lmk_3d = lmk_3d['lmk']
        chin_pos = lmk_3d[152,1]
        ms.compute_selection_by_condition_per_vertex(condselect="y>{}".format(chin_pos+10)) 
        ms.meshing_remove_selected_vertices()

        headtop_pos = lmk_3d[10,1]
        ms.compute_selection_by_condition_per_vertex(condselect="y<{}".format(headtop_pos)) 
        ms.meshing_remove_selected_vertices()

        left_top_idx = [338,297,332,284,251,389,356]
        right_top_idx = [109,67,103,54,21,162,127]
        for i in left_top_idx:
            pos = lmk_3d[i]
            ms.compute_selection_by_condition_per_vertex(condselect="x<{} && y<{} &&z<{}".format(pos[0],pos[1],pos[2])) 
            ms.meshing_remove_selected_vertices()
        for i in right_top_idx:
            pos = lmk_3d[i]
            ms.compute_selection_by_condition_per_vertex(condselect="x>{} && y<{} &&z<{}".format(pos[0],pos[1],pos[2])) 
            ms.meshing_remove_selected_vertices()

        left_pos = lmk_3d[454]
        ms.compute_selection_by_condition_per_vertex(condselect="x<{}".format(left_pos[0])) 
        ms.meshing_remove_selected_vertices()

        right_pos = lmk_3d[234]
        ms.compute_selection_by_condition_per_vertex(condselect="x>{}".format(right_pos[0])) 
        ms.meshing_remove_selected_vertices()

        ms.compute_selection_by_condition_per_vertex(condselect="z<{}".format(left_pos[2])) 
        ms.meshing_remove_selected_vertices()
        ms.compute_selection_by_condition_per_vertex(condselect="z<{}".format(right_pos[2])) 
        ms.meshing_remove_selected_vertices()


        # left_bot_idx = [323,361,288,397,365,379,378,400,377]
        # right_bot_idx = [93,132,58,172,136,150,149,176,148]
        # for i in left_bot_idx:
        #     pos = lmk_3d[i]
        #     ms.compute_selection_by_condition_per_vertex(condselect="x<{} && y>{} &&z<{}".format(pos[0],pos[1],pos[2])) 
        #     ms.meshing_remove_selected_vertices()
        # for i in right_bot_idx:
        #     pos = lmk_3d[i]
        #     ms.compute_selection_by_condition_per_vertex(condselect="x>{} && y>{} &&z<{}".format(pos[0],pos[1],pos[2])) 
        #     ms.meshing_remove_selected_vertices()



        ms.transform_rotate(rotaxis = 2 , angle = 180)#rotate mesh to flame fitting +y up +z foward
        ms.save_current_mesh(scan_folder + '/mp_cut.obj')

    if args.flamefit:
        cmd = "python /hdd1/lhs/dev/code/Github_melungl/flame-fitting/fit_scan.py --scandata " + scan_folder + '/mp_cut.obj' +" --lmkdata "+ point_folder + '/lmk_3d.npz' + " --resultpath " + scan_folder
        print(cmd)
        os.system(cmd)
        ms2 = pymeshlab.MeshSet()
        ms2.load_new_mesh(scan_folder + '/flame_fit_scan.obj')
        ms2.transform_rotate(rotaxis = 2 , angle = -180)#rotate mesh to flame fitting +y up +z foward
        ms2.transform_scale_normalize(axisx = 1000, axisy = 1000, axisz = 1000)
        ms2.meshing_surface_subdivision_loop(iterations = 2)
        ms2.save_current_mesh(scan_folder + '/flame_fit_scan_sub.obj')


    if args.face_opt:
        img_idx = args.flameimgidx
        os.makedirs(os.path.join(testpath,scene,'only_face'), exist_ok=True)
        os.makedirs(os.path.join(testpath,scene,'only_face/images'), exist_ok=True)
        os.makedirs(os.path.join(testpath,scene,'only_face/cams'), exist_ok=True)
        for i, k in enumerate(img_idx):
            img_path = os.path.join(point_folder,'2d_landmark',format(k,"08")+'._mask.png')
            cam_path = os.path.join(testpath,'cams',format(k,"08")+'_cam.txt')
            shutil.copy(img_path, os.path.join(testpath,scene,'only_face/images',format(i,"08")+'.png'))
            shutil.copy(cam_path, os.path.join(testpath,scene,'only_face/cams',format(i,"08")+'_cam.txt'))
            
            
            
        #os.makedirs(testpath + '/optimize', exist_ok=True)
        thf = Process(target=face_optim, args=(testpath, scene, '/flame_fit_scan_sub.obj', 100, 1))
        thf.start()    
        target = '/optimized_mesh_final.obj'
        # thf2 = Process(target=face_optim, args=(testpath, scene, "filtered_mesh_9.obj"))
        # thf2.start() 
        thf.join()

    
    if args.faceseg:
        cmd = "python /root/lhs/eye_mask/face_mask.py -p " + point_folder 
        os.system(cmd)
        
        image_folder = os.path.join(point_folder, 'seg_img')
        cmd = fusibile_exe_path
        
        
    



    return
   
def face_optim(data_path, data_name, meshname, iter, test =0):

    cmd = "python /hdd1/lhs/dev/code/Github_melungl/FaceOptimization/main_face.py --data_path " + data_path + " --data_name " + data_name + " --obj_filename " + meshname + " --iters "+ str(iter) + " --test "+ str(test) 
    print(cmd)
    os.system(cmd)



def data_loader_thread(testpath, frontfaceidx, scene, num_view, numdepth, max_h, max_w, fix_res, dataset):
    MVSDataset = find_dataset_def(dataset)
    
    tmp_test_dataset = MVSDataset(testpath, frontfaceidx, [scene], "test", num_view, numdepth, Interval_Scale,max_h=max_h, max_w=max_w, fix_res=fix_res)
    
    return tmp_test_dataset        


def save_depth(testlist):
    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                        share_cr=args.share_cr,
                        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                        grad_method=args.grad_method)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    start_time = time.time()
    state_dict = torch.load(args.loadckpt, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=True)

    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    end_time = time.time()
    print('Model_loaded Parallel Time: ' + str(end_time - start_time))
    MVSDataset = find_dataset_def(args.dataset)
    
    
    pool = ThreadPool(processes=1)
    for i, scene in enumerate(testlist):
        print(scene)
        if i == 0:
            test_dataset = MVSDataset(args.testpath, args.frontfaceidx, [scene], "test", args.num_view, args.numdepth, Interval_Scale,max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
        
        if i+1 < len(testlist):
            tmp_test_dataset = pool.apply_async(data_loader_thread, (args.testpath, args.frontfaceidx, testlist[i+1], args.num_view, args.numdepth, args.max_h, args.max_w, args.fix_res, args.dataset))

        
        t = time.time()
        save_scene_depth(model, test_dataset)
        t1 = time.time()

        print("iter : ", t1 - t)
        locals()['th_{}'.format(i)] = Process(target=depth_map_fusion, args=(args.testpath+'_output' + "/" +scene, args.fusibile_exe_path, args.disp_threshold, args.num_consistent, args.testpath, scene))
        locals()['th_{}'.format(i)].start()

        
        #pc2mesh('./outputs/scan1444')
        
        if i+1 < len(testlist):
            test_dataset = tmp_test_dataset.get()
        
    for i, scene in enumerate(testlist):    
        locals()['th_{}'.format(i)].join()
        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    start_time = time.time()
    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        testlist = []
        for i in range(100):
            testlist.append("scan"+format(i,"04")) 
    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist)
    end_time = time.time()  


    print("Total time", str(end_time - start_time))


