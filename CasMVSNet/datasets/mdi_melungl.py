from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *
import mediapipe as mp
#Inside CasMVSNet datasets

s_h, s_w = 0, 0
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  #whether to fix the resolution of input image.
        self.fix_wh = False

        assert self.mode == "test"
        self.metas = self.build_list()

        self.datalist = self.build_data()
        
    
            
        



    def build_list(self):
        metas = []
        scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def build_data(self):
        global s_h, s_w
        meta = self.metas[0]
        numview = len(self.metas)
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        imgs = []
        depth_values = None
        proj_matrices = []
        proj_for_landmarks = []
        for i in range(numview):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, i))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, i))
                if not os.path.exists(img_filename):
                    img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.png'.format(scan, i))

            proj_mat_filename = os.path.join(self.datapath, 'cams/{:0>8}_cam.txt'.format(i))
            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=
                                                                                   self.interval_scale[scene_name])
            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h


            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            intrinsics[:2,:] *= 4.0
            intrinsics_s = np.zeros((4, 4))
            intrinsics_s[:3,:3] = intrinsics
            proj_for_landmarks.append(intrinsics_s@extrinsics)


            if i == 0:  # reference view
                #print(self.ndepths)
                ####################### fix this code later 
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)




        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        proj_for_landmarks = np.stack(proj_for_landmarks)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        

        # print(np.shape(proj_matrices))
        # print(proj_matrices[:,0,:,:])
        # print(proj_matrices[:,1,:,:])
        


        face_idx = [1,2,3,4,5]
        face_mesh = mp.solutions.face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) 
        total_face_landmarks =np.zeros((len(face_idx), 478, 2))  
        total_projection =np.zeros((len(face_idx), 4,4))                       
        debug_img = []                  
        for k, i in enumerate(face_idx):
            img = imgs[i]
            proj = proj_for_landmarks[i]

            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            image_height, image_width, _ = img.shape
            img_landmark = img
            results = face_mesh.process(img_landmark)
            img_landmark = cv2.cvtColor(img_landmark, cv2.COLOR_RGB2BGR)
            if not results.multi_face_landmarks:
                print("no detected data - ", file)
                continue
            face_landmarks = results.multi_face_landmarks[0].landmark
            lmk_array = np.zeros((len(face_landmarks), 2))
            #print(len(face_landmarks))
            for f_idx in range(len(face_landmarks)):
                lmk_array[f_idx, 0] = face_landmarks[f_idx].x * image_width
                lmk_array[f_idx, 1] = face_landmarks[f_idx].y * image_height
                #lmk_array[f_idx, 2] = face_landmarks[f_idx].z
                landmark_image = cv2.circle(img_landmark,(int(lmk_array[f_idx,0]),int(lmk_array[f_idx,1])),1,(255,255,255),-1)
            total_face_landmarks[k,:,:] = lmk_array
            total_projection[k,:,:] = proj
            # cv2.imshow('A',landmark_image)
            # cv2.waitKey(0)


        N,M,_ = total_face_landmarks.shape
        X = np.zeros((M,3))
        for i in range(M):
            A = np.empty((2*N,4))
            for j in range(N):
                A[2*j:2*j+2] = [total_face_landmarks[j,i,0]*total_projection[j,2,:]-total_projection[j,0,:],total_face_landmarks[j,i,1]*total_projection[j,2,:]-total_projection[j,1,:]]

            _,_,V = np.linalg.svd(A)
            X[i,:] = V[-1,:-1]/V[-1,-1]

        #print(X)
        np.savez(os.path.join(self.datapath+'_output', '{}'.format(scan, i), 'lmk_3d.npz'), lmk=X)

        N,_,_ = proj_for_landmarks.shape
        re_depth = np.zeros((N,len(face_landmarks)))
        re_pos = np.zeros((N,len(face_landmarks),2))
  
        for i in range(N):
            for j in range(M):
                X_cam = np.dot(proj_for_landmarks[i],np.append(X[j],1))
                re_depth[i,j] = X_cam[2]
                re_pos[i,j,0] = X_cam[0]/X_cam[2]
                re_pos[i,j,1] = X_cam[1]/X_cam[2]
        #print(re_depth)
        #print(re_pos)

        for i in range(N):
            img = imgs[i]
            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            image_height, image_width, _ = img.shape
            img_landmark = img
            img_landmark = cv2.cvtColor(img_landmark, cv2.COLOR_RGB2BGR)
            for f_idx in range(len(face_landmarks)):

                    #lmk_array[f_idx, 2] = face_landmarks[f_idx].z
                    relandmark_image = cv2.circle(img_landmark,(int(re_pos[i,f_idx,0]),int(re_pos[i,f_idx,1])),1,(0,0,255),-1)
            cv2.imwrite(os.path.join(self.datapath, '{}/{:0>8}_re_landmark.jpg'.format(scan, i)),relandmark_image)

        re_depth = re_depth.min(axis=1)
        print(re_depth)
        depth_values = np.zeros((N,2))
        depth_values[:,0] = re_depth - 300.0
        depth_values[:,1] = re_depth + 300.0
        print(depth_values)
        return {"imgs": imgs,
                "stage1": proj_matrices,
                "stage2": stage2_pjmats,
                "stage3": stage3_pjmats,
                "depth_values": depth_values}
    
    
    
    
    
    
    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) #use cam.txt depth min data
        #depth_min = 850 #use direct depth min for blender
        depth_min = 1300 #use direct depth min
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]

            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        #print(filename)
        # scale 0~255 to 0~1
        np_img = np.asarray(img, dtype=np.float32) / 255.

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def __getitem__(self, idx):
        global s_h, s_w
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        
        
        imgs = self.datalist["imgs"][view_ids]
        proj_matrices_ms = {
            "stage1": self.datalist["stage1"][view_ids],
            "stage2": self.datalist["stage2"][view_ids],
            "stage3": self.datalist["stage3"][view_ids],
        }
        
        depth_values = self.datalist["depth_values"][ref_view]
        
        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
