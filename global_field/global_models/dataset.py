import os, trimesh, torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from einops import rearrange
from global_models.local import  midpoint_interpolated_up_rate, normalize_point_cloud, extract_knn_patch


def process_data(data_dir, query_dir, dataname):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'):
        pointcloud = np.loadtxt(os.path.join(data_dir, dataname)+ '.xyz').astype(np.float32)
    else:
        print('Only support .xyz or .ply data. Please make adjust your data.')
        print(os.path.join(data_dir, dataname))
        exit()
    pointcloud = torch.from_numpy(pointcloud)
    input_pcd = rearrange(pointcloud, 'n c -> c n').contiguous()
    input_pcd = input_pcd.unsqueeze(0)
    pointcloud, centroid, furthest_distance = normalize_point_cloud(input_pcd)
    pointcloud = pointcloud.squeeze().transpose(0,1).numpy()
    centroid = centroid.reshape(3,1).transpose(0,1).numpy()
    furthest_distance = furthest_distance.reshape(1,1).numpy()

    POINT_NUM = pointcloud.shape[0]
    POINT_NUM_GT = pointcloud.shape[0]
    QUERY_EACH = 500000//POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud[point_idx,:]
    ptree = cKDTree(pointcloud)
    sigmas = []
    nn_dis = [] # nearest neighbor distance for testing
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])
        nn_dis.append(d[0][:, 1])
    sigmas = np.concatenate(sigmas)
    nn_dis = np.concatenate(nn_dis)
    sample = []
    neighbors = []
    for i in tqdm(range(QUERY_EACH)):
        scale = 0.2
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        rand_points = np.random.rand(int(POINT_NUM*0.05), 3)*2-1
        tt = np.concatenate((tt, rand_points), axis=0)
        _, ind = ptree.query(tt, k=1)
        neighbors.append(pointcloud[ind])
        sample.append(tt)

    sample = np.asarray(sample)
    neighbors = np.asarray(neighbors)
    os.makedirs(os.path.join(data_dir, query_dir), exist_ok=True)
    np.savez(os.path.join(data_dir, query_dir, dataname)+'.npz', sample = sample, pointcloud=pointcloud, centroid=centroid, scale=furthest_distance, nn_dis=nn_dis, neighbors=neighbors)
    
class GlobalDataset:
    def __init__(self, conf, dataname):
        super(GlobalDataset, self).__init__()
        self.device = torch.device('cuda')
        self.conf = conf
        self.k = conf['k']
        self.up_rate = conf['up_rate']
        self.up_name = conf['up_name']
        self.data_dir = conf.get_string('data_dir')
        self.shape_name = dataname + '.npz'
        self.query_dir = conf['query_dir']

        if os.path.exists(os.path.join(self.data_dir, self.query_dir, self.shape_name)):
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...')
            process_data(self.data_dir, self.query_dir, dataname)

        load_data = np.load(os.path.join(self.data_dir, self.query_dir, self.shape_name))
        
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        self.pointcloud = np.asarray(load_data['pointcloud']).reshape(-1,3)
        self.nn_dis = np.asarray(load_data['nn_dis']).reshape(-1,1)
        self.scale = np.asarray(load_data['scale'].reshape(-1,1))
        self.centroid = np.asarray(load_data['centroid']).reshape(-1,3)
        self.sample_points_num = self.sample.shape[0]-1
        self.neighbors = np.asarray(load_data['neighbors']).reshape(-1,3)

        self.object_bbox_min = np.array([np.min(self.pointcloud[:,0]), np.min(self.pointcloud[:,1]), np.min(self.pointcloud[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.pointcloud[:,0]), np.max(self.pointcloud[:,1]), np.max(self.pointcloud[:,2])]) +0.05
        print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)
        
        self.sample = torch.from_numpy(self.sample).to(self.device).float()

        self.tree = cKDTree(self.pointcloud)
        self.pointcloud = torch.from_numpy(self.pointcloud).to(self.device).float()
        self.nn_dis = torch.from_numpy(self.nn_dis).to(self.device).float()
        self.neighbors = torch.from_numpy(self.neighbors).to(self.device).float()
        self.interpolated_pcd = midpoint_interpolated_up_rate(conf['up_rate'], self.pointcloud.unsqueeze(0).transpose(1,2))
        print('Global NP Load data: End')

    def global_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse
        sample = self.sample[index]
        nn = self.neighbors[index]
        return sample, nn


    def get_nn(self, sample):
        _, nn = self.tree.query(sample.detach().cpu())
        return self.pointcloud[nn]

    def get_knn(self, sample, sample_moved, center=False):
        sample = rearrange(sample, 'n c -> c n') # c* n
        sample = sample.unsqueeze(0) # 1 * c *n
        # b*k * 3 * n
        # patches = extract_knn_patch(self.k, self.pointcloud.unsqueeze(0).transpose(1,2), sample)
        patches = extract_knn_patch(self.k, self.interpolated_pcd, sample) # n * c * k
        patches, centroid, furthest_distance = normalize_point_cloud(patches) # n * c * k;  n * c * 1; n * 1 * 1
        
        # change the sample
        sample_moved = (sample_moved - centroid.reshape(-1, 3))/furthest_distance.reshape(-1,1) # n * c 
        sample_moved = rearrange(sample_moved.contiguous(), 'n c -> c n').unsqueeze(-1)
        sample_moved = rearrange(sample_moved, 'c n b -> n c b')
        if center:
            return patches, sample_moved, furthest_distance, centroid
        return patches, sample_moved, furthest_distance
