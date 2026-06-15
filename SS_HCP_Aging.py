import os
import numpy as np
import sys
sys.path.append("/ifs/loni/faculty/shi/spectrum/jiazhang/Codes/Python/")
# from utils.mesh_utils.mesh_io_utils import read_shape_obj
# from utils.mesh_utils.mesh_processing_utils import get_disjoint_parts
# import utils.freesurfer_utils.roi_utils as roi_utils
import scipy.io as spio
from tqdm import tqdm
from pathlib import Path

def chi2_distance(A, B):
 
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b+0.01) 
                      for (a, b) in zip(A, B)])
 
    return chi


# adni3_dti_dir = '/ifs/loni/faculty/shi/spectrum/jiazhang/Datasets/EEAJ/data/SWM'

adni3_dti_dir = '/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/datasets/HCP_aging/'
hcp_dti_dir = '/ifs/loni/faculty/shi/spectrum/jiazhang/Datasets/HCP_DTI/'

roi1_list = '1 1 1 1 3 3 3 3 3 5 5 5 5 5 6 7 7 7 7 7 8 8 8 8 8 9 9 9 11 11 11 11 12 12 12 13 13 13 13 14 14 15 15 15 17 17 17 18 18 18 18 18 19 20 20 21 22 22 22 22 22 24 24 24 25 25 27 27 28 29 29 30 30 30 30 31 31'.split(' ')
roi2_list = '1 8 15 30 3 18 24 27 28 5 11 13 25 29 16 7 9 11 13 16 8 11 15 29 31 9 11 15 11 13 15 29 12 19 35 13 16 21 25 14 28 15 30 31 17 25 28 18 20 24 27 35 20 20 27 21 22 24 29 30 31 24 28 35 25 29 27 28 28 29 31 30 31 34 35 31 34'.split(' ')


# roi1_list = '8 29 15 22 12 3 24 22 8 8 8 11 9 9 11'.split(' ')
# roi2_list = '15 29 30 24 12 28 28 29 29 8 11 11 11 15 29'.split(' ')

baseid = np.genfromtxt('/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/SWM_UFiber/Atlas/journal_v1/code/HCP_Aging_SWM_gen/HCA_id.txt',delimiter = ',',dtype = 'str')

# HCP_list = np.genfromtxt('/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/SWM_UFiber/Atlas/journal_v1/code/TractShapeSimilarity/HCP_583_ref483.txt')

HCP_list = np.genfromtxt('/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/SWM_UFiber/Atlas/journal_v1/code/TractShapeSimilarity/HCP_540.txt')


# for hemi in ['lh']:
for hemi in ['lh','rh']:
    for roi1,roi2 in zip(roi1_list,roi2_list):

        file_checker = Path(f'/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/SWM_UFiber/Atlas/journal_v1/code/TractShapeSimilarity/HCP540/HCA_{hemi}_{roi1}_{roi2}_distance_mat_ref_HCP540.mat')
        if not file_checker.exists():
            # print(file_checker,'not exists')

            print (f'Computing ROI pair {roi1} and {roi2}')
            adni_test_sub_list = os.listdir(adni3_dti_dir)
            adni_test_sub_date_list = []
            failed_sub_date_list = []
            for ii in range(len(baseid)):
                
                # base_date = date[0]
                exist0 = os.path.exists(os.path.join(adni3_dti_dir,baseid[ii],'SS',f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_DT.raw'))
                exist1 = os.path.exists(os.path.join(adni3_dti_dir,baseid[ii],'SS',f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_ShapeIndex.raw'))
                if not (exist0 and exist1):
                    failed_sub_date_list.append(baseid[ii])
                else:
                    adni_test_sub_date_list.append(baseid[ii])
            print (f'Verified adni3 sub dates:{len(adni_test_sub_date_list)}')
            print (f'Failed adni3 sub date:{len(failed_sub_date_list)}')

            # HCP_list = np.genfromtxt('/ifs/loni/faculty/shi/spectrum/Student_2020/yuanli/SWM_UFiber/Atlas/journal_v1/code/TractShapeSimilarity/HCP_583.txt')
            hcp_test_sub_list = []
            failed_sub_list = []
            for ii in range(len(HCP_list)):
                # sub = str(int(HCP_list[ii,0]))
                sub = str(int(HCP_list[ii]))

                exist0 = os.path.exists(os.path.join(hcp_dti_dir,sub,f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_DT.raw'))
                exist1 = os.path.exists(os.path.join(hcp_dti_dir,sub,f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_ShapeIndex.raw'))
                if not (exist0 and exist1):
                    failed_sub_list.append(sub)
                else:
                    hcp_test_sub_list.append(sub)
            print (f'Verified hcp sub dates:{len(hcp_test_sub_list)}')
            print (f'Failed hcp sub date:{len(failed_sub_list)}')

            print ('loading adni data...')
            adni_sub_data_dict = {}
            for sub_date in tqdm(adni_test_sub_date_list,desc='Loading adni3_dti data'):
                adni_sub_data_dict[sub_date] = (np.fromfile(os.path.join(adni3_dti_dir,sub_date,'SS',f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_DT.raw'),np.float32),
                                                np.fromfile(os.path.join(adni3_dti_dir,sub_date,'SS',f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_ShapeIndex.raw'),np.float32))


            # hcp_sub_data_dict   = adni_sub_data_dict
            print ('loading hcp data...')
            hcp_sub_data_dict = {}
            for sub in tqdm(hcp_test_sub_list,desc='Loading hcp_dti data'):
                hcp_sub_data_dict[sub] = (np.fromfile(os.path.join(hcp_dti_dir,sub,f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_DT.raw'),np.float32),
                                                np.fromfile(os.path.join(hcp_dti_dir,sub,f'{hemi}_white_DTISpace_rm_roi_pair_{roi1}_{roi2}_ShapeIndex.raw'),np.float32))

            distance_mat = np.zeros((len(adni_test_sub_date_list),len(hcp_test_sub_list)))
                   

            for i,adni_sub_date in enumerate(tqdm(adni_test_sub_date_list,desc='Computing pairwise distance score')):
                adni_dt_hist,_ = np.histogram(adni_sub_data_dict[adni_sub_date][0],bins=100,range=(0,25))
                adni_si_hist,_ = np.histogram(adni_sub_data_dict[adni_sub_date][1],bins=100,range=(-1,1))
                for j,hcp_sub in enumerate(hcp_test_sub_list):
                    hcp_dt_hist,_ = np.histogram(hcp_sub_data_dict[hcp_sub][0],bins=100,range=(0,25))
                    hcp_si_hist,_ = np.histogram(hcp_sub_data_dict[hcp_sub][1],bins=100,range=(-1,1))
                    distance_mat[i,j] = (chi2_distance(adni_dt_hist,hcp_dt_hist) * 0.5 + 
                                        chi2_distance(adni_si_hist,hcp_si_hist) * 0.5)
                    

            spio.savemat(file_checker,{'distance_mat':distance_mat,
                                                                'adni_sub_date_list':adni_test_sub_date_list,
                                                                'hcp_sub_list':hcp_test_sub_list})    
        else:
            print(file_checker,'already exists')
