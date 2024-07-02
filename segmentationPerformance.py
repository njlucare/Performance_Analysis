import numpy as np
import multiprocessing
#from PIL import Image
#from subprocess import call
from joblib import Parallel, delayed
from xml_to_mask_ome import xml_to_mask
from skimage.measure import regionprops,label
import matplotlib.pyplot as plt
import pandas as pd
import tifffile as ti
import seaborn as sns
from tqdm import tqdm

class ConfMat:
    def __init__(self,confusionMatrix,classLabels):
        self.confusionMatrix = confusionMatrix
        self.classLabels = classLabels
        assert self.confusionMatrix.shape[0] == self.confusionMatrix.shape[1], "Confusion Matrix must be square"
        assert len(classLabels) == self.confusionMatrix.shape[0], f"Class Labels ({len(classLabels)}) must be same size as confusion matrix ({self.confusionMatrix.shape[0]})"


    def get_balanced_accuracy(self,classes):
        assert max(classes) < self.confusionMatrix.shape[0], f"Highest class requested: {max(classes)}, Matrix size: {self.confusionMatrix.shape[0]}"

        sens=[]
        spec=[]

        for i in classes:

            TP = self.confusionMatrix[i,i]
            TN = np.sum(self.confusionMatrix[:,:]) - np.sum(self.confusionMatrix[i,:]) - np.sum(self.confusionMatrix[:,i]) + self.confusionMatrix[i,i]
            FP = np.sum(self.confusionMatrix[:,i]) - self.confusionMatrix[i,i]
            FN = np.sum(self.confusionMatrix[i,:]) - self.confusionMatrix[i,i]

            sens.append(TP/(TP+FN))
            spec.append(TN/(TN+FP))

        balanced_accuracy = np.nansum(np.array(sens))/(len(sens))

        return balanced_accuracy

    def plot_matrix(self,remove_background):
        
        if remove_background:
            conf_subset = self.confusionMatrix[1:,1:]
            label_subset = self.classLabels[1:]
        else:
            conf_subset = self.confusionMatrix
            label_subset = self.classLabels
            
        colors = np.vectorize(self.__color_mapping)(conf_subset)
        plt.figure(figsize=(40,32))#24x24
        sns.set(font_scale=1.3)
        sns.heatmap(colors,annot=conf_subset,fmt=".1e",cmap="Blues",xticklabels=label_subset,yticklabels=label_subset)
        plt.xlabel("Predicted",fontsize=32)
        plt.ylabel("Ground Truth",fontsize=32)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)

    def __color_mapping(self,value):

        row,col = np.where(self.confusionMatrix==value)
        true_label = row[0]
        class_sizes = np.sum(self.confusionMatrix,axis=1)
        color_intensity = value / class_sizes[true_label]

        return color_intensity




class Performance:
    def __init__(self,segmentationList,groundTruthList,unusable,mask_subtract):
        self.segList = segmentationList
        self.gtList = groundTruthList
        self.unusable = unusable
        self.maskSubtract = mask_subtract
        runningClassNum = 0


        for seg,gt in zip(self.segList,self.gtList):
            max_gt = np.max(ti.imread(gt))
            max_seg = np.max(ti.imread(seg))
            
            if max_gt == 255:
                max_gt = np.unique(ti.imread(gt))[-2]
                
            
            if max_seg > runningClassNum:
                runningClassNum = max_seg
            if max_gt > runningClassNum:
                runningClassNum = max_gt

        self.classNum = runningClassNum
        self.confusionMatrix = np.zeros((self.classNum+1,self.classNum+1))
        
    def get_confusion(self):
        pbar = tqdm(total=len(self.segList),desc='Getting Confusion Matrix...')
        for seg,gt in zip(self.segList,self.gtList):
            gt_mask = ti.imread(gt)
            seg_mask = ti.imread(seg)
            gt_mask[gt_mask==255] = 0
            if len(seg_mask.shape) > 2:
                seg_mask = seg_mask[0,:,:]

            if self.unusable is not None:
                seg_mask,gt_mask = self.__mask_unusable(seg_mask,gt_mask,gt)
            if self.maskSubtract is not None:
                seg_mask,gt_mask = self.__mask_annotated(seg_mask,gt_mask,gt)


            confmat = np.zeros((self.classNum+1,self.classNum+1))
            
            #This is where i should implement parallelization
            num_splits = int(np.floor(multiprocessing.cpu_count()**0.5))
            
            
            confmat = self._process_image(confmat, seg_mask, gt_mask, num_splits)
            confmat = sum(confmat)
            
            self.confusionMatrix+=confmat
            pbar.update(1)
        pbar.close()
        return self.confusionMatrix
            

    def get_PQ(self):
        PQ_list = []
        IoU_list = []
        pbar = tqdm(total=len(self.segList),desc='Getting PQ...')
        for seg,gt in zip(self.segList,self.gtList):
            gt_mask = ti.imread(gt)
            seg_mask = ti.imread(seg)
            if len(seg_mask.shape) > 2:
                seg_mask = seg_mask[0,:,:]

            PQs = []
            mIoUs = []

            for objectClass in range(1,self.classNum+1):

                temp_gt = (gt_mask == objectClass).astype(np.int32)
                temp_gt = label(temp_gt)
                temp_seg = (seg_mask == objectClass).astype(np.int32)
                temp_seg = label(temp_seg)

                gt_bb = regionprops(temp_gt)
                ss_bb = regionprops(temp_seg)

                IoU = 0
                TP = []
                TP_gcheck = []


                for object in range(1,np.max(temp_gt)+1):

                    temp_IoU = 0
                    x1,y1,x2,y2 = gt_bb[object-1].bbox

                    gg = temp_gt[x1:x2,y1:y2]
                    ss = temp_seg[x1:x2,y1:y2]

                    unique_overlaps = np.unique(np.multiply((gg==object).astype(np.int32),ss))

                    if unique_overlaps[0] == 0:
                        unique_overlaps = unique_overlaps[1:]

                    unique_segs = np.unique(ss)
                    if unique_segs[0] == 0:
                        unique_segs = unique_segs[1:]

                    for obj in unique_segs:
                        overlap_sum = np.sum(np.multiply((gg==object).astype(np.int32),(ss==obj).astype(np.int32)))
                        gt_sum = np.sum((gg==object).astype(np.int32))
                        if float(overlap_sum)/float(gt_sum) >= 0.5:

                            x3,y3,x4,y4 = ss_bb[obj-1].bbox
                            x,xx,y,yy = self.__maximize_bbox(x1,x2,x3,x4,y1,y2,y3,y4)

                            gg = temp_gt[x:xx,y:yy]
                            ss = temp_gt[x:xx,y:yy]

                            union = np.sum((((ss==obj).astype(np.int32)+(gg==object).astype(np.int32))>0).astype(np.int32))
                            temp_IoU = float(overlap_sum)/float(union)
                            TP.append(obj)
                            TP_gcheck.append(object)


                    IoU += temp_IoU


                FP_count = np.max(temp_seg) - len(TP)
                FN_count = np.max(temp_gt) - len(TP_gcheck)
                TP_count = len(TP)


                if TP_count+FN_count+FP_count==0:
                    PQ=0
                else:
                    PQ = IoU/(float(TP_count)+0.5*float(FN_count)+0.5*float(FP_count))
                PQs.append(PQ)
                mIoUs.append(IoU/np.max(temp_gt))
            PQ_list.append(PQs)
            IoU_list.append(mIoUs)
            pbar.update(1)
        pbar.close()
        return PQ_list,IoU_list
    
    def _process_crop(self,confmat,seg_crop,gt_crop):
        
        for i in np.unique(gt_crop):
                for j in np.unique(seg_crop):
                    temp_seg = (seg_crop==j).astype(np.int32)
                    temp_gt = (gt_crop==i).astype(np.int32)

                    TP = np.sum(np.multiply(temp_gt,temp_seg))
                    confmat[i,j] = TP
        
   
        return confmat
    
    def __split_image(self,image, num_splits):
        # Split the image into a grid of sub-images
        height, width = image.shape
        
        split_height = height // num_splits
        split_width = width // num_splits
        
        crops = []
        for i in range(num_splits):
            for j in range(num_splits):
                crop = image[i*split_height:(i+1)*split_height, j*split_width:(j+1)*split_width]
                crops.append(crop)
        return crops
    
    def _process_image(self,confmat,seg_mask,gt_mask, num_splits):        
    
        # Split the image into crops
        seg_crops = self.__split_image(seg_mask, num_splits)
        gt_crops = self.__split_image(gt_mask, num_splits)
        
        # Use joblib to process each crop in parallel
        zipped_crops = zip(seg_crops,gt_crops)
        
        confmats = Parallel(n_jobs=int(num_splits**2))(delayed(self._process_crop)(confmat,seg_crop,gt_crop) for (seg_crop,gt_crop) in zipped_crops)
    
        # Sum all the results together
        return confmats


    def __maximize_bbox(self,x1,x2,x3,x4,y1,y2,y3,y4):

        if x1<x3:
            x = x1
        else:
            x = x3

        if x2 > x4:
            xx = x2
        else:
            xx = x4

        if y1 < y3:
            y = y1
        else:
            y = y3

        if y2 > y4:
            yy = y2
        else:
            yy = y4


        return x,xx,y,yy

    def __get_bounds(self,gt):
        unusable = pd.read_excel(self.unusable)
        unusable_slides = unusable["Slide"]

        slide_nm = gt.split('/')[-1].split('_Registered.tif')[0]

        x_s = (unusable["X0"][unusable_slides==slide_nm]).to_numpy()[0]
        x_e = (unusable["Xe"][unusable_slides==slide_nm]).to_numpy()[0]
        y_s = (unusable["Y0"][unusable_slides==slide_nm]).to_numpy()[0]
        y_e = (unusable["Ye"][unusable_slides==slide_nm]).to_numpy()[0]

        return x_s,x_e,y_s,y_e

    def __mask_unusable(self,seg_mask,gt_mask,gt):

        x_s,x_e,y_s,y_e = self.__get_bounds(gt)
        dimx,dimy = gt_mask.shape

        if x_s > 0:
            gt_mask = gt_mask[:x_s,:]
            seg_mask = seg_mask[:x_s,:]
        if x_e < dimx:
            gt_mask = gt_mask[x_e:,:]
            seg_mask = seg_mask[x_e:,:]
        if y_s > 0:
            gt_mask = gt_mask[:,:y_s]
            seg_mask = seg_mask[:,:y_s]
        if y_e < dimy:
            gt_mask = gt_mask[:,y_e:]
            seg_mask = seg_mask[:,y_e:]

        return seg_mask,gt_mask

    def __mask_annotated(self,temp_seg,temp_gt,gt):
        try:
            slide_count = self.groundTruthList.index(gt)
        except:
            print("Something went wrong")

        dimx,dimy = temp_gt.shape
        artifacts = xml_to_mask(self.__maskSubtract[slide_count],(0,0),(dimy,dimx),[1])

        artifacts = artifacts.astype(np.uint8)

        x_s,x_e,y_s,y_e = self.__get_bounds(gt)

        if x_s > 0:
            artifacts = artifacts[:x_s,:]
        if x_e < dimx:
            artifacts = artifacts[x_e:,:]
        if y_s > 0:
            artifacts = artifacts[:,:y_s]
        if y_e < dimy:
            artifacts = artifacts[:,y_e:]

        temp_seg = temp_seg*artifacts
        temp_gt = temp_gt*artifacts

        return temp_seg,temp_gt


#Create a list of all your segmentation outputs
segmentations = [
        '/blue/pinaki.sarder/nlucarelli/BL_segmentation/slides_frozen_kidney/HE_0020_section1.tif'
        ]

#Create a list of all your corresponding ground truths
gts = [
       '/blue/pinaki.sarder/nlucarelli/BL_segmentation/slides_frozen_kidney/HE_0020_section1_Registered.tif'
        ]

#Any annotated files for areas you'd like to exclude (artifacts, etc.)
mask_subtract = None
#Excel file containing coordinates of unusable regions of the tissue
unusable = None
#Where you want to save the confusion matrix
# output_csv = '/blue/pinaki.sarder/nlucarelli/Performance/intestine_attention.csv'

# #Create the object
performance1 = Performance(segmentations, gts, unusable, None)

# #Get confusion matrix
cm = performance1.get_confusion()
# cm = pd.read_csv('/blue/pinaki.sarder/nlucarelli/Performance/frozen_pec.csv').to_numpy()[1:,2:]
#Get panoptic quality, and intersection over union
# pqs,ious = performance1.get_PQ()

#Create a ConfMat object, provide a list of the class labels
# confmat1 = ConfMat(cm, ['ATL','IMM','CNT','DCT','ENDO','PEC','GLOM','CD','PT','SMC','TAL'])

# # #Get metrics and plot
# confmat1.get_balanced_accuracy(list(range(11))[0:])
# confmat1.plot_matrix(False)
# plt.savefig('/blue/pinaki.sarder/nlucarelli/perf.png')




