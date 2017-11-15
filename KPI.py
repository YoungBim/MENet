import numpy as np
from PIL import Image


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

class KPI(object):


    def __init__(self, Tasks, num_classes, depth_range_m):
        self.Tasks = Tasks

        # Depth related
        self.depth_range_m = depth_range_m
        self.depth_steps_m = np.array([1])
        while self.depth_steps_m[-1]*2 <= self.depth_range_m :
            self.depth_steps_m = np.append(self.depth_steps_m, [2*self.depth_steps_m[-1]])
        self.depth_gt_counter = np.zeros(shape=self.depth_steps_m.shape) # TODO : explain how this is constructed
        self.depth_error_m = np.zeros(shape=self.depth_steps_m.shape) # TODO : explain how this is constructed
        self.depth_error_relative = np.zeros(shape=self.depth_steps_m.shape) # TODO : explain how this is constructed

        # Segmentation-related
        self.num_classes = num_classes # Note that the last class is background
        self.confusion_matrix = np.zeros(shape=[self.num_classes,self.num_classes], dtype=np.uint32) # Initialize confusion matrix

    def update_segKPI(self, anot, pred):
        pred = np.argmax(pred, axis=3)

        for smpl in range(anot.shape[0]):
            self.update_confusion_matrix(np.take(anot, indices=smpl, axis=0), np.take(pred, indices=smpl, axis=0))

    def update_depthKPI(self, anot, pred):
        for smpl in range(anot.shape[0]):
            self.update_depth_errors(np.take(anot, indices=smpl, axis=0), np.take(pred, indices=smpl, axis=0))

    def compute_KPIs(self, path):
        KPIs = {}
        for task in self.Tasks:
            KPIs[task] = {}
            if task == 'segmentation':
                KPIs[task]['mean_accuracy'] = self.get_mean_accuracy()
                KPIs[task]['mean_iou'] = self.get_mean_iou()
            elif task == 'depth':
                depth_error_m = np.zeros(shape=self.depth_steps_m.shape)
                depth_error_relative = np.zeros(shape=self.depth_steps_m.shape)
                for step in range(self.depth_steps_m.shape[0]):
                    if self.depth_gt_counter[step]>0:
                        depth_error_m[step] = self.depth_error_m[step]/self.depth_gt_counter[step]
                        depth_error_relative[step] = self.depth_error_relative[step]/self.depth_gt_counter[step]
                KPIs[task]['depth_error_m'] = depth_error_m
                KPIs[task]['depth_error_relative'] = depth_error_relative
        import pickle
        with open(path, 'wb') as output:
            # Pickle dictionary using protocol 0.
            pickle.dump(KPIs, output)
        print('KPIs written at', path)

    def parse_KPIs(self, path):
        import pickle
        with open(path, "rb") as fp:
            KPIs = pickle.load(fp)
        return KPIs
    #----------------------------------------------------------------------------------
    # Below this line, the original source code is from :
    #  https://github.com/TensorVision/TensorVision/blob/master/tensorvision/analyze.py
    #----------------------------------------------------------------------------------
    def update_confusion_matrix(self, correct_seg, segmentation):
        """
        Updates the confuscation matrix of a segmentation image and its ground truth.
        The confuscation matrix is a detailed count of which classes i were
        classifed as classes j, where i and j take all (elements) names.
        Parameters
        ----------
        correct_seg : numpy array
            Representing the ground truth.
        segmentation : numpy array
            Predicted segmentation
        elements : iterable
            A list / set or another iterable which contains the possible
            segmentation classes (commonly 0 and 1).
        Updates
        -------
        np.array
            self.confusion_matrix confusion matrix m[correct][classified] = number of pixels in this
            category.
        """
        height, width = correct_seg.shape


        for x in range(width):
            for y in range(height):
                self.confusion_matrix[correct_seg[y][x]][segmentation[y][x]] += 1

    def get_mean_accuracy(self):
        """
        Get the mean accuracy from a confusion matrix n.
        Parameters
        ----------
        self.confusion_matrix : np.array
            Confusion matrix which has integer keys 0, ..., nb_classes - 1;
            an entry n[i][j] is the count how often class i was classified as
            class j.
        Returns
        -------
        float
            mean accuracy (in [0, 1])
        """
        t = []
        k = len(self.confusion_matrix[0])-1
        for i in range(k):
            t.append(sum([self.confusion_matrix[i][j] for j in range(k)]))
        return (1.0 / k) * sum([float(self.confusion_matrix[i][i]) / t[i] for i in range(k)])

    def get_mean_iou(self):
        """
        Get mean intersection over union from a confusion matrix n.
        Parameters
        ----------
        self.confusion_matrix : np.array
            Confusion matrix which has integer keys 0, ..., nb_classes - 1;
            an entry n[i][j] is the count how often class i was classified as
            class j.
        Returns
        -------
        float
            mean intersection over union (in [0, 1])
        """
        t = []
        k = len(self.confusion_matrix[0])-1
        for i in range(k):
            t.append(sum([self.confusion_matrix[i][j] for j in range(k)]))
        return (1.0 / k) * sum([float(self.confusion_matrix[i][i]) / (t[i] - self.confusion_matrix[i][i] +
                                                  sum([self.confusion_matrix[j][i] for j in range(k)]))
                                for i in range(k)])


    def update_depth_errors(self, depth_gt, depth_pred):
        """
        Updates the self.depth_accuracy_m
        Parameters
        ----------
        depth_gt : numpy array
            Ground truth depth
        depth_pred : numpy array
            Predicted depth

        Updates
        -------
        np.array
            TODO : explain
        """
        height, width = depth_gt.shape
        for x in range(width):
            for y in range(height):
                # GT value
                gt = depth_gt[y][x]
                step = find_nearest(self.depth_steps_m,gt)
                if gt > 0:
                    diff = abs(gt-depth_pred[y][x])
                    self.depth_gt_counter[step] += 1
                    self.depth_error_m[step] += diff
                    self.depth_error_relative[step] += diff/gt