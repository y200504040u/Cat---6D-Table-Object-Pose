# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse
import os
import cv2
import time
import numpy as np
from random import sample
from scipy.spatial.transform import Rotation as Sci_R

import tensorrt as trt
import common

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

TRT_LOGGER = trt.Logger()

path_root = "./imgs/rgb"

list_color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (0,153,255), (153,0,255)]

min_z = 1200
step = 32
num_iter = 500
thre_dis = 0.010   #m
thre_dis1 = 0.020

f_x, f_y, pp_x, pp_y, width, height = 517.623840, 517.720154, 319.173828, 238.091400, 640, 480
# f_x, f_y, pp_x, pp_y, width, height = 345.421112, 345.368286, 320.968750, 180.520447, 640, 360

camera_matrix = np.array([[f_x, 0, pp_x],
                          [0, f_y, pp_y],
                          [0, 0, 1]])
inv_camera_matrix = np.linalg.inv(camera_matrix)

# def get_engine(engine_file_path):
#     if os.path.exists(engine_file_path):
#         # If a serialized engine exists, use it instead of building an engine.
#         print("Reading engine from file {}".format(engine_file_path))
#         with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())
#     else:
#         return None

def get_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return None
    

class YOLOv8Seg:
    """YOLOv8 segmentation model."""

    def __init__(self, trt_model):
        """
        Initialization.

        Args:
            trt_model (str): Path to the trt model.
        """
        self.input_size = [640, 640]
        with get_engine(
            trt_model
        ) as self.engine, self.engine.create_execution_context() as self.context:
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.single

        # Get model width and height(YOLOv8-seg only has one input)
        self.model_height, self.model_width = self.input_size

        # Load COCO class names
        self.classes = {0: 'apple', 1: 'orange', 2: 'banana', 3: 'mouse', 4: 'brush', 5: 'box', 6: 'pringles', 7: 'watering_can'}
        self.classes_nopose = ['apple', 'orange']
        self.classes_pose = ['mouse', 'watering_can']
        self.radius = [0.04, 0.04]

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45, nm=32):
        # Pre-process
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
        self.inputs[0].host = im

        # Ort inference
        # preds = self.session.run(None, {self.session.get_inputs()[0].name: im})
        trt_outputs = []
        
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        
        preds = [trt_outputs[1].reshape((1, 44, 8400)), trt_outputs[0].reshape((1, 32, 160, 160))]
        # Post-process
        boxes, masks, centroids = self.postprocess(
            preds,
            im0=im0,
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            nm=nm,
        )
        return boxes, masks, centroids

    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """

        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)

    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls, nm) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

        # NMS filtering
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

        # Decode and return
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            # Process masks
            masks = self.process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)

            centroids = self.masks_rm_bad(masks)
            return x[..., :6], masks, centroids  # boxes, masks, centroids
        else:
            return [], [], []

    @staticmethod
    def masks_rm_bad(masks):
        list_centroids = []
        for idx, x in enumerate(masks):
            num_labels, labels, stats, cts = cv2.connectedComponentsWithStats(x, connectivity=8)
            
            stats = stats[1:, :]
            max_area_id = np.argmax(stats[:, 4]) + 1
            list_centroids.append(cts[max_area_id])
            if num_labels > 2:
                x = np.zeros_like(x)
                x = np.where(labels == max_area_id, 1, 0).astype(np.uint8)
                masks[idx] = x
        centroids = np.array(list_centroids).astype(np.uint16)
        return centroids

    @staticmethod
    def crop_mask(masks, boxes):
        """
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L599)

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher quality
        but is slower. (Borrowed from https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L618)

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5).astype(np.uint8)

    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size. (Borrowed from
        https://github.com/ultralytics/ultralytics/blob/465df3024f44fa97d4fad9986530d5a13cdabdca/ultralytics/utils/ops.py#L305)

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks


class RegressionModel:
    """Regression model."""
    def __init__(self, trt_model, image_size=[224, 224]):
        self.input_size = image_size
        with get_engine(
            trt_model
        ) as self.engine, self.engine.create_execution_context() as self.context:
            self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def __call__(self, roi_bgr_mask, bbox):
        roi_rgb_mask = cv2.cvtColor(roi_bgr_mask, cv2.COLOR_BGR2RGB)
        roi_rgb_mask_rs = cv2.resize(roi_rgb_mask, dsize=self.input_size, fx=0., fy=0.).astype(np.float32)
        img_norm = roi_rgb_mask_rs / 255.
        img_norm -= np.array([0.485, 0.456, 0.406])
        img_norm /= np.array([0.229, 0.224, 0.225])
        img_input = img_norm.transpose((2,0,1))[np.newaxis,:]

        self.inputs[0].host = img_input.copy()

        # Ort inference
        trt_outputs = []
        
        trt_outputs = common.do_inference_v2(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)[0]

        h_roi, w_roi, _ = roi_bgr_mask.shape
        pts_cen_2d = np.array(trt_outputs[:2] * np.array([w_roi, h_roi]) + np.array([bbox[0], bbox[1]]), np.uint16)

        rst_vec = trt_outputs[2:]
        rst_vec /= np.linalg.norm(rst_vec)

        return pts_cen_2d, rst_vec
    

class ObjMakeUp:
    def __init__(self):
        self.cls = -1
        self.conf = 0.
        self.coord3d_cen = None
        self.coord2d_cen = None
        self.mask = None
        self.bbox = None
        self.rst_vec = None
        self.pts_cen_2d = None
        self.R = None
        self.qua = None


class GetObjPose:
    def __init__(self, trt_model_seg, trt_model_poser):
        self.flagShow = True
        self.plane = None
        # self.img_desk = None
        self.objs = []
        
        self.segger = YOLOv8Seg(trt_model_seg)
        self.poser = RegressionModel(trt_model_poser)

    def find_low_pts_split_lr(self, dep16, start_col, end_col):
        pts_1st = []
        pts_2st = []
        for i in range(dep16.shape[0] // 3, dep16.shape[0], step):
            max_dep = -1
            pts_coord = (0, 0)
            for j in range(start_col, end_col):
                if dep16[i, j] > max_dep and dep16[i, j] < min_z:
                    max_dep = dep16[i, j]
                    pts_coord = (i, j)
            if (max_dep > 0):
                pts_1st.append(pts_coord)

        for idx in range(len(pts_1st) - 2):
            p = pts_1st[idx]
            pre_p = pts_1st[idx + 1]
            pre_pre_p = pts_1st[idx + 2]
            if (dep16[p] > dep16[pre_p] and dep16[pre_p] > dep16[pre_pre_p]):
                pts_2st.append(p)
                pts_2st.append(pre_p)
                pts_2st.append(pre_pre_p)
                idx += 3
            else:
                idx += 2
        new_pts_2st = list(dict.fromkeys(pts_2st))    

        return new_pts_2st
        
    def calc_3d_cood(self, dep16, coor2d):
        z = dep16[coor2d[:, 1], coor2d[:, 0]].reshape(-1, 1) * 1e-3
        coor2d = np.hstack((coor2d, np.ones_like(z)))
        coor2d *= z
        coor2d = np.transpose(coor2d)
        coor3d = np.dot(inv_camera_matrix, coor2d)
        coor3d = np.vstack((coor3d, np.ones_like(np.transpose(z))))
        return coor3d

    def calc_plane(self, p3):
        d0 = p3[1] - p3[0]
        d1 = p3[2] - p3[1]
        n = np.cross(d0, d1)
        bias = -np.sum(n * p3[0])
        return np.array([n[0], n[1], n[2], bias])

    def calc_dis(self, plane, pts):
        return np.abs(np.dot(plane, pts)) / np.linalg.norm(plane)

    def desk_detect(self, dep16):
        pts_2st = []
        pts_2st_l = self.find_low_pts_split_lr(dep16, 0, dep16.shape[1] // 2)
        pts_2st_r = self.find_low_pts_split_lr(dep16, dep16.shape[1] // 2, dep16.shape[1])
        pts_2st.extend(pts_2st_l)
        pts_2st.extend(pts_2st_r)
        if len(pts_2st) < 10:
            return
        coor2d = np.array(pts_2st)
        coor2d = coor2d[:, ::-1]

        coor3d = self.calc_3d_cood(dep16, coor2d)
        list_coor3d = list(np.transpose(coor3d[:3, :]))
        if len(list_coor3d) < 8:
            print("Find desk plane failed.")
            return

        i = 0
        max_num_ok_pts = 0
        best_plane = np.array([0])
        num_coords3d = len(list_coor3d)
        max_num_iter = num_coords3d * (num_coords3d - 1) * (num_coords3d - 2) / 6
        while(i < min(max_num_iter, num_iter)):
            p3 = sample(list_coor3d, 3)
            plane = self.calc_plane(p3)
            dis = self.calc_dis(plane, coor3d)
            idx_ok = dis < thre_dis
            num_ok_pts = np.sum(idx_ok)
            if num_ok_pts > max_num_ok_pts:
                max_num_ok_pts = num_ok_pts
                best_plane = plane
            i += 1
        if best_plane[2] > 0:
            best_plane *= -1 
        
        self.plane = best_plane# / np.linalg.norm(best_plane)

    def draw_desk(self, dep16):
        plane = self.plane
        x = np.linspace(0, dep16.shape[1] - 1, dep16.shape[1])
        y = np.linspace(0, dep16.shape[0] - 1, dep16.shape[0])

        c, r = np.meshgrid(x, y)
        coords2d = np.hstack((c.astype(np.uint16).reshape(-1, 1), r.astype(np.uint16).reshape(-1, 1)))
        coords3d = self.calc_3d_cood(dep16, coords2d)
        dis = self.calc_dis(plane, coords3d).reshape(dep16.shape[0], dep16.shape[1])
        desk16 = dep16.copy()
        desk16 = np.where(dis < thre_dis1, 0, desk16)
        desk_show = np.asarray(desk16 * 0.125, np.uint8)
        return desk16, desk_show
    
    def get_pt3d_ondesk(self, pt_2d):
        Z_cen = -self.plane[3] / (self.plane[0] * (pt_2d[0] - pp_x) / f_x + self.plane[1] * (pt_2d[1] - pp_y) / f_y + self.plane[2])
        X_cen = Z_cen * (pt_2d[0] - pp_x) / f_x
        Y_cen = Z_cen * (pt_2d[1] - pp_y) / f_y
        pt_3d = np.array([X_cen, Y_cen, Z_cen])
        return pt_3d

    def get_direction(self, img_bgr, obj):
        bbox = obj.bbox
        mask = obj.mask
        mask_ch3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        roi_bgr = img_bgr[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        roi_mask = mask_ch3[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
        roi_bgr_mask = np.where(roi_mask, roi_bgr, 0)

        pts_cen_2d, rst_vec = self.poser(roi_bgr_mask, bbox)
        obj.rst_vec = rst_vec.copy()
        obj.coord3d_cen = self.get_pt3d_ondesk(pts_cen_2d)
        obj.coord2d_cen = np.transpose(np.dot(camera_matrix, obj.coord3d_cen) / obj.coord3d_cen[2])[:2].astype(np.uint16)

        obj = self.get_R(obj)
        return obj

    def get_R(self, obj):
        length = 200.
        pt_end_2d = (obj.rst_vec * length + obj.coord2d_cen).astype(np.uint16)
        pt_end_3d = self.get_pt3d_ondesk(pt_end_2d)

        y_c = pt_end_3d - obj.coord3d_cen
        y_c /= np.linalg.norm(y_c)
        z_c = self.plane[:3] / np.linalg.norm(self.plane[:3])
        x_c = np.cross(y_c, z_c)
        x_c /= np.linalg.norm(x_c)
        obj.R = np.hstack((x_c.reshape((3, 1)), y_c.reshape((3, 1)), z_c.reshape((3, 1))))
        rr = Sci_R.from_matrix(obj.R)
        obj.qua = rr.as_quat()
        return obj

    def Run(self, img_bgr, dep16_raw):
        self.plane = None
        # self.img_desk = None
        self.objs = []

        dep16_raw.astype(np.uint16)
        dep16 = dep16_raw#[:,:,1] * 256 + dep16_raw[:, :, 0]
        
        self.desk_detect(dep16)
        
        if self.plane is None:
            print("Failed to detect desk!")
            return [[-1., 0., 0., 0., 0., 0., 0., 0.]]
     
        bboxes, masks, cen_2ds = self.segger(img_bgr, conf_threshold=0.6, iou_threshold=0.)
        if len(masks) == 0:
            print("No object detected!")
            return [[-1., 0., 0., 0., 0., 0., 0., 0.]]
        
        l_out_put = []
        cen_3ds_ = np.transpose(self.calc_3d_cood(dep16, cen_2ds)[:3, :])
        for (*bbox, conf, cls_), mask, c3d_ in zip(bboxes, masks, cen_3ds_):
            obj = ObjMakeUp()
            obj.cls = int(cls_)
            obj.conf = conf
            obj.mask = mask.copy()
            obj.bbox = np.array(bbox, dtype=np.uint16)
            name = self.segger.classes[obj.cls]
            if name in self.segger.classes_nopose:
                rr = self.segger.radius[obj.cls]
                c3d = c3d_ * (1. + rr)
                obj.coord3d_cen = c3d
                obj.coord2d_cen = np.transpose(np.dot(camera_matrix, c3d) / c3d[2])[:2].astype(np.uint16)
                out_put = [float(obj.cls)] + list(obj.coord3d_cen) + [0.] * 4

                self.objs.append(obj)
                l_out_put.append(out_put)
            elif name in self.segger.classes_pose:
                obj = self.get_direction(img_bgr, obj)
                out_put = [float(obj.cls)] + list(obj.coord3d_cen) + list(obj.qua)

                self.objs.append(obj)
                l_out_put.append(out_put)
            else:
                pass

        if len(self.objs) == 0:
            return [[-1., 0., 0., 0., 0., 0., 0., 0.]]

        # out_put = [[float(x) for x in l_out_put]]
        return l_out_put
    
    def draw(self, img_bgr, dep16):
        if self.plane is None:
            img_desk = np.asarray(dep16 * 0.125, np.uint8)
        else:
            _, img_desk = self.draw_desk(dep16)
        # cv2.imshow("desk", img_desk)
        if len(self.objs) > 0:
            for obj in self.objs:
                cls_ = obj.cls
                mask = obj.mask
                conf = obj.conf
                bbox = obj.bbox
                mask_ch3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_alpha = mask_ch3 * list_color[cls_]
                plus = np.where(mask_ch3, mask_alpha, img_bgr)
                img_bgr = cv2.addWeighted(img_bgr.astype(np.float32), 0.4, plus.astype(np.float32), 0.6, 0).astype(np.uint8)
                cv2.circle(img_bgr, obj.coord2d_cen, 2, (0,0,0), -1)
                cv2.putText(img_bgr, f"{self.segger.classes[cls_]}: {conf:.3f}", bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

                name = self.segger.classes[cls_]
                if name in self.segger.classes_nopose:
                    continue
                coord_3d_yb_o = np.array([0, 0.1, 0])
                coord_3d_yb_c = np.dot(obj.R, coord_3d_yb_o) + obj.coord3d_cen
                
                coord_3d_x_o = np.array([0.1, 0, 0])
                coord_3d_x_c = np.dot(obj.R, coord_3d_x_o) + obj.coord3d_cen
                coord_2d_x = np.transpose(np.dot(camera_matrix, coord_3d_x_c) / coord_3d_x_c[2])[:2].astype(np.uint16)
                if coord_2d_x[0] >= 0 and coord_2d_x[0] < width and coord_2d_x[1] >= 0 and coord_2d_x[1] < height:
                    cv2.arrowedLine(img_bgr, obj.coord2d_cen, coord_2d_x, (255, 0, 0), 2, 0, 0, 0.2)
                    cv2.putText(img_bgr, "x", coord_2d_x + 5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # coord_3d_y_c = 2 * obj.coord3d_cen - coord_3d_yb_c
                coord_3d_y_c = coord_3d_yb_c
                coord_2d_y = np.transpose(np.dot(camera_matrix, coord_3d_y_c) / coord_3d_y_c[2])[:2].astype(np.uint16)
                if coord_2d_y[0] >= 0 and coord_2d_y[0] < width and coord_2d_y[1] >= 0 and coord_2d_y[1] < height:
                    cv2.arrowedLine(img_bgr, obj.coord2d_cen, coord_2d_y, (0, 255, 0), 2, 0, 0, 0.2)
                    cv2.putText(img_bgr, "y", coord_2d_y + 5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                coord_3d_z_o = np.array([0, 0, 0.1])
                coord_3d_z_c = np.dot(obj.R, coord_3d_z_o) + obj.coord3d_cen
                coord_2d_z = np.transpose(np.dot(camera_matrix, coord_3d_z_c) / coord_3d_z_c[2])[:2].astype(np.uint16)
                if coord_2d_z[0] >= 0 and coord_2d_z[0] < width and coord_2d_z[1] >= 0 and coord_2d_z[1] < height:
                    cv2.arrowedLine(img_bgr, obj.coord2d_cen, coord_2d_z, (0, 0, 255), 2, 0, 0, 0.2)
                    cv2.putText(img_bgr, "z", coord_2d_z + 5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img_bgr, img_desk 

    

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str("./data/imgs"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    args = parser.parse_args()

    # Build model
    #cObjPose = GetObjPose(args.model, args.model.replace(".onnx", ".trt"))
    path_trt1 = "./data/checkpoints/seg.trt"
    path_trt2 = "./data/checkpoints/vec.trt"
    cObjPose = GetObjPose(path_trt1, path_trt2)

    total = 0
    list_imgs_bgr = []
    for dir_path, dir_names, file_names in os.walk(args.source):
        for name in file_names:
            if name.endswith(".png") and "rgb" in dir_path:
                if True: # total % 10 == 0:
                    path_bgr = os.path.join(dir_path, name)
                    list_imgs_bgr.append(path_bgr)
                    total += 1

    for path_bgr in list_imgs_bgr:#[::-1]:
        img_bgr = cv2.imread(path_bgr)
        path_dep = path_bgr.replace("rgb", "depth")
        dep16 = cv2.imread(path_dep, -1)
        # Inference
        start_time_est = time.time()
        out_put = cObjPose.Run(img_bgr, dep16)
        end_time_est = time.time()
        time_est = end_time_est - start_time_est
        print("Est time / frame: {:.2f} ms".format(1000 * time_est))

        if cObjPose.flagShow:
            # if cObjPose.plane is not None:
            #     _, img_desk = cObjPose.draw_desk(dep16)
            # cv2.imshow("desk", img_desk)
            # if len(cObjPose.objs) > 0:
            #     for obj in cObjPose.objs:
            #         cls_ = obj.cls
            #         mask = obj.mask
            #         conf = obj.conf
            #         bbox = obj.bbox
            #         mask_ch3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            #         mask_alpha = mask_ch3 * list_color[cls_]
            #         plus = np.where(mask_ch3, mask_alpha, img_bgr)
            #         img_bgr = cv2.addWeighted(img_bgr.astype(np.float32), 0.6, plus.astype(np.float32), 0.4, 0).astype(np.uint8)
            #         cv2.circle(img_bgr, obj.coord2d_cen, 2, (0,0,0), -1)
            #         cv2.putText(img_bgr, f"{cObjPose.segger.classes[cls_]}: {conf:.3f}", bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
            #         print(cls_, obj.coord3d_cen)
            img_rst, img_desk = cObjPose.draw(img_bgr, dep16)
            cv2.imshow("rst", img_rst)
            cv2.imshow("desk", img_desk)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break

