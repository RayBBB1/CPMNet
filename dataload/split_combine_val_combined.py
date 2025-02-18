from typing import List, Tuple
import numpy as np


class SplitComb():
    def __init__(self, crop_size: List[int]=[96, 96, 96], overlap_size:List[int]=[32, 32, 32], pad_value:float=0):
        self.crop_size = np.array(crop_size) // 4
        self.stride_size = [crop_size[0]-overlap_size[0], 
                            crop_size[1]-overlap_size[1], 
                            crop_size[2]-overlap_size[2]]
        self.overlap = overlap_size
        self.pad_value = pad_value

    def split(self, data):
        splits = []
        d, h, w = data.shape

        # Number of splits in each dimension
        nz = int(np.ceil(float(d) / self.stride_size[0]))
        ny = int(np.ceil(float(h) / self.stride_size[1]))
        nx = int(np.ceil(float(w) / self.stride_size[2]))

        nzyx = [nz, ny, nx]
        pad = [[0, int(nz * self.stride_size[0] + self.overlap[0] - d)],
                [0, int(ny * self.stride_size[1] + self.overlap[1] - h)],
                [0, int(nx * self.stride_size[2] + self.overlap[2] - w)]]

        data = np.pad(data, pad, 'constant', constant_values=self.pad_value)  

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    start_z = int(iz * self.stride_size[0])
                    end_z = int((iz + 1) * self.stride_size[0] + self.overlap[0])
                    start_y = int(iy * self.stride_size[1])
                    end_y = int((iy + 1) * self.stride_size[1] + self.overlap[1])
                    start_x = int(ix * self.stride_size[2])
                    end_x = int((ix + 1) * self.stride_size[2] + self.overlap[2])

                    split = data[np.newaxis, np.newaxis, start_z:end_z, start_y:end_y, start_x:end_x]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzyx, pad

    def combine(self, outputs, start_idx: int, nzhw, image_shape: Tuple[int, int, int], pad: List[List[int]]):
        nz, nh, nw = nzhw
        idx = 0
        
        padded_shape = [image_shape[0] + pad[0][0] + pad[0][1],
                        image_shape[1] + pad[1][0] + pad[1][1],
                        image_shape[2] + pad[2][0] + pad[2][1]]
        padded_shape = np.array(padded_shape) // 4
        Cls_feats = np.zeros(padded_shape, dtype=np.float32) # [crop_size[0], crop_size[1], crop_size[2]]
        Max_Cls_feats = np.ones(padded_shape, dtype=np.float32) * (-np.inf)
        Shape_feats = np.zeros([3] + list(padded_shape), dtype=np.float32) # [3, crop_size[0], crop_size[1], crop_size[2]]
        Offset_feats = np.zeros([3] + list(padded_shape), dtype=np.float32) # [3, crop_size[0], crop_size[1], crop_size[2]
        Cls_counts = np.zeros(padded_shape, dtype=np.float32)
        
        idx = start_idx
        for iz in range(nz): 
            for ih in range(nh):
                for iw in range(nw):
                    sz = int(iz * self.stride_size[0] / 4)
                    sh = int(ih * self.stride_size[1] / 4)
                    sw = int(iw * self.stride_size[2] / 4)
                    # Key: 'Cls', 'Shape', 'Offset'
                    Cls = outputs['Cls'][idx].squeeze() # [1, crop_size[0], crop_size[1], crop_size[2]]
                    Shape = outputs['Shape'][idx] # [3, crop_size[0], crop_size[1], crop_size[2]]
                    Offset = outputs['Offset'][idx] # [3, crop_size[0], crop_size[1], crop_size[2]]
                    
                    Cls_feats[sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]] += Cls
                    Cls_counts[sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]] += 1
                    
                    # Shape_feats[:, sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]] = Shape
                    # Offset_feats[:, sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]] = Offset
                    # Combine Shape and Offset
                    crop_max_cls_feats = Max_Cls_feats[sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]]
                    gt_mask = (Cls > crop_max_cls_feats)
                    gt_mask = np.repeat(gt_mask[np.newaxis, ...], 3, axis=0)
                    
                    Shape_feats[:, sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]][gt_mask] = Shape[gt_mask]
                    Offset_feats[:, sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]][gt_mask] = Offset[gt_mask]
                    
                    # Update Max_Cls_feats
                    Max_Cls_feats[sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]] = np.maximum(Max_Cls_feats[sz:sz+self.crop_size[0], sh:sh+self.crop_size[1], sw:sw+self.crop_size[2]], Cls)
                    idx += 1
        
        # np.save('Cls.npy', Cls_feats)
        # np.save('Shape.npy', Shape_feats)
        # np.save('Offset.npy', Offset_feats)
        # np.save('Cls_counts.npy', Cls_counts)
        # np.save('Max_Cls.npy', Max_Cls_feats)
        # raise ValueError('Stop here')
        
        Cls_feats = Cls_feats / np.maximum(Cls_counts, 1)
        print(np.percentile(Cls_feats, 95), np.percentile(Cls_feats, 85))
        Cls_feats = Cls_feats[:image_shape[0], :image_shape[1], :image_shape[2]]
        Shape_feats = Shape_feats[:, :image_shape[0], :image_shape[1], :image_shape[2]]
        Offset_feats = Offset_feats[:, :image_shape[0], :image_shape[1], :image_shape[2]]
        
        return {'Cls': np.expand_dims(Cls_feats, axis=[0, 1]),
                'Shape': np.expand_dims(Shape_feats, axis=[0]),
                'Offset': np.expand_dims(Offset_feats, axis=[0])}