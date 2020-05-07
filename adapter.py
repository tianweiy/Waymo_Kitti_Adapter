import argparse
import os
import math
# import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import progressbar

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import json 

# max indexing length
INDEX_LENGTH = 15
# PATH 
DATA_PATH = None 
CONVERT_PATH = None 
LABEL_ALL_PATH = None 
LIDAR_PATH = None 

class Adapter:
    def __init__(self):
        self.__lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']
        self.__type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

        self.get_file_names()
        self.create_folder()

    def cvt(self, islidar=False, islabel=False):
        """ convert dataset from Waymo to KITTI
        Args:
        return:
        """
        bar = progressbar.ProgressBar(maxval=len(self.__file_names)+1,
                    widgets= [progressbar.Percentage(), ' ',
                    progressbar.Bar(marker='>',left='[',right=']'),' ',
                    progressbar.ETA()])

        annos = {}

        tf.enable_eager_execution()
        file_num = 1
        frame_num = 0  # frame_num serve as the unique identifier to correspond lidar point with labels
        print("start converting ...")
        bar.start()
        for seq_id, file_name in enumerate(self.__file_names):
            dataset = tf.data.TFRecordDataset(file_name, compression_type='')
            for frame_id, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                
                # parse lidar
                if islidar:
                    self.save_lidar(frame, frame_num)

                # parse label
                if islabel:
                    annos.update({str(frame_num).zfill(INDEX_LENGTH) : self.save_label(frame, frame_id, seq_id, frame_num)})

                frame_num += 1

            bar.update(file_num)
            file_num += 1
        bar.finish()
        print("\nfinished ...")

        if islabel:
            with open(LABEL_ALL_PATH, "w") as f:
                json.dump(annos, f)

    def save_lidar(self, frame, frame_num):
        """ parse and save the lidar data in psd format
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :return:
                """
        range_images, range_image_top_pose = self.parse_range_image_and_camera_projection(
            frame)

        points, intensity = self.convert_range_image_to_point_cloud(
            frame,
            range_images,
            range_image_top_pose)

        points_all = np.concatenate(points, axis=0)
        intensity_all = np.concatenate(intensity, axis=0)
        point_cloud = np.column_stack((points_all, intensity_all))

        pc_path = LIDAR_PATH + '/' + str(frame_num).zfill(INDEX_LENGTH) + '.bin'
        point_cloud.tofile(pc_path)

    def save_label(self, frame, frame_id, seq_id, lidar_id):
        """ parse and save the label data in .txt format
                :param frame: open dataset frame proto
                :param frame_id: identifier for current frame in the sequence (used for tracking)
                :param seq_id: identifier for current sequence
                :param lidar_id: same value as frame_num, use this to identify the ldiar point cloud 
                :return:
                """
        results = []      
        metadata = {}

        frame_timestamp = frame.timestamp_micros
        frame_pose = frame.pose.transform
        context_name = frame.context.name

        metadata = {
            'frame_timestamp': frame_timestamp,
            'context_name': context_name,
            'frame_pose': frame_pose,
            'frame_id': frame_id,
            'seq_id': seq_id,
            'lidar_id': str(lidar_id).zfill(INDEX_LENGTH)
        }

        for obj in frame.laser_labels:
            my_type = self.__type_list[obj.type]
            height = obj.box.height
            width = obj.box.width
            length = obj.box.length
            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z
            rotation_y = obj.box.heading

            speed_x = obj.metadata.speed_x
            speed_y = obj.metadata.speed_y
            acc_x = obj.metadata.accel_x
            acc_y = obj.metadata.accel_y
            num_point_in_gt = obj.num_lidar_points_in_box

            target = {
                'type': my_type,
                'dim': [width, length, height],
                'rotation': rotation_y,
                'location': [x,y,z],
                'velocity': [speed_x, speed_y],
                'acceleration': [acc_x, acc_y],
                'frame_pose': list(frame_pose),
                'num_points_in_gt': num_point_in_gt,
            }
            results.append(target)

        annos = {}
        annos['results'] = results
        annos['metadata'] = metadata

        return annos 

    def get_file_names(self):
        self.__file_names = []
        for i in os.listdir(DATA_PATH):
            if i.split('.')[-1] == 'tfrecord':
                self.__file_names.append(DATA_PATH + '/' + i)

    def create_folder(self):
        if not os.path.exists(LIDAR_PATH):
            os.mkdir(LIDAR_PATH)

    def extract_intensity(self, frame, range_images, lidar_num):
        """ extract the intensity from the original range image
                :param frame: open dataset frame proto
                :param frame_num: the current frame number
                :param lidar_num: the number of current lidar
                :return:
                """
        intensity_0 = np.array(range_images[lidar_num][0].data).reshape(-1,4)
        intensity_0 = intensity_0[:,1]
        intensity_1 = np.array(range_images[lidar_num][1].data).reshape(-1,4)[:,1]
        return intensity_0, intensity_1

    def image_show(self, data, name, layout, cmap=None):
        """Show an image."""
        plt.subplot(*layout)
        plt.imshow(tf.image.decode_jpeg(data), cmap=cmap)
        plt.title(name)
        plt.grid(False)
        plt.axis('off')

    def parse_range_image_and_camera_projection(self, frame):
        """Parse range images and camera projections given a frame.
        Args:
           frame: open dataset frame proto
        Returns:
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
        """
        self.__range_images = {}
        # camera_projections = {}
        # range_image_top_pose = None
        for laser in frame.lasers:
            if len(laser.ri_return1.range_image_compressed) > 0:
                range_image_str_tensor = tf.decode_compressed(
                    laser.ri_return1.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name] = [ri]

                if laser.name == open_dataset.LaserName.TOP:
                    range_image_top_pose_str_tensor = tf.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                    range_image_top_pose = open_dataset.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        bytearray(range_image_top_pose_str_tensor.numpy()))

            if len(laser.ri_return2.range_image_compressed) > 0:
                range_image_str_tensor = tf.decode_compressed(
                    laser.ri_return2.range_image_compressed, 'ZLIB')
                ri = open_dataset.MatrixFloat()
                ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
                self.__range_images[laser.name].append(ri)

        return self.__range_images, range_image_top_pose

    def plot_range_image_helper(self, data, name, layout, vmin=0, vmax=1, cmap='gray'):
        """Plots range image.
        Args:
          data: range image data
          name: the image title
          layout: plt layout
          vmin: minimum value of the passed data
          vmax: maximum value of the passed data
          cmap: color map
        """
        plt.subplot(*layout)
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(name)
        plt.grid(False)
        plt.axis('off')

    def get_range_image(self, laser_name, return_index):
        """Returns range image given a laser name and its return index."""
        return self.__range_images[laser_name][return_index]

    def show_range_image(self, range_image, layout_index_start=1):
        """Shows range image.
        Args:
          range_image: the range image data from a given lidar of type MatrixFloat.
          layout_index_start: layout offset
        """
        range_image_tensor = tf.convert_to_tensor(range_image.data)
        range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
        lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
        range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                      tf.ones_like(range_image_tensor) * 1e10)
        range_image_range = range_image_tensor[..., 0]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        self.plot_range_image_helper(range_image_range.numpy(), 'range',
                                [8, 1, layout_index_start], vmax=75, cmap='gray')
        self.plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                                [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
        self.plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                                [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')

    def convert_range_image_to_point_cloud(self, frame, range_images, range_image_top_pose, ri_index=0):
        """Convert range images to point cloud.
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name,
             [range_image_first_return, range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
              camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.
        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
          intensity: {[N, 1]} list of intensity of length 5 (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        intensity = []

        frame_pose = tf.convert_to_tensor(
            np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == open_dataset.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            intensity_tensor = tf.gather_nd(range_image_tensor,
                                         tf.where(range_image_mask))
            points.append(points_tensor.numpy())
            intensity.append(intensity_tensor.numpy()[:, 1])

        return points, intensity

    def rgba(self, r):
        """Generates a color based on range.
        Args:
          r: the range value of a given point.
        Returns:
          The color for a given range
        """
        c = plt.get_cmap('jet')((r % 20.0) / 20.0)
        c = list(c)
        c[-1] = 0.5  # alpha
        return c

    def plot_image(self, camera_image):
        """Plot a cmaera image."""
        plt.figure(figsize=(20, 12))
        plt.imshow(tf.image.decode_jpeg(camera_image.image))
        plt.grid("off")

    def plot_points_on_image(self, projected_points, camera_image, rgba_func, point_size=5.0):
        """Plots points on a camera image.
        Args:
          projected_points: [N, 3] numpy array. The inner dims are
            [camera_x, camera_y, range].
          camera_image: jpeg encoded camera image.
          rgba_func: a function that generates a color from a range value.
          point_size: the point size.
        """
        self.plot_image(camera_image)

        xs = []
        ys = []
        colors = []

        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colors.append(rgba_func(point[2]))

        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")


def parse_args():
    parser = argparse.ArgumentParser(description="Waymo 3D Extractor")
    parser.add_argument("--data_path", type=str, default="data/Waymo/tfrecord_training")
    parser.add_argument("--convert_path", type=str, default="data/Waymo_convert")
    parser.add_argument("--is_lidar", action='store_true')
    parser.add_argument("--is_label", action='store_true')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    DATA_PATH = args.data_path
    CONVERT_PATH = args.convert_path 

    LABEL_ALL_PATH = CONVERT_PATH + '/target_info.json'
    LIDAR_PATH = CONVERT_PATH + '/lidar'

    print("LIDAR SAVE TO ", LIDAR_PATH)
    print("LABEL SAVE TO ", LABEL_ALL_PATH)
    print("CONVERT LIDAR ", args.is_lidar)
    print("CONVERT LABEL ", args.is_label)


    adapter = Adapter()
    adapter.cvt(args.is_lidar, args.is_label)
