# Optics formulas
import math


class Sensor:

    def __init__(self, h_pixels, v_pixels, pixel_size_um):
        self.__h_pixels = h_pixels    # Number of horizontal pixels
        self.__v_pixels = v_pixels    # Number of vertical pixels
        self.__pixel_size_um = pixel_size_um  # Pixel Size in um
        return

    def resolution_lp_mm(self):
        return 1000.0 / (2 * self.__pixel_size_um)

    def size_mm(self):
        h_mm = self.__pixel_size_um * self.__h_pixels / 1000.0
        v_mm = self.__pixel_size_um * self.__v_pixels / 1000.0
        d_mm = math.sqrt(h_mm * h_mm + v_mm * v_mm)
        return dict([('h_mm', h_mm),
                     ('v_mm', v_mm),
                     ('d_mm', d_mm)
                     ])

    def pixel_size_um(self):
        return self.__pixel_size_um

    def size_pixels(self):
        return dict([('h_pixels', self.__h_pixels),
                     ('v_pixels', self.__v_pixels)
                     ])


class Lens:

    def __init__(self, f_num, fl_mm):
        self.fl_mm = fl_mm
        self.f_num = f_num
        return

    def focal_length_mm(self):
        return self.fl_mm

    def hyper_focal_dist_mm(self, coc_mm=0.030):
        return self.fl_mm * self.fl_mm / self.f_num / coc_mm + self.fl_mm

    def depth_of_field_mm(self, object_dist_mm, coc_mm=0.030):
        H_mm = self.hyper_focal_dist_mm(coc_mm)
        DoFn = object_dist_mm * (H_mm - self.fl_mm) / (H_mm + object_dist_mm - 2 * self.fl_mm)
        if object_dist_mm >= H_mm:
            DoFf = object_dist_mm * (H_mm - self.fl_mm) / 1.0e-12
        else:
            DoFf = object_dist_mm * (H_mm - self.fl_mm) / (H_mm - object_dist_mm )
        return dict([('DoFn_mm', DoFn),
                     ('DoFf_mm', DoFf)])

    def focus_position_mm(self, object_dist_mm):
        return 1.0 / (1.0 / self.fl_mm - 1.0 / object_dist_mm)

    def focus_position_delta_um(self, object_dist_mm):
        return self.focus_position_mm(object_dist_mm) - self.fl_mm


class Optical:

    def __init__(self, h_pixels, v_pixels, pixel_size_um, f_num, fl_mm):
        self.lens = Lens(f_num, fl_mm)
        self.sensor = Sensor(h_pixels, v_pixels, pixel_size_um)
        return

    def angular_fov(self):
        ss = self.sensor.size_mm()
        hfov = 180.0 / math.pi * 2.0 * math.atan(ss['h_mm'] / (2.0 * self.lens.focal_length_mm()))
        vfov = 180.0 / math.pi * 2.0 * math.atan(ss['v_mm'] / (2.0 * self.lens.focal_length_mm()))
        d_mm = math.sqrt(ss['h_mm'] * ss['h_mm'] + ss['v_mm'] * ss['v_mm'])
        dfov = 180.0 / math.pi * 2.0 * math.atan(d_mm / (2.0 * self.lens.focal_length_mm()))
        return dict([('hfov_deg', hfov),
                     ('vfov_deg', vfov),
                     ('dfov_deg', dfov)
                     ])

    def focal_length_35mm(self):
        # Assumes 4:3
        h_35mm = 36
        v_35mm = 24
        ss = self.sensor.size_mm()
        return h_35mm * self.lens.focal_length_mm() / ss['h_mm']

    def object_space_resolution(self, object_dist_mm):
        # Compute the magnification of the lens for an oject at d
        mag = -1.0 * self.lens.fl_mm / (self.lens.fl_mm - object_dist_mm)
        #print("{}".format(mag))

        # for a sensor pixel, what is the object size
        obj_pixel_size_mm = self.sensor.pixel_size_um() / mag / 1000.0
        #print("{}".format(obj_pixel_size_mm))

        # compute the lp/mm at a distance
        obj_lp_mm = 0.5 / obj_pixel_size_mm

        # Compute the scene size
        ss = self.sensor.size_pixels()
        scene_h = ss['h_pixels'] * obj_pixel_size_mm
        scene_v = ss['v_pixels'] * obj_pixel_size_mm

        return dict([('lp_mm', obj_lp_mm), ('pixel_mm', obj_pixel_size_mm),
                     ('h_mm', scene_h), ('v_mm', scene_v)])