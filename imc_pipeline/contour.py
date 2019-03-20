import numpy as np

import cv2


class Contour:
    """[summary]

    Parameters
    ----------
    cnt : [type]
        [description]
    """

    # Provides detailed parameter informations about a contour

    # Create a Contour instant as follows: c = Contour(contour)
    #         where src_img should be grayscale image.

    # Attributes:

    # c.area -- gives the area of the region
    # c.parameter -- gives the perimeter of the region
    # c.moments -- gives all values of moments as a dict
    # c.centroid -- gives the centroid of the region as a tuple (x,y)
    # c.bounding_box -- gives the bounding box parameters as a tuple => (x,y,width,height)
    # c.bx,c.by,c.bw,c.bh -- corresponds to (x,y,width,height) of the bounding box
    # c.aspect_ratio -- aspect ratio is the ratio of width to height
    # c.equi_diameter -- equivalent diameter of the circle with same as area as that of region
    # c.extent -- extent = contour area/bounding box area
    # c.convex_hull -- gives the convex hull of the region
    # c.convex_area -- gives the area of the convex hull
    # c.solidity -- solidity = contour area / convex hull area
    # c.center -- gives the center of the ellipse
    # c.majoraxis_length -- gives the length of major axis
    # c.minoraxis_length -- gives the length of minor axis
    # c.orientation -- gives the orientation of ellipse
    # c.eccentricity -- gives the eccentricity of ellipse

    def __init__(self, cnt):
        """[summary]

        Parameters
        ----------
        cnt : [type]
            [description]
        """

        self.cnt = cnt
        self.size = len(cnt)

        # MAIN PARAMETERS

        # Contour.area - Area bounded by the contour region'''
        self.area = cv2.contourArea(self.cnt)

        # contour perimeter
        self.perimeter = cv2.arcLength(cnt, True)

        # centroid
        self.moments = cv2.moments(cnt)
        if self.moments['m00'] != 0.0:
            self.cx = self.moments['m10'] / self.moments['m00']
            self.cy = self.moments['m01'] / self.moments['m00']
            self.centroid = (self.cx, self.cy)
        else:
            self.centroid = 'Region has zero area'

        # bounding box
        self.bounding_box = cv2.boundingRect(cnt)
        (self.bx, self.by, self.bw, self.bh) = self.bounding_box

        # aspect ratio
        self.aspect_ratio = self.bw / float(self.bh)

        # equivalent diameter
        self.equi_diameter = np.sqrt(4 * self.area / np.pi)

        # extent = contour area/boundingrect area
        self.extent = self.area / (self.bw * self.bh)

        # Minimum Enclosing Circle
        (
            self.x_mincircle,
            self.y_mincircle,
        ), self.radius_mincircle = cv2.minEnclosingCircle(cnt)
        self.center_mincircle = (int(self.x_mincircle), int(self.y_mincircle))
        self.mincircle_area = np.pi * self.radius_mincircle * self.radius_mincircle

        # Checking Convexity
        self.convexity = cv2.isContourConvex(cnt)

        # convex hull
        self.convex_hull = cv2.convexHull(cnt)

        # convex hull area
        self.convex_area = cv2.contourArea(self.convex_hull)

        # solidity = contour area / convex hull area
        if self.convex_area != 0:
            self.solidity = self.area / float(self.convex_area)
        else:
            self.solidity = 0

        # ellipse
        self.ellipse = cv2.fitEllipse(cnt)

        # center, axis_length and orientation of ellipse
        (self.center, self.axes, self.orientation) = self.ellipse

        # length of MAJOR and minor axis
        self.majoraxis_length = max(self.axes)
        self.minoraxis_length = min(self.axes)
        self.area_ellipse = (
            np.pi * (self.majoraxis_length / 2.0) * (self.minoraxis_length / 2.0)
        )

        # rotation angle
        self.rotation_angle = self.orientation

        # eccentricity = sqrt( 1 - (ma/MA)^2) --- ma= minor axis --- MA= major axis
        self.eccentricity = np.sqrt(
            1 - (self.minoraxis_length / self.majoraxis_length) ** 2
        )
