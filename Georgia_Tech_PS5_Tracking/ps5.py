"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        #dt, state, covariance
        self.state = np.array([init_x, init_y, 0., 0.]).reshape(4,1)  # state
        self.covariance = np.matrix(np.eye(4, dtype=float))
        self.dt = 1
        #create noise matrices
        self.process_noise = np.matrix(Q, dtype=float)
        self.measurement_noise = np.matrix(R, dtype=float)
        #transition, measurement matrices
        self.transition_matrix = np.matrix([
                                            [1., 0., self.dt, 0.],
                                            [0., 1.,    0.,   self.dt],
                                            [0., 0.,    1.,   0.],
                                            [0., 0.,    0.,   1.],
                                                                   ])
        self.measurement_matrix = np.matrix([
                                             [1., 0., 0., 0.],
                                             [0., 1., 0., 0.],
                                                                ])
    def predict(self):
        self.state = np.dot(self.transition_matrix, self.state)
        self.covariance = np.dot(self.transition_matrix, np.dot(self.covariance, np.transpose(self.transition_matrix))) + self.process_noise

    def correct(self, meas_x, meas_y):
        diff = np.array([meas_x, meas_y]).reshape(2, 1) - np.dot(self.measurement_matrix, self.state)
        residual_covariance = np.dot(np.dot(self.measurement_matrix, self.covariance), np.transpose(self.measurement_matrix)) + self.measurement_noise
        kalman_gain = np.dot(self.covariance, np.dot(self.measurement_matrix.transpose(), np.linalg.inv(residual_covariance)))

        self.state = self.state + kalman_gain*diff

        identity_mat = np.matrix(np.eye(self.state.shape[0]))

        self.covariance = np.dot((identity_mat - np.dot(kalman_gain, self.measurement_matrix)), self.covariance)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_mse (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder

        if template.shape[0] % 2 != 0:
            template = template[:-1, :]
        if template.shape[1] % 2 != 0:
            template = template[:, :-1]

        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.frame = frame
        self.state = None #will update after first go
        self.particles = None #We have a better method for initializing our self.particles
        self.weights = np.ones(self.num_particles, dtype=float)/self.num_particles

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """

        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_cutout(self, frame, center):
        '''
        Returns the frame of points we think are akin to the template


        :return:
        '''



        temp_cutout = np.zeros((self.template.shape[0], self.template.shape[1]), dtype=float)

        frame_height, frame_width = frame.shape[0], frame.shape[1]
        temp_height, temp_width = self.template.shape[:2]

        center = center.astype(int)

        fr_start_x, fr_end_x = center[0] - temp_width/2, center[0] + temp_width/2
        fr_start_y, fr_end_y = center[1] - temp_height/2, center[1] + temp_height/2

        x_start = 0
        x_end = temp_width
        y_start = 0
        y_end = temp_height


        if fr_start_x < 0:
            # print 'x less than 0'
            x_start = abs(fr_start_x)
            fr_start_x = 0

        elif fr_end_x > frame_width:
            # print 'x greater than frame width'
            fr_end_x = frame_width
            x_end = frame_width - fr_start_x

        if fr_start_y < 0:
            # print 'y less than 0'
            y_start = abs(fr_start_y)
            fr_start_y = 0

        elif fr_end_y > frame_height:
            # print 'y greater than frame height'
            fr_end_y = frame_height
            y_end = frame_height - fr_start_y

        print frame.shape
        print fr_start_x, fr_end_x
        if fr_start_x > frame.shape[1]:
            return temp_cutout
        elif fr_end_x > frame.shape[1]:
            return temp_cutout

        temp_cutout[y_start:y_end, x_start:x_end] = frame[fr_start_y:fr_end_y, fr_start_x:fr_end_x]

        return temp_cutout


    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """

        float_template = template.astype(float)
        frame_cutout = frame_cutout.astype(float)
        y, x = float_template.shape[0], float_template.shape[1]

        #for each pixel in temp vs frame, we calulate diff. then square, then sum over each point
        mse = np.sum((float_template - frame_cutout)**2)/(y*x)
        return mse

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        return np.random.choice(a=self.num_particles, size=self.num_particles, p=self.weights)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
         """

        if self.state is None:
            #Use initial center position to make smarter predictions
            x = self.template_rect['x'] + np.floor(self.template_rect['w'] / 2.)
            y = self.template_rect['y'] + np.floor(self.template_rect['h'] / 2.)
        else:
            x, y = self.state

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #This is going to develop a random normal distribution of x and y given our dynamic sigma value

        rand_norm_x = np.random.normal(loc=x, scale=self.sigma_dyn, size=self.num_particles)\
            .reshape(self.num_particles, 1)
        rand_norm_y = np.random.normal(loc=y, scale=self.sigma_dyn, size=self.num_particles)\
            .reshape(self.num_particles, 1)

        self.particles = np.concatenate((rand_norm_x, rand_norm_y), axis=1)

        # print self.particles
        i = 0
        for p in self.particles:
            frame_cutout = self.get_cutout(gray_frame, p)
            mse = self.get_error_metric(self.template, frame_cutout)
            # print 'MSE is {}'.format(mse)
            self.weights[i] = np.exp(-mse / (2. * self.sigma_exp ** 2.))
            i += 1

        self.weights = self.weights/sum(self.weights)
        self.particles = self.particles[self.resample_particles()] #updates resampled particles
        self.state = np.average(self.particles, axis=0, weights=self.weights).astype(np.int)


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_mean, y_mean = means = self.state


        ##Taken from render function from piazza --> Sumeet Arnand
        # draw dots
        for particle in self.particles:
            cv2.circle(frame_in, tuple(particle.astype(int)), radius=1, color=(0, 0, 255), thickness=-1)

        # draw rectangle
        h, w = self.template.shape[:2]
        cv2.rectangle(frame_in, (x_mean - w / 2, y_mean - h / 2), (x_mean + w / 2, y_mean + h / 2),
                      color=(192, 192, 192), thickness=1)

        # draw circle
        particles_dist = ((self.particles - means) ** 2).sum(axis=1) ** 0.5
        radius = np.average(particles_dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, tuple(means), radius=radius, color=(192, 192, 192), thickness=1)


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder

        # cv2.imshow('template', template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)
        gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best_frame = self.get_cutout(gray_fr, self.state)
        temp_t = self.alpha*best_frame + (1-self.alpha)*self.template

        # cv2.imshow('template before', self.template)
        # cv2.waitKey(0)
        self.template = temp_t.astype(np.uint8)

        # cv2.imshow('template after', self.template)
        # cv2.waitKey(0)

class MDParticleFilter(ParticleFilter):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor

        #pos change references the position change of the last frame to the current
        self.pos_change = (0, 0)
        self.mse_change = 0
        # self.scale = None
        x = self.template_rect['x'] + np.floor(self.template_rect['w'] / 2.)
        y = self.template_rect['y'] + np.floor(self.template_rect['h'] / 2.)
        self.mse = 50
        self.scale = .994983133
        self.state = (int(x), int(y))

        # c = (0, 0)
        #
        # #c represents the expected position change in (x, y) format
        # self.state = (x, y, c)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        x, y = self.state[0], self.state[1]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #This is going to develop a random normal distribution of x and y given our dynamic sigma value
        rand_norm_x = np.random.normal(loc=x, scale=self.sigma_dyn, size=self.num_particles)\
            .reshape(self.num_particles, 1)
        rand_norm_y = np.random.normal(loc=y, scale=self.sigma_dyn, size=self.num_particles)\
            .reshape(self.num_particles, 1)

        self.particles = np.concatenate((rand_norm_x, rand_norm_y), axis=1)

        i = 0
        mse_array = []
        for p in self.particles:

            resized_temp = cv2.resize(src=self.template.copy(), dsize=(0, 0), fx=self.scale, fy=self.scale).astype(np.uint8)
            frame_cutout = self.get_cutout(gray_frame, p)
            resized_frame_cut = cv2.resize(src=frame_cutout, dsize=(0, 0), fx=self.scale, fy=self.scale).astype(np.uint8)
            mse = self.get_error_metric(resized_temp, resized_frame_cut)
            mse_array.append(mse)
            self.weights[i] = np.exp(-mse / (2. * self.sigma_exp ** 2.))
            i += 1

        self.weights = self.weights / sum(self.weights)
        self.particles = self.particles[self.resample_particles()]  # updates resampled particles

        current_avg_mse = np.average(mse_array)

        print 'Current Avg MSE {}'.format(current_avg_mse)
        if current_avg_mse > 7100.:
            print 'We have Occlusion'
            # print 'current avg mse is {}'.format(current_avg_mse)
            # print 'previous mse was {}'.format(self.mse)
            # print 'mse_change is {}'.format(self.mse_change)

            pass
            # self.state = (int(self.state[0] + self.pos_change[0]), int(self.state[1] + self.pos_change[1]))

        else:
            # print 'NO OCCLUSION'
            # print 'current avg mse is {}'.format(current_avg_mse)
            # print 'previous mse was {}'.format(self.mse)
            # print 'mse_change is {}'.format(self.mse_change)
            new_pos = np.average(self.particles, axis=0, weights=self.weights)
            self.pos_change = tuple(new_pos - self.state)
            print 'pos change is {}'.format(self.pos_change)
            self.state = np.average(self.particles, axis=0, weights=self.weights).astype(np.int)

        self.mse = current_avg_mse

        # print 'Scale from {} '.format(self.scale)
        self.scale = self.scale*.994983133
        # print 'To {}'.format(self.scale)


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_mean, y_mean = means = self.state


        #Taken from render function from piazza --> Sumeet Arnand
        # draw dots
        for particle in self.particles:
            cv2.circle(frame_in, tuple(particle.astype(int)), radius=1, color=(0, 0, 255), thickness=-1)

        # draw rectangle
        h, w = self.template.shape[:2]
        cv2.rectangle(frame_in, ((x_mean - w // 2), (y_mean - h // 2)), ((x_mean + w // 2), (y_mean +h // 2)),
                      color=(192, 192, 192), thickness=1)

        # draw circle
        particles_dist = ((self.particles - means) ** 2).sum(axis=1) ** 0.5
        radius = np.average(particles_dist, axis=0, weights=self.weights).astype(np.int)
        cv2.circle(frame_in, tuple(means), radius=radius, color=(0, 0, 255), thickness=5)